import os
from typing import DefaultDict
import yaml
import uuid
import time
import shutil
import signal
import subprocess
from os.path import join, exists

import htcondor  # for submitting jobs, querying HTCondor daemons, etc.
import classad   # for interacting with ClassAds, HTCondor's internal data format

# The JobStatus is an integer; the integers map into the following states:
#  - 1: Idle (I) 
#  - 2: Running (R) 
#  - 3: Removed (X)
#  - 4: Completed (C)
#  - 5: Held (H)
#  - 6: Transferring Output
#  - 7: Suspended

class CondorJob(object):
    def __init__(self, exe, arguments):
        self.exe = exe
        self.arguments = arguments
        self.cluster = self.submit(exe, arguments)
        self.schedd = htcondor.Schedd()

    def submit(self, exe, arguments):
        print("Submitting job %d" %arguments, end="\r")
        log_name = exe.replace("/", "-").replace(".", "-") + "-" + str(arguments).split(" ")[-1]
        BUFFER_PATH = os.getenv("BUFFER_PATH")
        submission_file = os.path.join(BUFFER_PATH, 'actors.sub')
        if not os.path.exists(join(BUFFER_PATH, "out")):
            os.mkdir(join(BUFFER_PATH, "out"))
        cfile = open(submission_file, 'w')
        common_command = "\
            requirements       = InMastodon \n\
            +Group              = \"GRAD\" \n\
            +Project            = \"AI_ROBOTICS\" \n\
            +ProjectDescription = \"Adaptive Planner Parameter Learning From Reinforcement\" \n\
            Executable          = %s \n\
            Universe            = vanilla\n\
            getenv              = true\n\
            transfer_executable = false \n\n" %(exe)
        cfile.write(common_command)

        # Add actor arguments
        run_command = "\
            arguments  = %d\n\
            output     = %s\n\
            log        = %s\n\
            error      = %s\n\
            queue 1\n\n" % (
                arguments,
                join(BUFFER_PATH, "out", "out-" + log_name + ".txt"),
                join(BUFFER_PATH, "out", "log-" + log_name + ".txt"),
                join(BUFFER_PATH, "out", "err-" + log_name + ".txt")
            )
        cfile.write(run_command)

        cfile.close()

        out = subprocess.run(["condor_submit", submission_file], stdout=subprocess.PIPE)
        return str(out.stdout).split("to cluster ")[-1].split(".")[0]

    def check_job_status(self):
        return self.schedd.query(
            constraint='ClusterId =?= {}'.format(self.cluster),
            projection=["ClusterId", "ProcId", "JobStatus", "EnteredCurrentStatus"],
        )

    def recover_job(self):
        # check job status, if it's done or hold, then Vacate the job
        job_status = self.check_job_status()
        if len(job_status) == 0 or job_status[0]["JobStatus"] not in [1, 2, 6]:
            print("Recovering job %d" %self.arguments, end="\r")
            self.Remove()
            self.cluster = self.submit(self.exe, self.arguments)

    def Vacate(self):
        self.schedd.act(htcondor.JobAction.Vacate, f"ClusterId == {self.cluster}")

    def Hold(self):
        self.schedd.act(htcondor.JobAction.Hold, f"ClusterId == {self.cluster}")

    def Remove(self):
        self.schedd.act(htcondor.JobAction.Remove, f"ClusterId == {self.cluster}")

    def Release(self):
        self.schedd.act(htcondor.JobAction.Release, f"ClusterId == {self.cluster}")
        
if __name__ == "__main__":
    import argparse
    import pickle
    import numpy as np
    
    parser = argparse.ArgumentParser(description = 'test on condor cluster')
    parser.add_argument('--model_dir', dest='model_dir', type = str)

    model_dir = parser.parse_args().model_dir

    print(">>>>>>>> Loading the model from %s" %(model_dir))

    # Load condor config
    CONFIG_PATH = join(model_dir, "config.yaml")
    with open(CONFIG_PATH, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    # We test each test world with two actors, so duplicate the lict by a factor of two
    num_actor = len(config["condor_config"]["test_worlds"])
    num_trials = config["condor_config"]["num_trials"]

    # Create buffer folder
    hash_code = uuid.uuid4().hex
    # buffer_path = os.path.join(os.environ['HOME'], hash_code)
    buffer_path = os.path.join("/scratch/cluster/zifan", hash_code)
    os.environ['BUFFER_PATH'] = buffer_path
    if not os.path.exists(buffer_path):
        os.mkdir(buffer_path)

    # Copy the model files
    shutil.copyfile(
        join(model_dir, "config.yaml"), 
        join(buffer_path, "config.yaml")    
    )
    shutil.copyfile(
        join(model_dir, "policy_actor"), 
        join(buffer_path, "policy_actor")
    )
    shutil.copyfile(
        join(model_dir, "policy_noise"), 
        join(buffer_path, "policy_noise")
    )
    # Set the exploration noise to be 0
    with open(join(buffer_path, 'eps.txt'), 'w') as f:
        f.write(str(0))

    # Create folders for HTCondor logging files
    out_path = "out"
    out_path = join(buffer_path, out_path)
    print("Find the logging under path: %s" %(out_path))
    if not os.path.exists(out_path):
        os.mkdir(out_path)
    
    jobs = []
    for i in range(num_actor):
        job = CondorJob('executable/tester.sh', i)
        jobs.append(job)
        
    def handler(signum, frame):
        for i, j in enumerate(jobs):
            print("canceling job %d" %i)
            j.Remove()
        exit(1)
        
    signal.signal(signal.SIGINT, handler)

    tmp_jobs = []
    tmp_ids = []
    ids = list(range(num_actor))
    results = DefaultDict(list)
    while len(jobs) > 0:
        time.sleep(20)
        for i, j in zip(ids, jobs):
            j.recover_job()
            tester_path = join(buffer_path, "actor_%d" %i)
            trajs = [f for f in os.listdir(tester_path) if f.endswith(".pickle")] if exists(tester_path) else []
            if len(trajs) >= num_trials:
                for traj in trajs:
                    p = join(buffer_path, "actor_%d" %i, traj)
                    with open(p, 'rb') as f:
                        traj = pickle.load(f)
                    world = traj[-1][-1]['world']
                    if isinstance(world, str):
                        world = int(world.split("_")[-1].split(".")[0])
                    ep_return = sum([t[2] for t in traj])
                    ep_length = len(traj)
                    success = int(traj[-1][-1]['success'])
                    ep_time = float(traj[-1][-1]['time'])
                    collision = sum([int(s[-1]["collision"]) for s in traj])
                    recovery = float(traj[-1][-1]['recovery'])
                j.Remove()
                results[world].append((ep_return, ep_length, success, ep_time, collision, recovery))
                mean_return = np.mean([t[0] for t in results[world]])
                mean_length = np.mean([t[1] for t in results[world]])
                mean_success = np.mean([t[2] for t in results[world]])
                times = [t[3] for t in results[world] if t[2] > 0]
                mean_time = np.mean(times) if len(times) > 0 else 0
                mean_collision = np.mean([t[4] for t in results[world]])
                mean_recovery = np.mean([t[5] for t in results[world]])
                
                print("finishing world %d: %.2f, %.2f, %.2f, %.2f, %.2f, %.2f" \
                    %(world, mean_return, mean_length, mean_success, mean_time, mean_collision, mean_recovery))
                continue
            tmp_jobs.append(j)
            tmp_ids.append(i)
        jobs = tmp_jobs
        ids = tmp_ids
        tmp_jobs = []
        tmp_ids = []
        
    with open(join(model_dir, "test_results.pickle"), "wb") as f:
        f.dump(results)

    shutil.rmtree(buffer_path)
            