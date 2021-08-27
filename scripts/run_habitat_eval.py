################################################################################
# The script generates submission files and submit them to HTCondor.
# Submission file only has executable/actor.sh
# We now prefer to run central node locally, because central learner on the 
# computing node is not stable. Idle node will destory the central learner
################################################################################

import subprocess
import yaml
import os
import time
import uuid
import argparse

parser = argparse.ArgumentParser(description = 'run DWA in habitat env and generate images')
parser.add_argument('--n_repeat', dest='n_repeat', default=5)
parser.add_argument('--applr', dest='applr', default="")
parser.add_argument('--save', dest='save', default="test_result.txt")
args = parser.parse_args()

n_repeat = args.n_repeat

# Actor submission

submission_file = os.path.join('habitat_eval.sub')
cfile = open(submission_file, 'w')
s = 'executable/run_habitat_eval.sh'
common_command = "\
    requirements       = InMastodon \n\
    +Group              = \"GRAD\" \n\
    +Project            = \"AI_ROBOTICS\" \n\
    +ProjectDescription = \"Adaptive Planner Parameter Learning From Reinforcement\" \n\
    Executable          = %s \n\
    Universe            = vanilla\n\
    getenv              = true\n\
    transfer_executable = false \n\n" %(s)
cfile.write(common_command)

# Add actor arguments
for _ in range(n_repeat):
    for a in [66, 68, 69, 70, 71]:
        for s in range(5):
            run_command = "\
                arguments  = %d %d %s %s\n\
                queue 1\n\n" % (a, s, args.save, args.applr)
            cfile.write(run_command)
cfile.close()

subprocess.run(["condor_submit", submission_file])
