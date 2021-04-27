################################################################################
# The script generates submission files and submit them to HTCondor.
# Submission file only has executable/actor.sh
# We now prefer to run central node locally, because central learner on the 
# computing node is not stable. Idle node will destory the central learner
################################################################################

import subprocess
import yaml
import os

# Load condor config
CONFIG_PATH = "td3/config.yaml"
with open(CONFIG_PATH, 'r') as f:
    config = yaml.load(f, Loader=yaml.FullLoader)
num_actor = config["condor_config"]["num_actor"]

if not os.path.exists('out'):
    os.mkdir('out')

cfile = open('actors.sub', 'w')
s = 'executable/actor.sh'
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

# Loop over various values of an argument and create different output file for each
# Then put them in the queue
s = "out"
for a in range(num_actor):
    run_command = "\
        arguments  = %d\n\
        output     = %s/out_%d.txt\n\
        log        = %s/log_%d.txt\n\
        error      = %s/err_%d.txt\n\
        queue 1\n\n" % (a, s, a, s, a, s, a)
    cfile.write(run_command)
cfile.close()

subprocess.run(["condor_submit", "actors.sub"])