#!/bin/bash
# export PATH=$PATH:/lusr/opt/singularity-3.2.1/bin
# export ROS_HOSTNAME=localhost
# export ROS_MASTER_URI=http://localhost:11311
# Here we set log_dir to /tmp to avoid 
# tons of log file saved tons of log file saved to the user homedir
export ROS_LOG_DIR=/tmp

./singularity_run_cluster.sh python3 actor.py --id ${@:1}
