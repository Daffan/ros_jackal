#!/bin/bash
export PATH=$PATH:/lusr/opt/singularity-3.2.1/bin
export ROS_HOSTNAME=localhost
export ROS_MASTER_URI=http://localhost:11311
export ROS_LOG_DIR=/tmp
export BUFFER_PATH=${2}
echo ${1}
./singularity_run.sh python3 td3/train.py --config_path ${1}
