#!/bin/bash
export ROS_LOG_DIR=/tmp
cmd="./singularity_run.sh python3 habitat_eval.py --habitat_index=$1 --seed=$2 --save=$3 --applr=$4"
echo ${cmd}
exec ${cmd}
