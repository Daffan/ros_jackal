#!/bin/bash
export ROS_LOG_DIR=/tmp
for i in {1..5}
do
   ./singularity_run.sh python3 habitat_eval.py --habitat_index=${@:1} --seed=${i} --save=${@:2} --applr=${@:3} 
done