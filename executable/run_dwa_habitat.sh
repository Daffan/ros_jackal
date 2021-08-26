#!/bin/bash
export ROS_LOG_DIR=/tmp
for i in {1..50}
do
   ./singularity_run.sh python3 dwa_habitat.py --habitat_index=${@:1}
done