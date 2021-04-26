#!/bin/bash
export PATH=$PATH:/lusr/opt/singularity-3.2.1/bin
export ROS_HOSTNAME=localhost
export ROS_MASTER_URI=http://localhost:11311
export ROS_LOG_DIR=/tmp

IMAGE_PATH=ros_jackal_image.simg
singularity_run.sh python3 td3/actor.py --id ${@:1}