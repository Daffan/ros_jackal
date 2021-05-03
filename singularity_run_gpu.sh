#!/bin/bash
export PATH=$PATH:/lusr/opt/singularity-3.2.1/bin
export IMAGE_PATH=ros_jackal_image.sif
singularity exec -i -n --network=none -p --nv -B `pwd`:/jackal_ws/src/ros_jackal ${IMAGE_PATH} ${@:1}