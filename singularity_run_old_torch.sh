#!/bin/bash
export PATH=$PATH:/lusr/opt/singularity-3.2.1/bin
export ROS_HOSTNAME=localhost
export ROS_MASTER_URI=http://localhost:11311
export IMAGE_PATH=/scratch/cluster/zifan/ros_jackal_image_cv2.sif

singularity exec -i -n --network=none -p -B /var/condor:/var/condor -B `pwd`:/jackal_ws/src/ros_jackal ${IMAGE_PATH} /bin/bash /jackal_ws/src/ros_jackal/entrypoint.sh ${@:1}
