#!/bin/bash
export ROS_HOSTNAME=localhost
export ROS_MASTER_URI=http://localhost:11311

singularity exec -i --nv -n --network=none -p -B `pwd`:/jackal_ws/src/ros_jackal -B ${BUFFER_PATH}:${BUFFER_PATH} ${1} /bin/bash /jackal_ws/src/ros_jackal/entrypoint.sh ${@:2}
