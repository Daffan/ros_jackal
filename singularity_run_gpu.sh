#!/bin/bash
export PATH=$PATH:/lusr/opt/singularity-3.2.1/bin
export IMAGE_PATH=/scratch/cluster/zifan/ros_jackal_image_old_torch.sif
singularity exec -i -n --network=none -p --nv -B `pwd`:/jackal_ws/src/ros_jackal ${IMAGE_PATH} /bin/bash /jackal_ws/src/ros_jackal/entrypoint.sh ${@:1}
