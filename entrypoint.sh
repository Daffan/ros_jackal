#!/bin/bash
pip3 install scipy 
source /jackal_ws/devel/setup.bash
cd /jackal_ws/src/ros_jackal
exec ${@:1}
