# ROS-Jackal
ROS-Jackal environment for RL

# Install the code without Singularity container
Create workspace
```
mkdir -p jackal_ws/src
cd jackal_ws/src
```
Clone the repo and install the python dependencies
```
git clone https://github.com/Daffan/ros_jackal.git; cd ros_jackal
pip3 install -r requirements.txt
```
Install Jackal packages
```
cd ..
git clone https://github.com/jackal/jackal.git
git clone https://github.com/jackal/jackal_simulator.git
git clone https://github.com/jackal/jackal_desktop.git
source /opt/ros/melodic/setup.bash
cd ..; catkin_make
```
Install ros dependencies
```
source devel/setup.bash
rosdep init; rosdep update
rosdep install -y --from-paths . --ignore-src --rosdistro=melodic
```
Test installation
```
python3 script/test_env.py
```

# Singularity container
Build the image
```
sudo singularity build --notest ros_jackal_image.sif Singularityfile.def
```
Test the container
```
./singularity_run.sh python3 script/test_env.py
```

# OpenAI gym environment
We now implemented four openai-gym environments for direct motion control and parameter tuning: 
* `dwa_param_continuous_laser-v0`
* `dwa_param_continuous_costmap-v0`
* `motion_control_continuous_laser-v0`
* `motion_control_continuous_costmap-v0`