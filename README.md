# ROS-Jackal
ROS-Jackal environment for RL

# Install the repo without Singularity container
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
Test installation: the script runs the `dwa_param_continuous_laser-v0` environment for 1000 steps
```
python3 script/test_env.py
```
Deploy APPLR policy (https://arxiv.org/pdf/2011.00397.pdf)
```
python3 scripts/test_param_policy.py --gui --verbose --repeats 5 --policy_path data/param_policy --world_id 0
```
Deploy default DWA navigation stack
```
python3 scripts/test_param_policy.py --gui --verbose --repeats 5 --policy_path data/param_policy --world_id 0 --default_dwa
```

# Singularity container
Build the image
```
sudo singularity build --notest ros_jackal_image.sif Singularityfile.def
```
All the command could be ran similarly by adding `./singularity_run.sh`, eg. `test_env.py`
```
./singularity_run.sh python3 script/test_env.py
```

# OpenAI gym environment
We now implemented four openai-gym environments for direct motion control and parameter tuning: 
* `dwa_param_continuous_laser-v0`
* `dwa_param_continuous_costmap-v0`
* `motion_control_continuous_laser-v0`
* `motion_control_continuous_costmap-v0`


# TD3 training
We now use [tianshou]() implementation of TD3.

## Configurations
All the environments and training configurations are under `td3/config.yaml`. All the ros and jackal related configurations are under `jackal_helper/configs`, eg. `move_base` related configurations.

## Train locally without cluster
Change `env_config.use_condor = False` in `td3/config.yaml`, then run: (the config is optimized for cluster, probably won't work very well locally)
```
python3 td3/train.py
```

## Train in parallel with cluster
Login to one of the submitting node: `darmok.cs.utexas.edu` or `jalad.cs.utexas.edu`. Then clone this repo

```
git clone https://github.com/Daffan/ros_jackal.git; cd ros_jackal
```

Submission node does not allow root operations, so build the singularity image locally, then copy to the submission node under folder `ros_jackal`. 

Make sure `env_config.use_condor = True` in `td3/config.yaml`, then run:
```
python3 gen_sub.py
```

The command will first submit the central learner node and hang up for 1 minute, then submit actor nodes. To check the submission status, run:
```
condor_q
```

Logging for training could be found under `logging\<env_id>\<algorithm>\<timestamp>`. To find the logging for HTCondor, check a temporary buffer folder under the home dictionary: `/${HOME}/<hashcode>/out`.

