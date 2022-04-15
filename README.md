# ROS-Jackal
This is the repository for the under review paper "[Benchmarking Reinforcement Learning Techniques for Autonomous Navigation]()".

The results shown in the paper use Condor Cluster to distribute 100 actors for collecting trajectories. This setting can greatly speed up the training and make it feasible to finish all the experiments presented in the paper, however Condor Cluster is relatively inaccessible to most users. Instead, to guarantee reproducibility, we provide this version of repository that distributes the actors over 10 Singularity containers that can run locally on a single machine.

## Installation
0. Clone this repository
```
git clone https://github.com/Daffan/ros_jackal.git
cd ros_jackal
```

1. In your virtual environment, install the python dependencies:
```
pip install -r requirements.txt
```

2. Follow this instruction to install Singularity: https://sylabs.io/guides/3.0/user-guide/installation.html. Singularity version >= 3.6.3 is recommended to build the image.

3. (Only do following step if you really need!) The code does not require ROS installation, since the rollout happens in the container, but if you have need to develop based on our repo, running ROS and Gazebo simulation out of the container enables GUI and is easier to debug. Follow steps below to install ROS dependencies (assume `melodic` ROS installed already):

* Create ROS workspace
```
mkdir -p /<YOUR_HOME_DIR>/jackal_ws/src
cd /<YOUR_HOME_DIR>/jackal_ws/src
```

* Clone this repo and required ros packages
```
git clone https://github.com/Daffan/ros_jackal.git
git clone https://github.com/jackal/jackal.git --branch melodic-devel
git clone https://github.com/jackal/jackal_simulator.git --branch melodic-devel
git clone https://github.com/jackal/jackal_desktop.git --branch melodic-devel
git clone https://github.com/utexas-bwi/eband_local_planner.git --branch melodic-devel
```

* Install ROS package dependencies
```
cd ..
source /opt/ros/melodic/setup.bash
rosdep init; rosdep update
rosdep install -y --from-paths . --ignore-src --rosdistro=melodic
```

* Build the workspace
```
source devel/setup.bash
catkin_make
```

4. Verify your installation: (this script will run open-ai gym environment for 5 episodes)
Pull image file
```
singularity pull --name <FOLDER_PATH_TO_SAVE_IMAGE>/image:latest.sif library://zifanxu/ros_jackal_image/image:latest
```
```
./singularity_run.sh <FOLDER_PATH_TO_SAVE_IMAGE>/image:latest.sif test_env.py
```

## Train a deep RL navigation policy
```
python train.py --config configs/motion_laser.yaml
```

## Results
Success rate of policies trained with different neural network architectures and history lengths in static (top) and dynamic-wall (bottom) environments.


| **Static**     |                           |                           |              |
|----------------|---------------------------|---------------------------|--------------|
| History length | 1                         | 4                         | 8            |
| MLP            | $\boldsymbol{65 \pm 4\%}$ | $57 \pm 7\%$              | $42 \pm 2\%$ |
| GRU            | -                         | $51 \pm 2\%$              | $43 \pm 4\%$ |
| CNN            | -                         | $55 \pm 4\%$              | $45 \pm 5\%$ |
| Transformer    | -                         | $\boldsymbol{68 \pm 2\%}$ | $46 \pm 3\%$ |

| **Dynamic wall** |              |                           |                           |
|------------------|--------------|---------------------------|---------------------------|
| History length   | 1            | 4                         | 8                         |
| MLP              | $67 \pm 7\%$ | $72 \pm 1\%$              | $69 \pm 4\%$              |
| GRU              | -            | $\boldsymbol{82 \pm 4\%}$ | $\boldsymbol{78 \pm 5\%}$ |
| CNN              | -            | $63 \pm 3\%$              | $43 \pm 3\%$              |
| Transformer      | -            | $33 \pm 28\%$             | $15 \pm 13\%$             |

Success rate, survival time and traversal time of policies trained with different safe-RL methods, MPC with probabilistic transition model and DWA.

| **Safe-RL method** | **MLP**                     | **Lagrangian**            | **MPC**         | **DWA**              |
|--------------------|-----------------------------|---------------------------|-----------------|----------------------|
| Success rate       | $65 \pm 4\%$                | $\boldsymbol{74 \pm 2\%}$ | $70 \pm 3\%$    | $43\%$               |   |
| Survival time      | $8.0 \pm 1.5s$              | $16.2 \pm 2.5s$           | $55.7 \pm 4.9s$ | $\boldsymbol{88.6s}$ |   |
| Traversal time     | $\boldsymbol{7.5 \pm 0.3s}$ | $8.6 \pm 0.2s$            | $24.7 \pm 2.0s$ | $38.5s$              |   |

Success rate of policies trained with different model-based methods and different number of transition samples

| **Transition samples**   | **100k**                  | **500k**                  | **2000k**                 |
|--------------------------|---------------------------|---------------------------|---------------------------|
| MLP                      | $\boldsymbol{13 \pm 7\%}$ | $\boldsymbol{58 \pm 2\%}$ | $65 \pm 4\%$              |
| Dyna-style deterministic | $8 \pm 2\%$               | $30 \pm 10\%$             | $66 \pm 5\%$              |
| MPC deterministic        | $0 \pm 0\%$               | $21 \pm 10\%$             | $62 \pm 3\%$              |
| Dyna-style probabilistic | $0 \pm 0\%$               | $48 \pm 4\%$              | $\boldsymbol{70 \pm 1\%}$ |
| MPC probabilistic        | $0 \pm 0\%$               | $45 \pm 4\%$              | $\boldsymbol{70 \pm 3\%}$ |

Success rate of policies trained with different number of training environments

| **Environments** | **5**        | **10**       | **50**       | **100**      | **250**       |
|------------------|--------------|--------------|--------------|--------------|---------------|
| Success rate     | $43 \pm 3\%$ | $54 \pm 8\%$ | $65 \pm 4\%$ | $72 \pm 6\%$ | $74 \pm 2 \%$ |


(See below for all the config files used to reproduce the experiments)
```
 |-configs
 | |-safe_rl
 | | |-mpc.yaml
 | | |-mlp.yaml
 | | |-lagrangian.yaml
 | |-architecture_static
 | | |-mlp_history_length_4.yaml
 | | |-cnn_history_length_8.yaml
 | | |-cnn_history_length_4.yaml
 | | |-mlp_history_length_8.yaml
 | | |-rnn_history_length_4.yaml
 | | |-mlp_history_length_1.yaml
 | | |-cnn_history_length_1.yaml
 | | |-rnn_history_length_8.yaml
 | | |-rnn_history_length_1.yaml
 | | |-transformer_history_length_1.yaml
 | | |-transformer_history_length_4.yaml
 | | |-transformer_history_length_8.yaml
 | |-architecture_dynamic_wall
 | | |-cnn_history_length_1.yaml
 | | |-cnn_history_length_4.yaml
 | | |-cnn_history_length_8.yaml
 | | |-mlp_history_length_1.yaml
 | | |-mlp_history_length_4.yaml
 | | |-mlp_history_length_8.yaml
 | | |-rnn_history_length_1.yaml
 | | |-rnn_history_length_4.yaml
 | | |-rnn_history_length_8.yaml
 | | |-transformer_history_length_1.yaml
 | | |-transformer_history_length_4.yaml
 | | |-transformer_history_length_8.yaml
 | |-architecture_dynamic_box
 | | |-cnn_history_length_1.yaml
 | | |-cnn_history_length_4.yaml
 | | |-cnn_history_length_8.yaml
 | | |-mlp_history_length_1.yaml
 | | |-mlp_history_length_4.yaml
 | | |-mlp_history_length_8.yaml
 | | |-rnn_history_length_1.yaml
 | | |-rnn_history_length_4.yaml
 | | |-rnn_history_length_8.yaml
 | | |-transformer_history_length_1.yaml
 | | |-transformer_history_length_4.yaml
 | | |-transformer_history_length_8.yaml
 | |-model_based
 | | |-dyna.yaml
 | | |-mpc.yaml
 | |-generalization
 | | |-num_world_50.yaml
 | | |-num_world_5.yaml
 | | |-num_world_10.yaml
 | | |-num_world_100.yaml
 | | |-num_world_250.yaml
```