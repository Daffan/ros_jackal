# ROS-Jackal
This is the repository for the paper "[Benchmarking Reinforcement Learning Techniques for Autonomous Navigation](https://arxiv.org/abs/2210.04839)".

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

2. Follow this instruction to install Singularity: https://docs.sylabs.io/guides/latest/admin-guide/installation.html#installation-on-linux. Singularity version >= 3.6.3 is **required** to build the image.

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
git clone https://github.com/utexas-bwi/eband_local_planner.git
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
catkin_make
source devel/setup.bash
```

4. Verify your installation: (this script will run open-ai gym environment for 5 episodes)

Pull image file (modify the <FOLDER_PATH_TO_SAVE_IMAGE> in the command, image file size ~ 3G
```
singularity pull --name <PATH_TO_THIS_REPO>/local_buffer/image:latest.sif library://zifanxu/ros_jackal_image/image:latest
```
```
./singularity_run.sh <PATH_TO_THIS_REPO>/local_buffer/nav_benchmark.sif python3 test_env.py
```

## Train a deep RL navigation policy
To train a navigation policy, you just need to specify a ```.yaml``` file that includes the parameters for specific experiment. For instance,
```
python train.py --config configs/e2e_default_TD3.yaml
```
We provide the full list of ```.yaml``` files used in our experiment in the end.

This repo saves the collected trajectories from each actor in a local buffer folder, also actors load the recent policy from this folder. By default, buffer folder is a folder named `local_buffer` in current dictionary. You can specify a new folder as `export BUFFER_FOLDER=/PATH/TO/YOUR/BUFFER_FOLDER`. The logging files can be found under folder `logging`.

## Train in computing cluster
Cluster requires a shared file system, where multiple actors load the lastest policy, rollout, and save the trajectory in the `BUFFER_FOLDER`. Then, a critic collects trajectories from `BUFFER_FOLDER` and updates the policy.

This is asyncronized training pipeline, namely the actors might fall behind and do not generate trajectories from the latest policy.

1. Download the Singularity image
```
singularity pull --name <PATH/TO/IMAGE>/image:latest.sif library://zifanxu/ros_jackal_image/image:latest
```
2. On critic computing node
```
export BUFFER_PATH=<BUFFER_PATH>
./singularity_run.sh <PATH/TO/IMAGE>/image:latest.sif python train.py --config configs/e2e_default_TD3_cluster.yaml
```
3. On actor computing node 0 (you need to run `0-50` computing nodes as defined in line 60 in `container_config.yaml`).
```
export BUFFER_PATH=<BUFFER_PATH>
./singularity_run.sh <PATH/TO/IMAGE>/image:latest.sif python actor.py --id 0
```

## Results
Success rate of policies trained with different neural network architectures and history lengths in static (top) and dynamic-wall (bottom) environments.


| **Static**     |                           |                           |              |
|----------------|---------------------------|---------------------------|--------------|
| History length | 1                         | 4                         | 8            |
| MLP            | 65 ± 4\%                  | 57 ± 7\%                  | 42 ± 2\%     |
| GRU            | -                         | 51 ± 2\%                  | 43 ± 4\%     |
| CNN            | -                         | 55 ± 4\%                  | 45 ± 5\%     |
| Transformer    | -                         | **68 ± 2\%**              | 46 ± 3\%     |

| **Dynamic box** |              |                           |                           |
|------------------|--------------|---------------------------|---------------------------|
| History length   | 1            | 4                         | 8                         |
| MLP              | 50 ± 5\%     | 35 ± 2\%                  | 46 ± 3\%                  |
| GRU              | -            | 48 ± 4\%                  | 45 ± 1\%                  |
| CNN              | -            | 42 ± 5\%                  | 40 ± 1\%                  |
| Transformer      | -            | **52 ± 1\%**              | 44 ± 4\%                  |

| **Dynamic wall** |              |                           |                           |
|------------------|--------------|---------------------------|---------------------------|
| History length   | 1            | 4                         | 8                         |
| MLP              | 67 ± 7\%     | 72 ± 1\%                  | 69 ± 4\%                  |
| GRU              | -            | **82 ± 4\%**              | 78 ± 5\%                  |
| CNN              | -            | 63 ± 3\%                  | 43 ± 3\%                  |
| Transformer      | -            | 33 ± 28\%                 | 15 ± 13\%                 |

Success rate, survival time and traversal time of policies trained with different safe-RL methods, MPC with probabilistic transition model and DWA.

| **Safe-RL method** | **MLP**                     | **Lagrangian**            | **MPC**         | **DWA**              |
|--------------------|-----------------------------|---------------------------|-----------------|----------------------|
| Success rate       | 65 ± 4\%                    | **74 ± 2\%**              | 70 ± 3\%        | 43\%                 |
| Survival time      | 8.0 ± 1.5s                  | 16.2 ± 2.5s               | 55.7 ± 4.9s     | **88.6s**            |
| Traversal time     | **7.5 ± 0.3s**              | 8.6 ± 0.2s                | 24.7 ± 2.0s     | 38.5s                |

Success rate of policies trained with different model-based methods and different number of transition samples

| **Transition samples**   | **100k**                  | **500k**                  | **2000k**                 |
|--------------------------|---------------------------|---------------------------|---------------------------|
| MLP                      | **13 ± 7\%**              | **58 ± 2\%**              | 65 ± 4\%                  |
| Dyna-style deterministic | 8 ± 2\%                   | 30 ± 10\%                 | 66 ± 5\%                  |
| MPC deterministic        | 0 ± 0\%                   | 21 ± 10\%                 | 62 ± 3\%                  |
| Dyna-style probabilistic | 0 ± 0\%                   | 48 ± 4\%                  | **70 ± 1\%**              |
| MPC probabilistic        | 0 ± 0\%                   | 45 ± 4\%                  | **70 ± 3\%**              |

Success rate of policies trained with different number of training environments

| **Environments** | **5**        | **10**       | **50**       | **100**      | **250**       |
|------------------|--------------|--------------|--------------|--------------|---------------|
| Success rate     | 43 ± 3\%     | 54 ± 8\%     | 65 ± 4\%     | 72 ± 6\%     | 74 ± 2 \%     |


(See below for all the config files used to reproduce the experiments)
```
 └─configs
 │ └─safe_rl
 │ │ └─mpc.yaml
 │ │ └─mlp.yaml
 │ │ └─lagrangian.yaml
 │ └─architecture_static
 │ │ └─mlp_history_length_4.yaml
 │ │ └─cnn_history_length_8.yaml
 │ │ └─cnn_history_length_4.yaml
 │ │ └─mlp_history_length_8.yaml
 │ │ └─rnn_history_length_4.yaml
 │ │ └─mlp_history_length_1.yaml
 │ │ └─cnn_history_length_1.yaml
 │ │ └─rnn_history_length_8.yaml
 │ │ └─rnn_history_length_1.yaml
 │ │ └─transformer_history_length_1.yaml
 │ │ └─transformer_history_length_4.yaml
 │ │ └─transformer_history_length_8.yaml
 │ └─architecture_dynamic_wall
 │ │ └─cnn_history_length_1.yaml
 │ │ └─cnn_history_length_4.yaml
 │ │ └─cnn_history_length_8.yaml
 │ │ └─mlp_history_length_1.yaml
 │ │ └─mlp_history_length_4.yaml
 │ │ └─mlp_history_length_8.yaml
 │ │ └─rnn_history_length_1.yaml
 │ │ └─rnn_history_length_4.yaml
 │ │ └─rnn_history_length_8.yaml
 │ │ └─transformer_history_length_1.yaml
 │ │ └─transformer_history_length_4.yaml
 │ │ └─transformer_history_length_8.yaml
 │ └─architecture_dynamic_box
 │ │ └─cnn_history_length_1.yaml
 │ │ └─cnn_history_length_4.yaml
 │ │ └─cnn_history_length_8.yaml
 │ │ └─mlp_history_length_1.yaml
 │ │ └─mlp_history_length_4.yaml
 │ │ └─mlp_history_length_8.yaml
 │ │ └─rnn_history_length_1.yaml
 │ │ └─rnn_history_length_4.yaml
 │ │ └─rnn_history_length_8.yaml
 │ │ └─transformer_history_length_1.yaml
 │ │ └─transformer_history_length_4.yaml
 │ │ └─transformer_history_length_8.yaml
 │ └─model_based
 │ │ └─dyna.yaml
 │ │ └─mpc.yaml
 │ └─generalization
 │ │ └─num_world_50.yaml
 │ │ └─num_world_5.yaml
 │ │ └─num_world_10.yaml
 │ │ └─num_world_100.yaml
 │ │ └─num_world_250.yamlconfigs
 │ └─safe_rl
 │ │ └─mpc.yaml
 │ │ └─mlp.yaml
 │ │ └─lagrangian.yaml
 │ └─architecture_static
 │ │ └─mlp_history_length_4.yaml
 │ │ └─cnn_history_length_8.yaml
 │ │ └─cnn_history_length_4.yaml
 │ │ └─mlp_history_length_8.yaml
 │ │ └─rnn_history_length_4.yaml
 │ │ └─mlp_history_length_1.yaml
 │ │ └─cnn_history_length_1.yaml
 │ │ └─rnn_history_length_8.yaml
 │ │ └─rnn_history_length_1.yaml
 │ │ └─transformer_history_length_1.yaml
 │ │ └─transformer_history_length_4.yaml
 │ │ └─transformer_history_length_8.yaml
 │ └─architecture_dynamic_wall
 │ │ └─cnn_history_length_1.yaml
 │ │ └─cnn_history_length_4.yaml
 │ │ └─cnn_history_length_8.yaml
 │ │ └─mlp_history_length_1.yaml
 │ │ └─mlp_history_length_4.yaml
 │ │ └─mlp_history_length_8.yaml
 │ │ └─rnn_history_length_1.yaml
 │ │ └─rnn_history_length_4.yaml
 │ │ └─rnn_history_length_8.yaml
 │ │ └─transformer_history_length_1.yaml
 │ │ └─transformer_history_length_4.yaml
 │ │ └─transformer_history_length_8.yaml
 │ └─architecture_dynamic_box
 │ │ └─cnn_history_length_1.yaml
 │ │ └─cnn_history_length_4.yaml
 │ │ └─cnn_history_length_8.yaml
 │ │ └─mlp_history_length_1.yaml
 │ │ └─mlp_history_length_4.yaml
 │ │ └─mlp_history_length_8.yaml
 │ │ └─rnn_history_length_1.yaml
 │ │ └─rnn_history_length_4.yaml
 │ │ └─rnn_history_length_8.yaml
 │ │ └─transformer_history_length_1.yaml
 │ │ └─transformer_history_length_4.yaml
 │ │ └─transformer_history_length_8.yaml
 │ └─model_based
 │ │ └─dyna.yaml
 │ │ └─mpc.yaml
 │ └─generalization
 │ │ └─num_world_50.yaml
 │ │ └─num_world_5.yaml
 │ │ └─num_world_10.yaml
 │ │ └─num_world_100.yaml
 │ │ └─num_world_250.yaml
```
