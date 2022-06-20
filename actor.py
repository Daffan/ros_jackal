import os
import yaml
import pickle
from os.path import join, dirname, abspath, exists
import sys
import torch
import gym
import numpy as np
import random
import time
import rospy
import argparse
import logging

from train import initialize_policy
from envs import registration
from envs.wrappers import StackFrame

BUFFER_PATH = os.getenv('BUFFER_PATH')
if not BUFFER_PATH:
    BUFFER_PATH = "local_buffer"

# add path to the plugins to the GAZEBO_PLUGIN_PATH
gpp = os.getenv('GAZEBO_PLUGIN_PATH') if os.getenv('GAZEBO_PLUGIN_PATH') is not None else ""
wd = os.getcwd()
os.environ['GAZEBO_PLUGIN_PATH'] = os.path.join(wd, "jackal_helper/plugins/build") + ":" + gpp
rospy.logwarn(os.environ['GAZEBO_PLUGIN_PATH'])

def initialize_actor(id):
    rospy.logwarn(">>>>>>>>>>>>>>>>>> actor id: %s <<<<<<<<<<<<<<<<<<" %(str(id)))
    assert os.path.exists(BUFFER_PATH), BUFFER_PATH
    actor_path = join(BUFFER_PATH, 'actor_%s' %(str(id)))

    if not exists(actor_path):
        os.mkdir(actor_path) # path to store all the trajectories

    f = None
    c = 0
    while f is None and c < 10:
        c += 1
        try:
            f = open(join(BUFFER_PATH, 'config.yaml'), 'r')
        except:
            rospy.logwarn("wait for critor to be initialized")
            time.sleep(2)

    config = yaml.load(f, Loader=yaml.FullLoader)

    return config

def load_policy(policy):
    f = True
    policy_name = "policy"
    while f:
        try:
            if not os.path.exists(join(BUFFER_PATH, "%s_copy_actor" %(policy_name))):
                policy.load(BUFFER_PATH, policy_name)
            f = False
        except FileNotFoundError:
            time.sleep(1)
        except:
            logging.exception('')
            time.sleep(1)
    return policy

def write_buffer(traj, id):
    file_names = os.listdir(join(BUFFER_PATH, 'actor_%s' %(str(id))))
    if len(file_names) == 0:
        ep = 0
    else:
        eps = [int(f.split("_")[-1].split(".pickle")[0]) for f in file_names]  # last index under this folder
        sorted(eps)
        ep = eps[-1] + 1
    if len(file_names) < 10:
        with open(join(BUFFER_PATH, 'actor_%s' %(str(id)), 'traj_%d.pickle' %(ep)), 'wb') as f:
            try:
                pickle.dump(traj, f)
            except OSError as e:
                logging.exception('Failed to dump the trajectory! %s', e)
                pass
    return ep

def get_world_name(config, id):
    if len(config["container_config"]["worlds"]) < config["container_config"]["num_actor"]:
        duplicate_time = config["container_config"]["num_actor"] // len(config["container_config"]["worlds"]) + 1
        worlds = config["container_config"]["worlds"] * duplicate_time
    else:  # if num_actors < num_worlds, then each actor will rollout in a random world
        worlds = config["container_config"]["worlds"].copy()
        random.shuffle(worlds)
        worlds = worlds[:config["container_config"]["num_actor"]]
    world_name = worlds[id]
    if isinstance(world_name, int):
        world_name = "BARN/world_%d.world" %(world_name)
    return world_name

def _debug_print_robot_status(env, count, rew, actions):
    p = env.gazebo_sim.get_model_state().pose.position
    print(actions)
    print('current step: %d, X position: %f(world_frame), Y position: %f(world_frame), rew: %f' %(count, p.x, p.y, rew))

def main(args):
    id = args.id
    config = initialize_actor(id)
    env_config = config['env_config']
    world_name = get_world_name(config, id)
    env_config["kwargs"]["world_name"] = world_name
    env = gym.make(env_config["env_id"], **env_config["kwargs"])
    env = StackFrame(env, stack_frame=env_config["stack_frame"])

    policy, _ = initialize_policy(config, env, init_buffer=False, device="cpu")
    num_ep = 0

    for _ in range(args.num_trajs):
        obs = env.reset()
        traj = []
        done = False
        policy = load_policy(policy)
        while not done:
            actions = policy.select_action(obs)
            obs_new, rew, done, info = env.step(actions)
            info["world"] = world_name
            traj.append([obs, actions, rew, done, info])
            obs = obs_new
        num_ep += 1
        write_buffer(traj, id)
    
    print(">>>>>>>>>>>>>>>>>>>>>>>>> actor_id: %d, world_idx: %s, num_episode: %d" %(id, world_name, num_ep))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = 'start an actor')
    parser.add_argument('--id', dest='id', type = int, default = 1)
    parser.add_argument('--num_trajs', dest='num_trajs', type = int, default = 5)

    args = parser.parse_args()
    main(args)
