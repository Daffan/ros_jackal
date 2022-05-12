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

# add path to the plugins to the GAZEBO_PLUGIN_PATH
gpp = os.getenv('GAZEBO_PLUGIN_PATH') if os.getenv('GAZEBO_PLUGIN_PATH') is not None else ""
wd = os.getcwd()
os.environ['GAZEBO_PLUGIN_PATH'] = os.path.join(wd, "jackal_helper/plugins/build") + ":" + gpp
rospy.logwarn(os.environ['GAZEBO_PLUGIN_PATH'])

def load_policy(policy, policy_path):
    f = True
    policy_name = "last_policy"
    policy.load(policy_path, policy_name)
    policy.exploration_noise = 0 
    return policy

def get_world_name(config, id):
    assert 0 <= id < 300, "BARN dataset world index ranges from 0-299"
    world_name = "BARN/world_%d.world" %(id)
    return world_name

def _debug_print_robot_status(env, count, rew, actions):
    p = env.gazebo_sim.get_model_state().pose.position
    print(actions)
    print('current step: %d, X position: %f(world_frame), Y position: %f(world_frame), rew: %f' %(count, p.x, p.y, rew))

def main(args):
    with open(join(args.policy_path, "config.yaml"), "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    env_config = config['env_config']
    world_name = get_world_name(config, args.id)
    env_config["kwargs"]["world_name"] = world_name
    env = gym.make(env_config["env_id"], **env_config["kwargs"])
    env = StackFrame(env, stack_frame=env_config["stack_frame"])

    policy, _ = initialize_policy(config, env, init_buffer=False, device="cpu")
    num_ep = 0
    num_step = 0
    for _ in range(args.num_trajs):
        obs = env.reset()
        done = False
        policy = load_policy(policy, args.policy_path)
        while not done:
            actions = policy.select_action(obs)
            obs_new, rew, done, info = env.step(actions)
            obs = obs_new
            _debug_print_robot_status(env, num_step, 0, actions)
            num_step += 1
        num_ep += 1
        num_step = 0

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = 'start an actor')
    parser.add_argument('--policy_path', type = str)
    parser.add_argument('--world_id', dest='id', type=int, default=0)
    parser.add_argument('--num_trajs', dest='num_trajs', type = int, default = 1)

    args = parser.parse_args()
    main(args)
