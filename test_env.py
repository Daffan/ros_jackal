########################################################################
# This script tests the jackal navigation environment with random action
########################################################################
import os
import json
import pickle
from os.path import join, dirname, abspath, exists
import sys
sys.path.append(dirname(dirname(abspath(__file__))))
import gym
import random
import numpy as np

import envs.registration
from envs.wrappers import ShapingRewardWrapper

def main():
    env = gym.make(
        id='motion_control_continuous_laser-v0', 
        world_name='BARN/world_192.world',
        gui=True,
        init_position=[-2, 2, np.pi/2],
        goal_position=[0, 10, 0],
        time_step=0.2,
        slack_reward=0,
        success_reward=10,
        collision_reward=-10,
        failure_reward=0,
        max_collision=1
    )
    env = ShapingRewardWrapper(env)
    env.reset()
    done  = False
    count = 0
    ep_count = 0
    ep_rew = 0

    high = env.action_space.high
    low = env.action_space.low
    bias = (high + low) / 2
    scale = (high - low) / 2
    while ep_count < 5:

        actions = 2*(np.random.rand(env.action_space.shape[0]) - 0.5)
        actions *= scale
        actions += bias

        count += 1
        obs, rew, done, info = env.step(actions)
        ep_rew += rew
        p = env.gazebo_sim.get_model_state().pose.position
        print('current episode: %d, current step: %d, time: %.2f, X position: %f(world_frame), Y position: %f(world_frame), rew: %f, collision: %d' %(ep_count, count, info["time"], p.x, p.y, rew, info["collision"]))
        print("actions: ", actions)
        if done:
            ep_count += 1
            env.reset()
            count = 0
            ep_rew = 0

    env.close()

if __name__ == '__main__':
    main()