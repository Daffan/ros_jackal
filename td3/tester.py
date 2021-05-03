from actor import initialize_actor, load_model, write_buffer
from train import initialize_policy
import os
from os.path import dirname, abspath
import gym
import torch
import argparse

from tianshou.data import Batch

import sys
sys.path.append(dirname(dirname(abspath(__file__))))
from envs.wrappers import ShapingRewardWrapper

BASE_PATH = os.getenv('BUFFER_PATH')

def get_world_name(config, id):
    # We test each test world with two actors, so duplicate the lict by a factor of two
    world_name = (config["condor_config"]["test_worlds"] * 2)[id]
    if isinstance(world_name, int):
        world_name = "BARN/world_%d.world" %(world_name)
    return world_name

def main(args):
    config = initialize_actor(args.id)
    num_trials = config["condor_config"]["num_trials"]
    env_config = config['env_config']
    world_name = get_world_name(config, args.id)
    test_object = config["condor_config"]["test_object"]

    env_config["kwargs"]["world_name"] = world_name
    env = gym.make(env_config["env_id"], **env_config["kwargs"])
    if env_config["shaping_reward"]:
        env = ShapingRewardWrapper(env)

    policy, _ = initialize_policy(config, env)
    policy, eps = load_model(policy)

    print(">>>>>>>>>>>>>> Running on %s <<<<<<<<<<<<<<<<" %(world_name))
    ep = 0
    while ep < num_trials:
        obs = env.reset()
        obs_batch = Batch(obs=[obs], info={})
        ep += 1
        traj = []
        done = False
        try:
            policy.set_exp_noise(GaussianNoise(sigma=eps))
        except:
            pass
        while not done:
            obs = torch.tensor([obs]).float()
            if test_object == "local":
                actions = policy(obs_batch).act.cpu().detach().numpy().reshape(-1)
            elif test_object == "dwa":
                actions = env_config["kwargs"]["param_init"]
            obs_new, rew, done, info = env.step(actions)
            info["world"] = world_name
            traj.append([None, None, rew, done, info])  # For testing, we only need rew and ep_length
            obs_batch = Batch(obs=[obs_new], info={})
            obs = obs_new

            # _debug_print_robot_status(env, len(traj), rew)

        write_buffer(traj, ep, args.id)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = 'start an tester')
    parser.add_argument('--id', dest='id', type = int, default = 1)
    args = parser.parse_args()
    main(args)