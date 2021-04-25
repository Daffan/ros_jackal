import os
import yaml
import pickle
from os.path import join, dirname, abspath, exists
import sys
sys.path.append(dirname(dirname(abspath(__file__))))
import torch
import gym
import numpy as np
import random
import time
import rospy

from tianshou.exploration import GaussianNoise
from tianshou.data import Batch

from policy import TD3Policy, SACPolicy
from train import initialize_envs, initialize_policy

random.seed(43)

BASE_PATH = join(os.getenv('HOME'), 'buffer')

def initialize_actor(id):
    rospy.logwarn(">>>>>>>>>>>>>>>>>> actor id: %s <<<<<<<<<<<<<<<<<<" %(str(id)))
    assert os.path.exists(BASE_PATH)
    actor_path = join(BASE_PATH, 'actor_%s' %(str(id)))

    if not exists(actor_path):
        os.mkdir(actor_path) # path to store all the trajectories

    f = None
    while f is None:
        try:
            f = open(join(BASE_PATH, 'config.yaml'), 'r')
        except:
            rospy.logwarn("wait for critor to be initialized")
            time.sleep(2)

    config = yaml.load(f, Loader=yaml.FullLoader)

    return config

def load_model(model):
    model_path = join(BASE_PATH, 'policy.pth')
    state_dict = {}
    state_dict_raw = None
    while state_dict_raw is None:
        try:
            state_dict_raw = torch.load(model_path)
        except:
            time.sleep(0.1)
            pass

    model.load_state_dict(state_dict_raw)
    model = model.float()
    # exploration noise std
    with open(join(BASE_PATH, 'eps.txt'), 'r') as f:
        eps = None
        while eps is not None:
            try:
                eps = float(f.readlines()[0])
            except IndexError:
                pass

    return model, eps

def write_buffer(traj, ep, id):
    with open(join(BASE_PATH, 'actor_%s' %(str(id)), 'traj_%d.pickle' %(ep)), 'wb') as f:
        pickle.dump(traj, f)

def get_world_name(config, id):
    if len(config["condor_config"]["worlds"]) < config["condor_config"]["num_actor"]:
        duplicate_time = config["condor_config"]["num_actor"] // config["condor_config"]["worlds"] + 1
        config["condor_config"]["worlds"] *= config["condor_config"]["worlds"] * duplicate_time
    world_name = config["condor_config"]["worlds"][id]
    if isinstance(world_name, int):
        world_name = "world_%d.world" %(world_name)
    return world_name

def main(id):

    config = init_actor(id)
    env_config = config['env_config']
    training_config = config["training_config"]
    world_name = get_world_name(config, id)
    env_config["kwargs"]["world_name"] = world_name
    env = gym.make(env_config["env"], **env_config["kwargs"])

    policy, _ = initialize_policy(training_config, env)

    print(">>>>>>>>>>>>>> Running on %s <<<<<<<<<<<<<<<<" %(world_name)
    ep = 0
    while True:
        obs = env.reset()
        obs_batch = Batch(obs=[obs], info={})
        ep += 1
        traj = []
        done = False
        count = 0
        policy, eps = load_model(policy)
        try:
            policy.set_exp_noise(GaussianNoise(sigma=eps))
        except:
            pass
        while not done:
            time.sleep(0.01)
            p = random.random()
            obs = torch.tensor([obs]).float()
            if isinstance(policy._noise, GaussianNoise) or p > eps:
                actions = policy(obs_batch).act.cpu().detach().numpy().reshape(-1)
            else:
                actions = get_random_action()
                actions = np.array(actions)
            obs_new, rew, done, info = env.step(actions)
            count += 1
            info["world"] = world_name
            traj.append([obs, actions, rew, done, info])
            obs_batch = Batch(obs=[obs_new], info={})
            obs = obs_new

        traj_new = traj
        write_buffer(traj_new, ep, id)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description = 'start an actor')
    parser.add_argument('--id', dest='actor_id', type = int, default = 1)
    id = parser.parse_args().actor_id
    main(id)