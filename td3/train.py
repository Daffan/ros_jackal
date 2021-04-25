import argparse
import yaml
import numpy as np
import gym
from datetime import datetime
from os.path import join, dirname, abspath, exists
import sys
import os
import shutil

import torch
from tensorboardX import SummaryWriter

from tianshou.utils.net.common import Net
from tianshou.exploration import GaussianNoise
from tianshou.utils.net.continuous import Actor, Critic
from tianshou.data import Collector, ReplayBuffer

sys.path.append(dirname(dirname(abspath(__file__))))
from policy import TD3Policy
from envs import registration
from offpolicy_trainer import offpolicy_trainer
from wrapper import DummyVectorEnvSpace

def parse_args():
    parser = argparse.ArgumentParser(description = 'ROS-Jackal TD3 training')
    parser.add_argument('--config', dest = 'config_path', type = str, default = 'td3/config.yaml', help = 'path to the configuration file')
    parser.add_argument('--save', dest = 'save_path', type = str, default = 'logging/', help = 'path to the saving folder')

    return parser.parse_args()

def initialize_config(args=parse_args()):
    config_path = args.config_path
    save_path = args.save_path

    # Load the config files
    with open(config_path, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    env_config = config['env_config']
    training_config = config['training_config']

    env_config["save_path"] = save_path
    env_config["config_path"] = config_path

    return env_config, training_config

def initialize_logging(env_config, training_config):
    # Config logging
    now = datetime.now()
    dt_string = now.strftime("%Y_%m_%d_%H_%M")
    save_path = join(
        env_config["save_path"], 
        env_config["env"], 
        training_config['algorithm'], 
        dt_string
    )

    if not exists(save_path):
        os.makedirs(save_path)
    writer = SummaryWriter(save_path)

    shutil.copyfile(
        env_config["config_path"], 
        join(save_path, "config.yaml")    
    )

    return save_path, writer

def initialize_envs(env_config):
    if not env_config["use_container"]:
        env = gym.make(env_config["env"], **env_config["kwargs"])
        train_envs = DummyVectorEnvSpace([lambda: env for _ in range(1)])
    else:
        raise NotImplementedError

    return train_envs

def seed(env_config):
    np.random.seed(env_config['seed'])
    torch.manual_seed(env_config['seed'])

def initialize_policy(training_config, env):
    state_shape = env.observation_space.shape
    action_shape = env.action_space.shape
    action_space_low = env.action_space.low
    action_space_high = env.action_space.high
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    net = Net(
        training_config['num_layers'],
        state_shape, device=device,
        hidden_layer_size=training_config['hidden_size']
    )
    actor = Actor(
        net, action_shape,
        1, device, 
        hidden_layer_size=training_config['hidden_size']
    )
    actor_optim = torch.optim.Adam(
        actor.parameters(), 
        lr=training_config['actor_lr'])

    net = Net(
        training_config['num_layers'], 
        state_shape,
        action_shape, 
        concat=True, 
        device=device, 
        hidden_layer_size=training_config['hidden_size']
    )
    critic1 = Critic(
        net, device, 
        hidden_layer_size=training_config['hidden_size']
    ).to(device)
    critic1_optim = torch.optim.Adam(
        critic1.parameters(), 
        lr=training_config['critic_lr']
    )
    critic2 = Critic(
        net, device, 
        hidden_layer_size=training_config['hidden_size']
    ).to(device)
    critic2_optim = torch.optim.Adam(
        critic2.parameters(), 
        lr=training_config['critic_lr']
    )

    training_args = training_config["policy_args"]
    exploration_noise = GaussianNoise(
        sigma=training_config['exploration_noise']
    )
    policy = TD3Policy(
        actor, actor_optim, 
        critic1, critic1_optim, 
        critic2, critic2_optim,
        exploration_noise=exploration_noise,
        action_range=[action_space_low, action_space_high],
        **training_args
    )

    buffer = ReplayBuffer(training_config['buffer_size'])

    return policy, buffer

def compute_exp_noise(e, start, ratio, epoch):
    exp_noise = start * (1. - (e - 1.) / epoch / ratio)
    return max(0, exp_noise)

def generate_train_fn(training_config, policy, save_path):
    return lambda e: [
        policy.set_exp_noise(
            GaussianNoise(
                sigma=compute_exp_noise(
                    e, training_config["exploration_noise"], 
                    training_config["exploration_ratio"], 
                    training_config["training_args"]['max_epoch']
                )
            )
        )
    ]

def train(train_envs, policy, buffer, env_config, training_config):
    save_path, writer = initialize_logging(env_config, training_config)

    train_collector = Collector(policy, train_envs, buffer)
    training_args = training_config["training_args"]
    train_fn = generate_train_fn(training_config, policy, save_path)

    result = offpolicy_trainer(
        policy, 
        train_collector, 
        train_fn=train_fn, 
        writer=writer,
        **training_args
    )

    train_envs.close()

if __name__ == "__main__":
    env_config, training_config = initialize_config()
    seed(env_config)
    train_envs = initialize_envs(env_config)
    policy, buffer = initialize_policy(training_config, train_envs)
    train(train_envs, policy, buffer, env_config, training_config)