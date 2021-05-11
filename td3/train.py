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

from tianshou.utils.net.common import Net as MLP
from tianshou.exploration import GaussianNoise
from tianshou.utils.net.continuous import Actor
from tianshou.data import Collector, ReplayBuffer
from tianshou.env import DummyVectorEnv

sys.path.append(dirname(dirname(abspath(__file__))))
from policy import TD3Policy
from envs import registration
from envs.wrappers import ShapingRewardWrapper
from offpolicy_trainer import offpolicy_trainer
from offpolicy_trainer_condor import offpolicy_trainer_condor
from collector import Collector as CondorCollector
from infomation_envs import InfoEnv
from model import CNN, Critic  # comtumized Critic to cover the the CNN case

def initialize_config(config_path, save_path):
    # Load the config files
    with open(config_path, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    config["env_config"]["save_path"] = save_path
    config["env_config"]["config_path"] = config_path

    return config

def initialize_logging(config):
    env_config = config["env_config"]
    training_config = config["training_config"]

    # Config logging
    now = datetime.now()
    dt_string = now.strftime("%Y_%m_%d_%H_%M")
    save_path = join(
        env_config["save_path"], 
        env_config["env_id"], 
        training_config['algorithm'], 
        dt_string
    )
    print("    >>>> Saving to %s" % save_path)
    if not exists(save_path):
        os.makedirs(save_path)
    writer = SummaryWriter(save_path)

    shutil.copyfile(
        env_config["config_path"], 
        join(save_path, "config.yaml")    
    )

    return save_path, writer

def initialize_envs(config):
    env_config = config["env_config"]
    
    if not env_config["use_condor"]:
        env = gym.make(env_config["env_id"], **env_config["kwargs"])
        if env_config["shaping_reward"]:
            env = ShapingRewardWrapper(env)
        train_envs = DummyVectorEnv([lambda: env for _ in range(1)])
    else:
        # If use condor, we want to avoid initializing env instance from the central learner
        # So here we use a fake env with obs_space and act_space information
        print("    >>>> Using actors on Condor")
        train_envs = InfoEnv(config)
    return train_envs

def seed(config):
    env_config = config["env_config"]
    
    np.random.seed(env_config['seed'])
    torch.manual_seed(env_config['seed'])

def initialize_policy(config, env):
    training_config = config["training_config"]

    state_shape = env.observation_space.shape
    action_shape = env.action_space.shape
    action_space_low = env.action_space.low
    action_space_high = env.action_space.high
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if training_config["network"] == "mlp":
        make_net = lambda act_shape: MLP(
            training_config['num_layers'],
            state_shape, 
            act_shape, 
            concat=True if act_shape != 0 else False,
            device=device,
            hidden_layer_size=training_config['hidden_size']
        )
    elif training_config["network"] == "cnn":
        make_net =  lambda act_shape: CNN(
            action_shape=act_shape,
            device=device
        )
    else:
        raise NotImplementedError
    actor_net = make_net(0)
    actor = Actor(
        actor_net,
        action_shape,
        max_action=1, 
        device=device, 
        hidden_layer_size=training_config['hidden_size']
    ).to(device)
    actor_optim = torch.optim.Adam(
        actor.parameters(), 
        lr=training_config['actor_lr']
    )

    critic_net = make_net(action_shape)
    critic1 = Critic(
        critic_net, 
        device=device,
        network=training_config["network"],
        hidden_layer_size=training_config['hidden_size']
    ).to(device)
    critic1_optim = torch.optim.Adam(
        critic1.parameters(), 
        lr=training_config['critic_lr']
    )
    critic2 = Critic(
        critic_net,
        device=device,
        network=training_config["network"],
        hidden_layer_size=training_config['hidden_size']
    ).to(device)
    critic2_optim = torch.optim.Adam(
        critic2.parameters(), 
        lr=training_config['critic_lr']
    )

    training_args = training_config["policy_args"]
    exploration_noise = GaussianNoise(
        sigma=training_config['exploration_noise_start']
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

def compute_exp_noise(e, start, end, ratio, epoch):
    exp_noise = start * (1. - (e - 1.) / epoch / ratio)
    return max(end, exp_noise)

def generate_train_fn(config, policy, save_path):
    training_config = config["training_config"]
    return lambda e: [
        policy.set_exp_noise(
            GaussianNoise(
                sigma=compute_exp_noise(
                    e, training_config["exploration_noise_start"], 
                    training_config["exploration_noise_end"],
                    training_config["exploration_ratio"], 
                    training_config["training_args"]["max_epoch"]
                )
            )
        ),
        torch.save(
            policy.state_dict(), 
            os.path.join(save_path, "policy.pth")
        )
    ]

def train(train_envs, policy, buffer, config):
    env_config = config["env_config"]
    training_config = config["training_config"]

    save_path, writer = initialize_logging(config)
    
    if env_config["use_condor"]:
        collector = CondorCollector
        offpolicy_trainer_instance = offpolicy_trainer_condor
    else:
        collector = Collector
        offpolicy_trainer_instance = offpolicy_trainer

    train_collector = collector(policy, train_envs, buffer)
    training_args = training_config["training_args"]
    train_fn = generate_train_fn(config, policy, save_path)
    print("    >>>> Pre-collect experience")
    train_collector.collect(n_step=training_config['pre_collect'])

    result = offpolicy_trainer_instance(
        policy, 
        train_collector, 
        train_fn=train_fn, 
        writer=writer,
        **training_args
    )
    if env_config["use_condor"]:
        BASE_PATH = os.getenv('BUFFER_PATH')
        shutil.rmtree(BASE_PATH, ignore_errors=True)  # a way to force all the actors to stop
    else:
        train_envs.close()

if __name__ == "__main__":
    CONFIG_PATH = "configs/config.yaml"
    SAVE_PATH = "logging/"
    print(">>>>>>>> Loading the configuration from %s" % CONFIG_PATH)
    config = initialize_config(CONFIG_PATH, SAVE_PATH)

    seed(config)
    print(">>>>>>>> Creating the environments")
    train_envs = initialize_envs(config)
    env = train_envs if config["env_config"]["use_condor"] else train_envs.env[0]
    
    print(">>>>>>>> Initializing the policy")
    policy, buffer = initialize_policy(config, env)
    print(">>>>>>>> Start training")
    train(train_envs, policy, buffer, config)
