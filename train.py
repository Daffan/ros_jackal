import argparse
import GPUtil
import yaml
import numpy as np
import gym
from datetime import datetime
from os.path import join, dirname, abspath, exists
import sys
import os
import shutil
import logging
import collections
import time
import uuid
from pprint import pformat

import torch
from tensorboardX import SummaryWriter

from envs import registration
from envs.wrappers import StackFrame
from rl_algos import algo_class
from rl_algos.net import *
from rl_algos.base_rl_algo import ReplayBuffer
from rl_algos.sac import GaussianActor
from rl_algos.td3 import Actor, Critic #, TD3, ReplayBuffer
from rl_algos.model_based import Model
# from rl_algos.safe_td3 import SafeTD3
from rl_algos.collector import ContainerCollector, LocalCollector

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
    string = now.strftime("%Y_%m_%d_%H_%M")

    save_path = join(
        env_config["save_path"], 
        env_config["env_id"], 
        training_config['algorithm'], 
        string,
        uuid.uuid4().hex[:4]
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
    if env_config["use_container"]:
        env_config["kwargs"]["init_sim"] = False

    env = gym.make(env_config["env_id"], **env_config["kwargs"])
    env = StackFrame(env, stack_frame=env_config["stack_frame"])
    return env

def seed(config):
    env_config = config["env_config"]
    np.random.seed(env_config['seed'])
    torch.manual_seed(env_config['seed'])

def get_encoder(encoder_type, args):
    if encoder_type == "mlp":
        encoder=MLPEncoder(**args)
    elif encoder_type == 'rnn':
        encoder=RNNEncoder(**args)
    elif encoder_type == 'cnn':
        encoder=CNNEncoder(**args)
    elif encoder_type == 'transformer':
        encoder=TransformerEncoder(**args)
    else:
        raise Exception(f"[error] Unknown encoder type {encoder_type}!")
    return encoder

def initialize_policy(config, env, init_buffer=True, device=None):
    training_config = config["training_config"]

    state_dim = env.observation_space.shape
    action_dim = np.prod(env.action_space.shape)
    action_space_low = env.action_space.low
    action_space_high = env.action_space.high

    # find available device
    if device is None:
        devices = GPUtil.getAvailable(order = 'first', limit = 1, maxLoad = 0.8, maxMemory = 0.8, includeNan=False, excludeID=[], excludeUUID=[])
        device = "cuda:%d" %(devices[0]) if len(devices) > 0 else "cpu"
    print("    >>>> Running on device %s" %(device))

    encoder_type = training_config["encoder"]
    encoder_args = {
        'input_dim': state_dim[-1],  # np.prod(state_dim),
        'num_layers': training_config['encoder_num_layers'],
        'hidden_size': training_config['encoder_hidden_layer_size'],
        'history_length': config["env_config"]["stack_frame"],
    }

    # initialize actor
    input_dim = training_config['hidden_layer_size']
    actor_class = GaussianActor if "SAC" in training_config["algorithm"] else Actor
    actor = actor_class(
        encoder=get_encoder(encoder_type, encoder_args),
        head=MLP(input_dim, training_config['encoder_num_layers'], training_config['encoder_hidden_layer_size']),
        action_dim=action_dim
    ).to(device)
    actor_optim = torch.optim.Adam(
        actor.parameters(), 
        lr=training_config['actor_lr']
    )
    # print("Total number of parameters: %d" %sum(p.numel() for p in actor.parameters()))

    # initialize critic
    input_dim += np.prod(action_dim)
    critic = Critic(
        encoder=get_encoder(encoder_type, encoder_args),
        head=MLP(input_dim, training_config['encoder_num_layers'], training_config['encoder_hidden_layer_size']),
    ).to(device)
    critic_optim = torch.optim.Adam(
        critic.parameters(), 
        lr=training_config['critic_lr']
    )

    # initialize agents
    algo = training_config["algorithm"]
    if "Dyna" in algo or "SMCP" in algo or "MBPO" in algo:
        model = Model(
            encoder=get_encoder(encoder_type, encoder_args),
            head=MLP(input_dim, training_config['encoder_num_layers'], training_config['encoder_hidden_layer_size']),
            state_dim=state_dim,
            deterministic=training_config['deterministic']
        ).to(device)
        model_optim = torch.optim.Adam(
            model.parameters(), 
            lr=training_config['model_lr']
        )
        policy = algo_class[algo](
            model, model_optim,
            actor, actor_optim,
            critic, critic_optim,
            action_range=[action_space_low, action_space_high],
            device=device,
            **training_config["policy_args"]
        )
    elif "Safe" in algo:
        safe_critic = Critic(
            encoder=get_encoder(encoder_type, encoder_args),
            head=MLP(input_dim, training_config['encoder_num_layers'], training_config['encoder_hidden_layer_size']),
        ).to(device)
        safe_critic_optim = torch.optim.Adam(
            safe_critic.parameters(), 
            lr=training_config['critic_lr']
        )
        policy = algo_class[algo](
            safe_critic, safe_critic_optim,
            actor, actor_optim,
            critic, critic_optim,
            action_range=[action_space_low, action_space_high],
            device=device,
            **training_config["policy_args"]
        )
    else:
        policy = algo_class[algo](
            actor, actor_optim,
            critic, critic_optim,
            action_range=[action_space_low, action_space_high],
            device=device,
            **training_config["policy_args"]
        )
    
    if init_buffer:
        replay_buffer = ReplayBuffer(
            state_dim, action_dim, training_config['buffer_size'],
            device=device,
            reward_norm=False  # config['training_config']["reward_norm"]
        )
    else:
        replay_buffer = None

    return policy, replay_buffer

def train(env, policy, replay_buffer, config):
    env_config = config["env_config"]
    training_config = config["training_config"]

    save_path, writer = initialize_logging(config)
    print("    >>>> initialized logging")
    
    if env_config["use_container"]:
        collector = ContainerCollector(policy, env, replay_buffer, config)
    else:
        collector = LocalCollector(policy, env, replay_buffer)

    training_args = training_config["training_args"]
    print("    >>>> Pre-collect experience")
    collector.collect(n_steps=training_config['pre_collect'])
    print("    >>>> Start training")

    n_steps = 0
    n_iter = 0
    n_ep = 0
    epinfo_buf = collections.deque(maxlen=300)
    world_ep_buf = collections.defaultdict(lambda: collections.deque(maxlen=20))
    t0 = time.time()
    
    while n_steps < training_args["max_step"]:
        # Linear decaying exploration noise from "start" -> "end"
        if "TD3" in training_config["algorithm"]:
            policy.exploration_noise = \
                - (training_config["exploration_noise_start"] - training_config["exploration_noise_end"]) \
                *  n_steps / training_args["max_step"] + training_config["exploration_noise_start"]
        steps, epinfo = collector.collect(n_steps=training_args["collect_per_step"])
        
        n_steps += steps
        n_iter += 1
        n_ep += len(epinfo)
        epinfo_buf.extend(epinfo)
        for d in epinfo:
            world = d["world"].split("/")[-1]
            world_ep_buf[world].append(d)

        loss_infos = []
        for _ in range(training_args["update_per_step"]):
            loss_info = policy.train(replay_buffer, training_args["batch_size"])
            loss_infos.append(loss_info)

        loss_info = {}
        for k in loss_infos[0].keys():
            loss_info[k] = np.mean([li[k] for li in loss_infos if li[k] is not None])

        t1 = time.time()
        log = {
            "Episode_return": np.mean([epinfo["ep_rew"] for epinfo in epinfo_buf]),
            "Episode_length": np.mean([epinfo["ep_len"] for epinfo in epinfo_buf]),
            "Success": np.mean([epinfo["success"] for epinfo in epinfo_buf]),
            "Time": np.mean([epinfo["ep_time"] for epinfo in epinfo_buf]),
            "Collision": np.mean([epinfo["collision"] for epinfo in epinfo_buf]),
            "fps": n_steps / (t1 - t0),
            "n_episode": n_ep,
            "Steps": n_steps
        }
        if "TD" in training_config["algorithm"] or "DDPG" in training_config["algorithm"]:
            log.update({
                "Exploration_noise": policy.exploration_noise,
            })
        if "SAC" in training_config["algorithm"]:
            log.update({
                "Alpha": policy.alpha,
            })
        log.update(loss_info)
        print(pformat(log))

        if n_iter % training_config["log_intervals"] == 0:
            for k in log.keys():
                writer.add_scalar('train/' + k, log[k], global_step=n_steps)
            policy.save(save_path, "last_policy")
            print("Logging to %s" %save_path)

            for k in world_ep_buf.keys():
                writer.add_scalar(k + "/Episode_return", np.mean([epinfo["ep_rew"] for epinfo in world_ep_buf[k]]), global_step=n_steps)
                writer.add_scalar(k + "/Episode_length", np.mean([epinfo["ep_len"] for epinfo in world_ep_buf[k]]), global_step=n_steps)
                writer.add_scalar(k + "/Success", np.mean([epinfo["success"] for epinfo in world_ep_buf[k]]), global_step=n_steps)
                writer.add_scalar(k + "/Time", np.mean([epinfo["ep_time"] for epinfo in world_ep_buf[k]]), global_step=n_steps)
                writer.add_scalar(k + "/Collision", np.mean([epinfo["collision"] for epinfo in world_ep_buf[k]]), global_step=n_steps)

if __name__ == "__main__":
    torch.set_num_threads(8)
    parser = argparse.ArgumentParser(description = 'Start training')
    parser.add_argument('--config_path', dest='config_path', default="configs/e2e_default.yaml")
    parser.add_argument('--device', dest='device', default=None)
    logging.getLogger().setLevel("INFO")
    args = parser.parse_args()
    CONFIG_PATH = args.config_path
    SAVE_PATH = "logging/"
    print(">>>>>>>> Loading the configuration from %s" % CONFIG_PATH)
    config = initialize_config(CONFIG_PATH, SAVE_PATH)

    seed(config)
    print(">>>>>>>> Creating the environments")
    env = initialize_envs(config)
    
    print(">>>>>>>> Initializing the policy")
    policy, replay_buffer = initialize_policy(config, env, device=args.device)

    print(">>>>>>>> Start training")
    train(env, policy, replay_buffer, config)
