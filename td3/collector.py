from os.path import exists, join
import numpy as np
import yaml
import os
import torch
import time
import logging
import re
import pickle
import shutil

BUFFER_PATH = os.getenv('BUFFER_PATH')


class LocalCollector(object):
    def __init__(self, policy, env, replaybuffer):
        self.policy = policy
        self.env = env
        self.buffer = replaybuffer
        
        self.last_obs = None
        
        self.global_episodes = 0
        self.global_steps = 0

    def collect(self, n_steps):
        n_steps_curr = 0
        env = self.env
        policy = self.policy
        results = []
        
        ep_rew = 0
        ep_len = 0
        
        if self.last_obs is not None:
            obs = self.last_obs
        else:
            obs = env.reset()
        while n_steps_curr < n_steps:
            act = policy.select_action(obs)
            obs_new, rew, done, info = env.step(act)
            obs = obs_new
            ep_rew += rew
            ep_len += 1
            n_steps_curr += 1
            self.global_steps += 1
            
            world = int(info['world'].split(
                "_")[-1].split(".")[0])  # task index
            
            self.buffer.add(obs, act,
                            obs_new, rew,
                            done, world)
            if done:
                obs = env.reset()
                results.append(dict(
                    ep_rew=ep_rew, 
                    ep_len=ep_len, 
                    success=info['success'], 
                    ep_time=info['time'], 
                    world=info['world'], 
                    collision=info['collision']
                ))
                ep_rew = 0
                ep_len = 0
                self.global_episodes += 1
            print("n_episode: %d, n_steps: %d" %(self.global_episodes, self.global_steps), end="\r")
        self.last_obs = obs
        return n_steps_curr, results
        

class CondorCollector(object):
    def __init__(self, policy, env, replaybuffer):
        '''
        it's a fake tianshou Collector object with the same api
        '''
        super().__init__()
        self.policy = policy
        self.num_actor = env.config['condor_config']['num_actor']
        self.ids = list(range(self.num_actor))
        self.ep_count = [0]*self.num_actor
        self.buffer = replaybuffer

        if not exists(BUFFER_PATH):
            os.mkdir(BUFFER_PATH)
        # save the current policy
        self.update_policy()
        # save the env config the actor should read from
        shutil.copyfile(
            env.config["env_config"]["config_path"],
            join(BUFFER_PATH, "config.yaml")
        )

    def buffer_expand(self, traj):
        for i in range(len(traj)):
            state, action, reward, done, info = traj[i]
            state_next = traj[i+1][0] if i < len(traj)-1 else traj[i][0]
            world = int(info['world'].split(
                "_")[-1].split(".")[0])  # task index
            self.buffer.add(state, action,
                            state_next, reward,
                            done, world)

    def natural_keys(self, text):
        return int(re.split(r'(\d+)', text)[1])

    def sort_traj_name(self, traj_files):
        ep_idx = np.array([self.natural_keys(fn) for fn in traj_files])
        idx = np.argsort(ep_idx)
        return np.array(traj_files)[idx]

    def update_policy(self):
        self.policy.save(BUFFER_PATH, "policy_copy")
        # To prevent failure of actors when reading the saved policy
        shutil.move(
            join(BUFFER_PATH, "policy_copy_actor"),
            join(BUFFER_PATH, "policy_actor")
        )
        shutil.move(
            join(BUFFER_PATH, "policy_copy_noise"),
            join(BUFFER_PATH, "policy_noise")
        )

    def collect(self, n_step):
        """ This method searches the buffer folder and collect all the saved trajectories
        """
        # collect happens after policy is updated
        self.update_policy()
        steps = 0
        results = []
        while steps < n_step:
            time.sleep(1)
            np.random.shuffle(self.ids)
            for id in self.ids:
                base = join(BUFFER_PATH, 'actor_%d' % (id))
                try:
                    traj_files = os.listdir(base)
                except:
                    traj_files = []
                traj_files = self.sort_traj_name(traj_files)[:-1]
                for p in traj_files:
                    try:
                        target = join(base, p)
                        if os.path.getsize(target) > 0:
                            if steps < n_step:  # if reach the target steps, don't put the experinece into the buffer
                                with open(target, 'rb') as f:
                                    traj = pickle.load(f)
                                    ep_rew = sum([t[2] for t in traj])
                                    ep_len = len(traj)
                                    success = float(traj[-1][-1]['success'])
                                    ep_time = traj[-1][-1]['time']
                                    world = traj[-1][-1]['world']
                                    collision = traj[-1][-1]['collision']
                                    results.append(dict(ep_rew=ep_rew, ep_len=ep_len, success=success, ep_time=ep_time, world=world, collision=collision))
                                    self.buffer_expand(traj)
                                    steps += ep_len
                            os.remove(join(base, p))
                    except:
                        logging.exception('')
                        print("failed to load actor_%s:%s" % (id, p))
                        # os.remove(join(base, p))
                        pass
        return steps, results