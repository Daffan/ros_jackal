from os.path import exists, join
from collections import defaultdict
import numpy as np
import os
import time
import logging
import re
import pickle
import shutil
import signal
from multiprocessing import Pool

try:
    from spython.main import Client as client
except:
    pass

BUFFER_PATH = os.getenv('BUFFER_PATH')
if not BUFFER_PATH:
    BUFFER_PATH = "local_buffer"

def run_actor_in_container(id=0):
    out = client.execute(
        join(BUFFER_PATH, "nav_benchmark.sif"),
        ['/bin/bash', '/jackal_ws/src/ros_jackal/entrypoint.sh', 'python3', 'actor.py', '--id=%d' %id],
        bind=['%s:%s' %(BUFFER_PATH, BUFFER_PATH), '%s:%s' %(os.getcwd(), "/jackal_ws/src/ros_jackal")],
        options=["-i", "-n", "--network=none", "-p"], nv=True
    )
    return out


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
            ep_rew += rew
            ep_len += 1
            n_steps_curr += 1
            self.global_steps += 1
            
            world = int(info['world'].split(
                "_")[-1].split(".")[0])  # task index
            collision_reward = -int(info['collided'])
            self.buffer.add(obs, act,
                            obs_new, rew,
                            done, world, collision_reward)
            obs = obs_new
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


class ContainerCollector(object):
    def __init__(self, policy, env, replaybuffer, config):
        '''
        This collector create Singularty containers and collect tracjories from them
        '''
        super().__init__()
        self.config = config
        self.policy = policy
        self.buffer = replaybuffer
        
        self.num_actor = config['container_config']['num_actor']
        self.ids = list(range(self.num_actor))
        self.ep_count = [0] * self.num_actor

        if not exists(BUFFER_PATH):
            os.mkdir(BUFFER_PATH)

        # save the current policy to the buffer 
        # which the actors in the containers can load from
        self.update_policy("policy")

        # save the env config the actor should read from
        shutil.copyfile(
            config["env_config"]["config_path"],
            join(BUFFER_PATH, "config.yaml")
        )

        # These two env variable ensure ROS running correctly in the container
        os.environ["ROS_HOSTNAME"] = "localhost"
        os.environ["ROS_MASTER_URI"] = "http://localhost:11311"

    def buffer_expand(self, traj):
        for i in range(len(traj)):
            state, action, reward, done, info = traj[i]
            state_next = traj[i+1][0] if i < len(traj)-1 else traj[i][0]
            world = int(info['world'].split(
                "_")[-1].split(".")[0])  # task index

            # For safeRL, separate the collision reward
            collision_reward = -int(info['collided'])
            assert collision_reward <= 0, "%.2f" %collision_reward

            self.buffer.add(state, action,
                            state_next, reward,
                            done, world, collision_reward)

    def natural_keys(self, text):
        return int(re.split(r'(\d+)', text)[1])

    def sort_traj_name(self, traj_files):
        ep_idx = np.array([self.natural_keys(fn) for fn in traj_files])
        idx = np.argsort(ep_idx)
        return np.array(traj_files)[idx]

    def update_policy(self, name):
        self.policy.save(BUFFER_PATH, "%s_copy" %name)
        # To prevent failure of actors when reading the saved policy
        shutil.move(
            join(BUFFER_PATH, "%s_copy_actor" %name),
            join(BUFFER_PATH, "%s_actor" %name)
        )
        shutil.move(
            join(BUFFER_PATH, "%s_copy_noise" %name),
            join(BUFFER_PATH, "%s_noise" %name)
        )
        if os.path.exists(join(BUFFER_PATH, "%s_copy_model" %name)):
            shutil.move(
                join(BUFFER_PATH, "%s_copy_model" %name),
                join(BUFFER_PATH, "%s_model" %name)
            )

    def collect(self, n_steps):
        """ This method searches the buffer folder and collect all the saved trajectories
        """
        # collect happens after policy is updated
        # save the updated policy to the buffer
        self.update_policy("policy")
        steps = 0
        results = []

        while steps < n_steps:
            # Launch containers to collect trajectories
            # Each subprocess is a container running an actor and collect 5 trajectories
            if not exists(join(BUFFER_PATH, "nav_benchmark.sif")):
                image, puller = client.pull('library://zifanxu/ros_jackal_image/image:latest', stream=True, pull_folder=BUFFER_PATH, name='nav_benchmark.sif')
                for line in puller:
                    print(line)
            
            with Pool(self.num_actor) as p:
                output = p.map(run_actor_in_container, self.ids)
            for o in output:
                print(o[0])
            
            np.random.shuffle(self.ids)
            for id in self.ids:
                id_steps, id_trajs, id_results = self.collect_worker_traj(id)
                steps += id_steps
                results.extend(id_results)
                for t in id_trajs:
                    self.buffer_expand(t)
        return steps, results
                
    def collect_worker_traj(self, id, skip_first=False):
        steps = 0
        trajs = []
        results = []
        base = join(BUFFER_PATH, 'actor_%d' % (id))
        try:
            traj_files = os.listdir(base)
        except:
            traj_files = []
        if skip_first:
            traj_files = self.sort_traj_name(traj_files)[:-1]
        else:
            traj_files = self.sort_traj_name(traj_files)
        for p in traj_files:
            try:
                target = join(base, p)
                if os.path.getsize(target) > 0:
                    with open(target, 'rb') as f:
                        traj = pickle.load(f)
                        ep_rew = sum([t[2] for t in traj])
                        ep_len = len(traj)
                        success = float(traj[-1][-1]['success'])
                        ep_time = traj[-1][-1]['time']
                        world = traj[-1][-1]['world']
                        collision = traj[-1][-1]['collision']
                        assert not np.isnan(ep_rew), "%s" %target
                        results.append(dict(ep_rew=ep_rew, ep_len=ep_len, success=success, ep_time=ep_time, world=world, collision=collision))
                        trajs.append(traj)
                        steps += ep_len
                    os.remove(join(base, p))
            except:
                logging.exception('')
                print("failed to load actor_%s:%s" % (id, p))
                # os.remove(join(base, p))
                pass
        return steps, trajs, results
