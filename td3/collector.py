from os.path import exists, join
from collections import defaultdict
import numpy as np
import os
import time
import logging
import re
import pickle
import shutil

BUFFER_PATH = os.getenv('BUFFER_PATH')


class LocalCollector(object):
    def __init__(self, policy, env, replaybuffer):
        self.policy = policy
        self.buffer = replaybuffer

    def collect(self, n_steps):
        raise NotImplementedError
        

class CondorCollector(object):
    def __init__(self, policy, env, replaybuffer, config):
        '''
        it's a fake tianshou Collector object with the same api
        '''
        super().__init__()
        self.config = config
        self.policy = policy
        self.buffer = replaybuffer
        
        self.num_actor = config['condor_config']['num_actor']
        self.validation_worlds = ["BARN/world_%d.world" %w if isinstance(w, int) else w \
            for w in config['condor_config']['validation_worlds']]
        self.ids = list(range(self.num_actor))
        self.validation_ids = [i + self.num_actor for i in list(range(2 * len(self.validation_worlds)))]
        self.ep_count = [0] * self.num_actor
        self.validation_results = None
        self.validation_step = None

        if not exists(BUFFER_PATH):
            os.mkdir(BUFFER_PATH)
        # save the current policy
        self.update_policy("policy")
        # save the env config the actor should read from
        shutil.copyfile(
            config["env_config"]["config_path"],
            join(BUFFER_PATH, "config.yaml")
        )

    def buffer_expand(self, traj):
        for i in range(len(traj)):
            state, action, reward, done, info = traj[i]
            state_next = traj[i+1][0] if i < len(traj)-1 else traj[i][0]
            world = int(info['world'].split(
                "_")[-1].split(".")[0])  # task index
            collision_reward = -int(info['collided'])
            assert collision_reward <= 0, "%.2f" %collision_reward

            if self.policy.safe_rl:
                self.buffer.add(state, action,
                                state_next, reward,
                                done, world, collision_reward)
            else:
                self.buffer.add(state, action,
                                state_next, reward,
                                done, world)

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

    def collect(self, n_step):
        """ This method searches the buffer folder and collect all the saved trajectories
        """
        # collect happens after policy is updated
        self.update_policy("policy")
        steps = 0
        results = []
        
        while steps < n_step:
            time.sleep(1)
            np.random.shuffle(self.ids)
            for id in self.ids:
                id_steps, id_trajs, id_results = self.collect_worker_traj(id)
                steps += id_steps
                results.extend(id_results)
                for t in id_trajs:
                    self.buffer_expand(t)
        return steps, results
    
    def collect_validation(self):
        assert self.validation_results is not None, "call set_validation first"
        for id in self.validation_ids:
            id_steps, id_trajs, id_results = self.collect_worker_traj(id, skip_first=False)
            # dict list to dict of dict list
            for i in range(len(id_results)):
                w = id_results[i]["world"]
                assert w in self.validation_worlds, w
                id_results[i].pop("world")
                self.validation_results[w].append(id_results[i])
        num_trials = self.config["condor_config"]["validation_trials"]
        for k in self.validation_worlds:
            if len(self.validation_results[k]) < num_trials:
                return None
                
        # make sure workers don't collect trajs anymore
        if os.path.exists(join(BUFFER_PATH, "validation_policy_actor")):
            os.remove(join(BUFFER_PATH, "validation_policy_actor"))
        if os.path.exists(join(BUFFER_PATH, "validation_policy_noise")):
            os.remove(join(BUFFER_PATH, "validation_policy_noise"))
        if os.path.exists(join(BUFFER_PATH, "validation_policy_model")):
            os.remove(join(BUFFER_PATH, "validation_policy_model"))
        
        # all validation collected
        ep_rew = []
        ep_len = []
        success = []
        ep_time = []
        collision = []
        for vv in self.validation_results.values():
            ep_rew.append(np.mean([v["ep_rew"] for v in vv][-num_trials:]))
            ep_len.append(np.mean([v["ep_len"] for v in vv][-num_trials:]))
            success.append(np.mean([v["success"] for v in vv][-num_trials:]))
            ep_time.append(np.mean([v["ep_time"] for v in vv][-num_trials:]))
            collision.append(np.mean([v["collision"] for v in vv][-num_trials:]))
        return dict(
            val_ep_rew=np.mean(ep_rew),
            val_ep_len=np.mean(ep_len),
            val_success=np.mean(success),
            val_ep_time=np.mean(ep_time),
            val_collision=np.mean(collision),
            val_step=self.validation_step
        )
            
    def set_validation(self, n_steps):
        val_results = None
        while val_results is None and n_steps > 0:
            time.sleep(1)
            val_results = self.collect_validation()
            
            # This will clear the trajs from actors (prevent disk exceeded) 
            for i in self.ids:
                self.collect_worker_traj(i)
        self.validation_step = n_steps
        self.update_policy("policy_%d" %n_steps)
        self.update_policy("validation_policy")
        self.validation_results = defaultdict(list)
        return val_results
                
    def collect_worker_traj(self, id, skip_first=True):
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
