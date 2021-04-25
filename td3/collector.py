from os.path import exists, join
import yaml
import os
import torch
import time
import logging
import re
import pickle

BASE_PATH = join(os.getenv('HOME'), 'buffer')
class Collector(object):

    def __init__(self, policy, env, replaybuffer):
        '''
        it's a fake tianshou Collector object with the same api
        '''
        super().__init__()
        self.policy = policy
        self.env = env
        self.num_actor = env['condor_config']['num_actor']
        self.ids = list(range(self.num_actor))
        self.ep_count = [0]*self.num_actor
        self.buffer = replaybuffer

        if not exists(BASE_PATH):
            os.mkdir(BASE_PATH)
        # save the current policy
        self.update_policy()
        # save the env config the actor should read from
        with open(join(BASE_PATH, 'config.yaml'), 'w') as fp:
            yaml.dump(self.env.config, fp)

    def update_policy(self):
        torch.save(self.policy.state_dict(), join(BASE_PATH, 'policy.pth'))
        with open(join(BASE_PATH, 'eps.txt'), 'w') as f:
            try:
                std = self.policy._noise._sigma
            except:
                std = self.policy._noise
            f.write(str(std))

    def buffer_expand(self, traj):
        for i in range(len(traj)):
            obs_next = traj[i+1][0] if i < len(traj)-1 else traj[i][0]
            self.buffer.add(traj[i][0], traj[i][1], \
                            traj[i][2], traj[i][3], \
                            obs_next, traj[i][4])
    
    def natural_keys(self, text):
        return int(re.split(r'(\d+)', text)[1])

    def collect(self, n_step):
        # collect happens after policy is updated
        self.update_policy()
        steps = 0
        ep_rew = []
        ep_len = []
        succeed = []
        world = []
        while steps < n_step:
            for id in self.ids:
                base = join(BASE_PATH, 'actor_%d' %(id))
                try:
                    traj_files = os.listdir(base)
                except:
                    traj_files = [] #["traj_0.pickle"]
                for p in traj_files:
                    try:
                        target = join(base, p)
                        if os.path.getsize(target) > 0:
                            with open(target, 'rb') as f:
                                traj = pickle.load(f)
                                ep_rew.append(sum([t[2] for t in traj]))
                                ep_len.append(len(traj))
                                succeed.append(int(traj[-1][-1]['success']))
                                world.append(traj[-1][-1]['world'])
                                self.buffer_expand(traj)
                                steps += len(traj)
                            os.remove(join(base, p))
                    except:
                        logging.exception('')
                        print("failed to load actor_%s:%s" %(id, p))
                        os.remove(join(base, p))
                        pass
        return {'n/st': steps, 'n/stt': steps, 'ep_rew': ep_rew, 'ep_len': ep_len, 'success': succeed, 'world': world}

