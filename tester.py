from actor import initialize_actor, write_buffer, _debug_print_robot_status
from train import initialize_policy
import os
from os.path import dirname, abspath
import gym
import argparse
# from habitat_utils import sample_start_goal_position
import numpy as np

import sys
sys.path.append(dirname(dirname(abspath(__file__))))
from envs.wrappers import StackFrame

BUFFER_PATH = os.getenv('BUFFER_PATH')

def get_world_name(config, id):
    # We test each test world with two actors, so duplicate the lict by a factor of two
    world_name = (config["condor_config"]["test_worlds"] * 2)[id]
    if isinstance(world_name, int):
        world_name = "BARN/world_%d.world" %(world_name)
    return world_name

def load_policy(policy):
    policy.load(BUFFER_PATH, "policy")
    policy.exploration_noise = 0.0
    return policy

def main(args):
    config = initialize_actor(args.id)
    num_trials = config["condor_config"]["num_trials"]
    env_config = config['env_config']
    world_name = get_world_name(config, args.id)
    test_object = "dwa"  # config["condor_config"]["test_object"]

    env_config["kwargs"]["world_name"] = world_name
    env = gym.make(env_config["env_id"], **env_config["kwargs"])
    env = StackFrame(env, stack_frame=env_config["stack_frame"]) 

    policy, _ = initialize_policy(config, env, init_buffer=False)
    policy = load_policy(policy)
    
    if config["training_config"]["MPC"]:
        policy.exploration_noise = config["training_config"]["exploration_noise_end"]
        print("exploration_noise: %.2f" %(policy.exploration_noise))

    print(">>>>>>>>>>>>>> Running on %s <<<<<<<<<<<<<<<<" %(world_name))
    ep = 0
    bad_traj_count = 0
    while ep < num_trials:
        if world_name.startswith("habitat"):
            habitat_index = int(world_name.split("habitat")[-1].split(".")[0])
            sp, ep = sample_start_goal_position(habitat_index)
            init_position = [*sp, np.arctan2( ep[1] - sp[1], ep[0] - sp[0])]
            goal_position = [ep[0], ep[1], 0]
            env.set_init_position(init_position)
            env.set_goal_position(goal_position)
            print(">>>>>>>>>>>>>>>>>", init_position)
            print(">>>>>>>>>>>>>>>>>", goal_position)
        obs = env.reset()
        traj = []
        done = False
        while not done:
            actions = policy.select_action(obs)
            obs_new, rew, done, info = env.step(actions)
            info["world"] = world_name
            traj.append([None, None, rew, done, info])  # For testing, we only need rew and ep_length
            obs = obs_new

            _debug_print_robot_status(env, len(traj), rew, actions)
        
        ep = write_buffer(traj, args.id)
        #time_per_step = info['time'] / len(traj)  # sometimes, the simulation runs very slow, need restart
        #if len(traj) >= 1 and time_per_step < (0.05 + config["env_config"]["kwargs"]["time_step"]):
        #    ep = write_buffer(traj, args.id)
        #else:  # for some reason, the progress might just dead or always give fail traj with only 1 step
        #    bad_traj_count += 1
        #    if bad_traj_count >= 5:
        #        break

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = 'start an tester')
    parser.add_argument('--id', dest='id', type = int, default = 1)
    args = parser.parse_args()
    main(args)
