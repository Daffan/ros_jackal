from dwa_habitat import *
import argparse
import numpy as np
from collections import deque

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description = 'Start condor training')
    parser.add_argument('--habitat_index', dest='habitat_index', default=0)
    parser.add_argument('--seed', dest='seed', default=13)
    parser.add_argument('--save', dest='save', default="test_result.txt")
    parser.add_argument('--applr', dest="applr", default="")
    parser.add_argument('--repeat', dest='repeat', default=1)
    args = parser.parse_args()

    np.random.seed(args.seed)

    habitat_index = int(args.habitat_index)
    sp, ep = sample_start_goal_position(habitat_index)

    print(sp, ep)

    with open("configs/dwa_habitat.yaml") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    env_config = config["env_config"]
    env_config["kwargs"]["init_position"] = [*sp, np.arctan2( ep[1] - sp[1], ep[0] - sp[0])]
    env_config["kwargs"]["goal_position"] = [ep[0] - sp[0], ep[1] - sp[1], 0]
    env_config["kwargs"]["world_name"] = "habitat%d.sdf" %(habitat_index)
    env_config["kwargs"]["time_step"] = 1
    print(env_config["kwargs"]["init_position"], env_config["kwargs"]["goal_position"])

    env = gym.make(env_config["env_id"], **env_config["kwargs"])

    if args.applr:
        policy, _ = initialize_policy(config, env)
        policy.load(args.applr, "policy")
        policy.exploration_noise = 0

    for _ in range(args.repeat):
        obs = env.reset()
        steps = 0
        done = False
        pos = deque(maxlen=40)
        robot_pos = env.gazebo_sim.get_model_state().pose.position
        pos.append(robot_pos)

        while not done:
            if args.applr:  # dwa_habitat config has timestep = 0.2s, while the td3 policy is 1s
                actions = policy.select_action(obs)
            else:
                actions = env_config["kwargs"]["param_init"]
            obs, _, done, info = env.step(actions)
            steps += 1

            robot_pos = env.gazebo_sim.get_model_state().pose.position
            pos.append(robot_pos)
            if len(pos) == 40:
                if elucidate_distance(pos[0], pos[-1]) < 0.4:
                    break
        recovery, total = env.move_base.get_bad_vel_num()
        with open(args.save, "a") as f:
            f.write("%d %d %.4f %d %d %d\n" %(args.habitat_index, args.seed, info["time"], steps, recovery, info["success"]))

    env.close()

    