from gym.spaces import Discrete, Box
import numpy as np

import sys
from os.path import dirname, abspath
sys.path.append(dirname(dirname(abspath(__file__))))
from envs.parameter_tuning_envs import RANGE_DICT

class InfoEnv:
    """ The infomation environment contains observation space and action space infomation only
    """
    def __init__(self, config):
        env_config = config["env_config"]
        env_id = env_config["env_id"]
        if env_id.startswith("dwa_param_continuous"):
            action_space = Box(
                low=np.array([RANGE_DICT[k][0] for k in env_config["kwargs"]["param_list"]]),
                high=np.array([RANGE_DICT[k][1] for k in env_config["kwargs"]["param_list"]]),
                dtype=np.float32
            )
        elif env_id.startswith("motion_control_continuous"):
            action_space = Box(
                low=np.array([0.1, -3.14]),
                high=np.array([2, 3.14]),
                dtype=np.float32
            )
        else:
            raise NotImplementedError
        
        if "laser" in env_id:
            observation_space = Box(
                low=0,
                high=env_config["kwargs"]["laser_clip"],
                shape=(721,),
                dtype=np.float32
            )
        elif "costmap" in env_id:
            observation_space = Box(
                low=-1,
                high=10,
                shape=(1, 84, 84),
                dtype=np.float32
            )
        else:
            raise NotImplementedError

        self.observation_space = observation_space
        self.action_space = action_space
        self.config = config
