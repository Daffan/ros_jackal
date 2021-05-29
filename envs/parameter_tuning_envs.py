from typing import Any, NamedTuple
from gym.spaces import Box
import numpy as np
import rospy

from envs.dwa_base_envs import DWABase, DWABaseLaser, DWABaseCostmap

# A contant dict that define the ranges of parameters
RANGE_DICT = {
    'max_vel_x': [0.2, 2],
    'max_vel_theta': [0.314, 3.14],
    'vx_samples': [4, 12],
    'vtheta_samples': [8, 40],
    'path_distance_bias': [0.1, 1.5],
    'goal_distance_bias': [0.1, 2],
    'inflation_radius': [0.1, 0.6]
}

class DWAParamContinuous(DWABase):
    def __init__(
        self, 
        param_init=[0.5, 1.57, 6, 20, 0.75, 1, 0.3],
        param_list=['max_vel_x', 
                    'max_vel_theta', 
                    'vx_samples', 
                    'vtheta_samples', 
                    'path_distance_bias', 
                    'goal_distance_bias', 
                    'inflation_radius'],
        **kwargs
    ):
        super().__init__(**kwargs)
        self.param_list = param_list
        self.param_init = param_init

        # same as the parameters to tune
        self.action_space = Box(
            low=np.array([RANGE_DICT[k][0] for k in self.param_list]),
            high=np.array([RANGE_DICT[k][1] for k in self.param_list]),
            dtype=np.float32
        )

    def _get_info(self):
        info = dict(success=self._get_success(), params=self.params)
        info.update(super()._get_info())
        return info

    def _take_action(self, action):
        assert len(action) == len(self.param_list), "length of the params should match the length of the action"
        self.params = action
        # Set the parameters
        self.gazebo_sim.unpause()
        for param_value, param_name in zip(action, self.param_list):
            high_limit = RANGE_DICT[param_name][1]
            low_limit = RANGE_DICT[param_name][0]
            param_value = float(np.clip(param_value, low_limit, high_limit))
            self.move_base.set_navi_param(param_name, param_value)
        # Wait for robot to navigate for one time step
        rospy.sleep(self.time_step)
        self.gazebo_sim.pause()


class DWAParamContinuousLaser(DWAParamContinuous, DWABaseLaser):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)


class DWAParamContinuousCostmap(DWAParamContinuous, DWABaseCostmap):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
