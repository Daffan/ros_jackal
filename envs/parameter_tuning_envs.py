from envs.dwa_base import DWABase
from gym.spaces import Box
import numpy as np
import rospy

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

        # 720 laser scan + local goal (in angle)
        self.observation_space = Box(
            low=np.array([-1]*(721)),
            high=np.array([-1]*(721)),
            dtype=np.float32
        )
        # same as the parameters to tune
        self.action_space = Box(
            low=np.array([RANGE_DICT[k][0] for k in self.param_list]),
            high=np.array([RANGE_DICT[k][1] for k in self.param_list]),
            dtype=np.float32
        )
        self.reward_range = (-np.inf, np.inf)

    def _get_observation(self):
        # observation is the 720 dim laser scan + one local goal in angle
        laser_scan = self._get_laser_scan()
        local_goal = self._get_local_goal()
        
        laser_scan = (laser_scan - self.laser_clip/2.) / self.laser_clip # scale to (-0.5, 0.5)
        local_goal = local_goal / (2.0 * np.pi) # scale to (-0.5, 0.5)

        obs = np.concatenate([laser_scan, local_goal])

        return obs

    def _get_done(self):
        success = self._get_success()
        done = success or self.step_count >= self.max_step
        return done

    def _get_success(self):
        # check the robot distance to the goal position
        robot_position = np.array([self.move_base.robot_config.X, 
                                   self.move_base.robot_config.Y]) # robot position in odom frame
        goal_position = np.array(self.goal_position[:2])
        self.goal_distance = np.sqrt(np.sum((robot_position - goal_position) ** 2))
        return self.goal_distance < 0.4

    def _get_reward(self):
        # we now use +10 for getting the goal, else 0
        rew = 10 if self._get_success() else 0
        return rew

    def _get_info(self):
        return dict(success=self._get_success(), params=self.params)

    def _take_action(self, action):
        assert len(action) == len(self.param_list), "length of the params should match the length of the action"
        self.params = action
        # Set the parameters
        for param_value, param_name in zip(action, self.param_list):
            high_limit = RANGE_DICT[param_name][1]
            low_limit = RANGE_DICT[param_name][0]
            param_value = float(np.clip(param_value, low_limit, high_limit))
            self.move_base.set_navi_param(param_name, param_value)
        # Wait for robot to navigate for one time step
        rospy.sleep(self.time_step)