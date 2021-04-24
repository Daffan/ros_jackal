from typing import Any, NamedTuple
from gym.spaces import Box
import numpy as np

import rospy
from geometry_msgs.msg import Twist

from envs.dwa_base import DWABase, DWABaseLaser, DWABaseCostmap

class MotionControlContinuous(DWABase):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._cmd_vel_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=1)

        # same as the parameters to tune
        self.action_space = Box(
            low=np.array([0.1, 0.314]),
            high=np.array([2, 3.14]),
            dtype=np.float32
        )

        self.reward_range = (0, 10)

    def reset(self):
        """reset the environment without setting the goal
        set_goal is replaced with make_plan
        """
        self.step_count=0
        # Reset robot in odom frame clear_costmap
        self.move_base.reset_robot_in_odom()
        # Resets the state of the environment and returns an initial observation
        self.gazebo_sim.reset()
        self.move_base.make_plan()
        self._clear_costmap()
        return self._get_observation()

    def step(self, action):
        self.move_base.make_plan()
        return super().step(action) 

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
        return dict(success=self._get_success())

    def _take_action(self, action):
        linear_speed, angular_speed = action
        cmd_vel_value = Twist()
        cmd_vel_value.linear.x = linear_speed
        cmd_vel_value.angular.z = angular_speed

        self._cmd_vel_pub.publish(cmd_vel_value)
        rospy.sleep(self.time_step)


class MotionControlContinuousLaser(MotionControlContinuous, DWABaseLaser):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)


class MotionControlContinuousCostmap(MotionControlContinuous, DWABaseCostmap):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)