from typing import Any, NamedTuple
from gym.spaces import Box
import numpy as np

import rospy
from geometry_msgs.msg import Twist

from envs.dwa_base_envs import DWABase, DWABaseLaser, DWABaseCostmap, DWABaseCostmapResnet

class MotionControlContinuous(DWABase):
    def __init__(self, collision_reward=-0.1, **kwargs):
        super().__init__(**kwargs)
        self.collision_reward = collision_reward
        self._cmd_vel_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=1)
        self.params = None
        # same as the parameters to tune
        self.action_space = Box(
            low=np.array([-0.2, -3.14]),
            high=np.array([2, 3.14]),
            dtype=np.float32
        )
        self.move_base = self.launch_move_base(goal_position=self.goal_position, base_local_planner=self.base_local_planner)

    def reset(self):
        """reset the environment without setting the goal
        set_goal is replaced with make_plan
        """
        self.step_count=0
        # Reset robot in odom frame clear_costmap
        self.gazebo_sim.unpause()
        self.move_base.reset_robot_in_odom()
        # Resets the state of the environment and returns an initial observation
        self.gazebo_sim.reset()
        self.move_base.make_plan()
        self._clear_costmap()
        self.start_time = rospy.get_time()
        obs = self._get_observation()
        self.gazebo_sim.pause()
        return obs

    def _get_reward(self):
        rew = super()._get_reward()
        # This part handles possible collision
        laser_scan = np.array(self.gazebo_sim.get_laser_scan().ranges)
        d = np.mean(sorted(laser_scan)[:10])
        if d < 0.3:  # minimum distance 0.3 meter 
            rew += self.collision_reward / (d + 0.05)
        return rew

    def _get_info(self):
        info = dict(success=self._get_success(), params=self.params)
        info.update(super()._get_info())
        return info

    def _take_action(self, action):
        linear_speed, angular_speed = action
        cmd_vel_value = Twist()
        cmd_vel_value.linear.x = linear_speed
        cmd_vel_value.angular.z = angular_speed

        self.gazebo_sim.unpause()
        self._cmd_vel_pub.publish(cmd_vel_value)
        self.move_base.make_plan()
        rospy.sleep(self.time_step)
        self.gazebo_sim.pause()


class MotionControlContinuousLaser(MotionControlContinuous, DWABaseLaser):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)


class MotionControlContinuousCostmap(MotionControlContinuous, DWABaseCostmap):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)


class MotionControlContinuousCostmapResnet(MotionControlContinuous, DWABaseCostmapResnet):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
