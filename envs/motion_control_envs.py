from typing import Any, NamedTuple
from gym.spaces import Box
import numpy as np

import rospy
from geometry_msgs.msg import Twist

from envs.dwa_base_envs import DWABase, DWABaseLaser, DWABaseCostmap, DWABaseCostmapResnet

class MotionControlContinuous(DWABase):
    def __init__(self, min_v=-1, max_v=2, min_w=-3.14, max_w=3.14, **kwargs):
        super().__init__(**kwargs)
        self._cmd_vel_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=1)
        self.params = None
        # same as the parameters to tune
        
        self.RANGE_DICT = RANGE_DICT = {
            "linear_velocity": [min_v, max_v],
            "angular_velocity": [min_w, max_w],
        }
        self.action_space = Box(
            low=np.array([RANGE_DICT["linear_velocity"][0], RANGE_DICT["angular_velocity"][0]]),
            high=np.array([RANGE_DICT["linear_velocity"][1], RANGE_DICT["angular_velocity"][1]]),
            dtype=np.float32
        )

    def reset(self):
        """reset the environment without setting the goal
        set_goal is replaced with make_plan
        """
        self.step_count = 0
        self.action = np.zeros(self.action_space.shape)
        # Reset robot in odom frame clear_costmap
        self.gazebo_sim.unpause()
        self.move_base.reset_robot_in_odom()
        # Resets the state of the environment and returns an initial observation
        self.gazebo_sim.reset()
        self.move_base.make_plan()
        self._clear_costmap()
        self.start_time = self.time = rospy.get_time()
        
        pose = self.gazebo_sim.get_model_state().pose
        pos = pose.position
        ori = pose.orientation.w
        self.traj_pos.append((pos))
        self.traj_ori.append(ori)
        
        self.collided = False
        obs = self._get_observation()
        self.gazebo_sim.pause()
        self.collision_count = 0
        self.smoothness = 0
        return obs

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
        super()._take_action(action)  # this will wait util next time step
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
