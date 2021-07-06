import gym
import rospy
import rospkg
import roslaunch
import time
import numpy as np
import os
import cv2
from os.path import dirname, join, abspath
import subprocess
from gym.spaces import Box, Discrete
from geometry_msgs.msg import Twist

from envs.move_base import MoveBase
from dwa_base_envs import DWABaseLaser, DWABaseCostmap
from parameter_tuning_envs import RANGE_DICT

class RealRobotEnv(gym.Env):
    def __init__(
        self,
        base_local_planner="base_local_planner/TrajectoryPlannerROS",
        world_name="jackal_world.world",
        gui=False,
        init_position=[0, 0, 0],
        goal_position=[4, 0, 0],
        max_step=100,
        time_step=1,
        slack_reward=-1,
        failure_reward=-50,
        success_reward=0,
        verbose=True
    ):
        """Base RL env that initialize jackal simulation in Gazebo
        """
        super().__init__()

        self.base_local_planner = base_local_planner
        self.world_name = world_name
        self.gui = gui
        self.init_position = init_position
        self.goal_position = goal_position
        self.verbose = verbose
        self.time_step = time_step
        self.max_step = max_step
        self.slack_reward = slack_reward
        self.failure_reward = failure_reward
        self.success_reward = success_reward

        # launch move_base
        rospy.logwarn(">>>>>>>>>>>>>>>>>> Load world: %s <<<<<<<<<<<<<<<<<<" %(world_name))
        rospack = rospkg.RosPack()
        self.BASE_PATH = rospack.get_path('jackal_helper')
        world_name = join(self.BASE_PATH, "worlds", world_name)
        launch_file = join(self.BASE_PATH, 'launch', 'move_base_launch.launch')

        self.gazebo_process = subprocess.Popen(['roslaunch', 
                                                launch_file,
                                                'base_local_planner:=' + base_local_planner
                                                ])
        time.sleep(10)  # sleep to wait until the gazebo being created

        # initialize the node for gym env
        rospy.init_node('gym', anonymous=True, log_level=rospy.FATAL)

        self.move_base = MoveBase(goal_position=self.goal_position, base_local_planner=base_local_planner)

        # Not implemented
        self.action_space = None
        self.observation_space = None
        self.reward_range = (
            min(slack_reward, failure_reward), 
            success_reward
        )

        self.step_count = 0

    def reset(self):
        """reset the environment
        """
        self.step_count=0
        self.move_base.set_global_goal()
        self._clear_costmap()
        self.start_time = rospy.get_time()
        obs = self._get_observation()
        return obs

    def _clear_costmap(self):
        self.move_base.clear_costmap()
        rospy.sleep(0.1)
        self.move_base.clear_costmap()
        rospy.sleep(0.1)
        self.move_base.clear_costmap()

    def step(self, action):
        """take an action and step the environment
        """
        self._take_action(action)
        self.step_count += 1
        obs = self._get_observation()
        return obs, 0, False, {}

    def _get_local_goal(self):
        """get local goal in angle
        Returns:
            float: local goal in angle
        """
        local_goal = self.move_base.get_local_goal()
        local_goal = np.array([np.arctan2(local_goal.position.y, local_goal.position.x)])
        return local_goal

    def _get_observation(self):
        raise NotImplementedError()


class RealRobotLaser(RealRobotEnv, DWABaseLaser):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)


class RealRobotCostmap(RealRobotEnv, DWABaseCostmap):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)


class RealRobotMotionControlContinuous(RealRobotEnv):
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

    def reset(self):
        """reset the environment without setting the goal
        set_goal is replaced with make_plan
        """
        self.step_count = 0
        self.move_base.make_plan()
        self._clear_costmap()
        self.start_time = rospy.get_time()
        obs = self._get_observation()
        return obs

    def _take_action(self, action):
        linear_speed, angular_speed = action
        cmd_vel_value = Twist()
        cmd_vel_value.linear.x = linear_speed
        cmd_vel_value.angular.z = angular_speed

        self._cmd_vel_pub.publish(cmd_vel_value)
        self.move_base.make_plan()
        rospy.sleep(self.time_step)


class RealRobotMotionControlContinuousLaser(RealRobotMotionControlContinuous, RealRobotLaser):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)


class RealRobotMotionControlContinuousCostmap(RealRobotMotionControlContinuous, RealRobotCostmap):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)


class RealRobotDWAParamContinuous(RealRobotEnv):
    def __init__(
        self, 
        param_init=[0.5, 1.57, 6, 20, 0.75, 1, 0.3],
        param_list=['TrajectoryPlannerROS/max_vel_x', 
                    'TrajectoryPlannerROS/max_vel_theta', 
                    'TrajectoryPlannerROS/vx_samples', 
                    'TrajectoryPlannerROS/vtheta_samples', 
                    'TrajectoryPlannerROS/path_distance_bias', 
                    'TrajectoryPlannerROS/goal_distance_bias', 
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


class RealRobotDWAParamContinuousLaser(RealRobotDWAParamContinuous, RealRobotLaser):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)


class RealRobotDWAParamContinuousCostmap(RealRobotDWAParamContinuous, RealRobotCostmap):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)