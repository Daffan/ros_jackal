import gym
import rospy
import rospkg
import roslaunch
import time
import numpy as np
import os
from os.path import dirname, join, abspath
import subprocess
from gym import spaces

from envs.gazebo_simulation import GazeboSimulation
from envs.move_base import  MoveBase


class DWABase(gym.Env):
    
    def __init__(
        self,
        world_name="jackal_world.world",
        gui=False,
        init_position=[0, 0, 0],
        goal_position=[4, 0, 0],
        laser_clip=4,
        max_step=400,
        time_step=1,
        verbose=True
    ):
        """Base RL env that initialize jackal simulation in Gazebo
        """
        super().__init__()

        self.world_name = world_name
        self.gui = gui
        self.init_position = init_position
        self.goal_position = goal_position
        self.laser_clip = laser_clip
        self.verbose = verbose
        self.time_step = time_step
        self.max_step = max_step

        # launch gazebo and dwa demo
        rospy.logwarn(">>>>>>>>>>>>>>>>>> Load world: %s <<<<<<<<<<<<<<<<<<" %(world_name))
        # TODO: not sure whether we still need to use jackal_helper or not
        rospack = rospkg.RosPack()
        BASE_PATH = rospack.get_path('jackal_helper')
        world_name = join(BASE_PATH, "worlds", world_name)
        launch_file = join(BASE_PATH, 'launch', 'ros_jackal_launch.launch')

        # TODO: remove the VLP16 and camera condition, just use the laser scan by default
        self.gazebo_process = subprocess.Popen(['roslaunch', 
                                                launch_file,
                                                'world_name:=' + world_name,
                                                'gui:=' + ("true" if gui else "false"),
                                                'verbose:=' + ("true" if verbose else "false")
                                                ])
        time.sleep(10) # sleep to wait until the gazebo being created

        # initialize the node for gym env
        rospy.init_node('gym', anonymous=True, log_level=rospy.FATAL)
        rospy.set_param('/use_sim_time', True)

        self.gazebo_sim = GazeboSimulation(init_position = self.init_position)
        self.move_base = MoveBase(goal_position = self.goal_position)

        # Not implemented
        self.action_space = None
        self.observation_space = None
        self.reward_range = None

        self.move_base.set_global_goal()
        self.reset()
        self.step_count = 0

    def seed(self, seed):
        # TODO: make sure numpy is the only module that need to be seeded
        np.random.seed(seed)

    def reset(self):
        """reset the environment
        """
        return self._get_observation()

    def step(self, action):
        """take an action and step the environment
        """
        self._take_action(action)
        self.step_count += 1
        obs = self._get_observation()
        rew = self._get_reward()
        done = self._get_done()
        info = self._get_info()
        return obs, rew, done, info

    def _take_action(self, action):
        raise NotImplementedError()

    def _get_observation(self):
        raise NotImplementedError()

    def _get_reward(self):
        raise NotImplementedError()

    def _get_done(self):
        raise NotImplementedError()

    def _get_info(self):
        raise NotImplementedError()

    def _get_laser_scan(self):
        """Get 720 dim laser scan
        Returns:
            np.ndarray: (720,) array of laser scan 
        """
        laser_scan = self.gazebo_sim.get_laser_scan()
        laser_scan = np.array(laser_scan.ranges)
        laser_scan[laser_scan > self.laser_clip] = self.laser_clip
        return laser_scan

    def _get_local_goal(self):
        """get local goal in angle
        Returns:
            float: local goal in angle
        """
        local_goal = self.move_base.get_local_goal()
        local_goal = np.array([np.arctan2(local_goal.position.y, local_goal.position.x)])
        return local_goal

    def close(self):
        # These will make sure all the ros processes being killed
        os.system("killall -9 rosmaster")
        os.system("killall -9 gzclient")
        os.system("killall -9 gzserver")
        os.system("killall -9 roscore")