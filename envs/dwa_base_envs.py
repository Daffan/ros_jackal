import gym
import rospy
import rospkg
import roslaunch
import time
import numpy as np
import os
from os.path import dirname, join, abspath
import subprocess
from gym.spaces import Box, Discrete

from envs.gazebo_simulation import GazeboSimulation
from envs.move_base import MoveBase


class DWABase(gym.Env):
    def __init__(
        self,
        world_name="jackal_world.world",
        gui=False,
        init_position=[0, 0, 0],
        goal_position=[4, 0, 0],
        max_step=100,
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
        self.verbose = verbose
        self.time_step = time_step
        self.max_step = max_step

        # launch gazebo and dwa demo
        rospy.logwarn(">>>>>>>>>>>>>>>>>> Load world: %s <<<<<<<<<<<<<<<<<<" %(world_name))
        rospack = rospkg.RosPack()
        self.BASE_PATH = rospack.get_path('jackal_helper')
        world_name = join(self.BASE_PATH, "worlds", world_name)
        launch_file = join(self.BASE_PATH, 'launch', 'ros_jackal_launch.launch')

        self.gazebo_process = subprocess.Popen(['roslaunch', 
                                                launch_file,
                                                'world_name:=' + world_name,
                                                'gui:=' + ("true" if gui else "false"),
                                                'verbose:=' + ("true" if verbose else "false")
                                                ])
        time.sleep(10)  # sleep to wait until the gazebo being created

        # initialize the node for gym env
        rospy.init_node('gym', anonymous=True, log_level=rospy.FATAL)
        rospy.set_param('/use_sim_time', True)

        self._set_start_goal_BARN(self.world_name)  # overwrite the starting goal if use BARN dataset
        self.gazebo_sim = GazeboSimulation(init_position = self.init_position)
        self.move_base = MoveBase(goal_position = self.goal_position)

        # Not implemented
        self.action_space = None
        self.observation_space = None
        self.reward_range = None

        self.step_count = 0

    def seed(self, seed):
        # TODO: make sure numpy is the only module that need to be seeded
        np.random.seed(seed)

    def reset(self):
        """reset the environment
        """
        self.step_count=0
        # Reset robot in odom frame clear_costmap
        self.move_base.reset_robot_in_odom()
        # Resets the state of the environment and returns an initial observation
        self.gazebo_sim.reset()
        self.move_base.set_global_goal()
        self._clear_costmap()
        return self._get_observation()

    def _clear_costmap(self):
        self.move_base.clear_costmap()
        time.sleep(0.1)
        self.move_base.clear_costmap()
        time.sleep(0.1)
        self.move_base.clear_costmap()

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
        return dict(world=self.world_name)

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

    def _set_start_goal_BARN(self, world_name):
        """Use predefined start and goal position for BARN dataset
        """
        if world_name.startswith("BARN"):
            path_dir = join(self.BASE_PATH, "worlds", "BARN", "path_files")
            world_id = int(world_name.split('_')[-1].split('.')[0])
            path = np.load(join(path_dir, 'path_%d.npy' % world_id))
            init_x, init_y = self._path_coord_to_gazebo_coord(*path[0])
            goal_x, goal_y = self._path_coord_to_gazebo_coord(*path[-1])
            init_y -= 1
            goal_x -= init_x
            goal_y -= (init_y-5) # put the goal 5 meters backward
            self.init_position = [init_x, init_y, np.pi/2]
            self.goal_position = [goal_x, goal_y, 0]

    def _path_coord_to_gazebo_coord(self, x, y):
        RADIUS = 0.075
        r_shift = -RADIUS - (30 * RADIUS * 2)
        c_shift = RADIUS + 5

        gazebo_x = x * (RADIUS * 2) + r_shift
        gazebo_y = y * (RADIUS * 2) + c_shift

        return (gazebo_x, gazebo_y)


class DWABaseLaser(DWABase):
    def __init__(self, laser_clip=4, **kwargs):
        super().__init__(**kwargs)
        self.laser_clip = laser_clip
        
        # 720 laser scan + local goal (in angle)
        self.observation_space = Box(
            low=0,
            high=laser_clip,
            shape=(721,),
            dtype=np.float32
        )

    def _get_laser_scan(self):
        """Get 720 dim laser scan
        Returns:
            np.ndarray: (720,) array of laser scan 
        """
        laser_scan = self.gazebo_sim.get_laser_scan()
        laser_scan = np.array(laser_scan.ranges)
        laser_scan[laser_scan > self.laser_clip] = self.laser_clip
        return laser_scan

    def _get_observation(self):
        # observation is the 720 dim laser scan + one local goal in angle
        laser_scan = self._get_laser_scan()
        local_goal = self._get_local_goal()
        
        laser_scan = (laser_scan - self.laser_clip/2.) / self.laser_clip # scale to (-0.5, 0.5)
        local_goal = local_goal / (2.0 * np.pi) # scale to (-0.5, 0.5)

        obs = np.concatenate([laser_scan, local_goal])

        return obs


class DWABaseCostmap(DWABase):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        # 720 laser scan + local goal (in angle)
        self.observation_space = Box(
            low=-1,
            high=10,
            shape=(1, 84, 84),
            dtype=np.float32
        )

    def _get_costmap(self):
        costmap = self.move_base.get_costmap().data
        costmap = np.array(costmap, dtype="uint8").reshape(1, 800, 800)
        
        x, y = self.move_base.robot_config.X, self.move_base.robot_config.Y
        X, Y = int(x*20) + 400, int(y*20) + 400
        costmap = costmap[:, Y - 42:Y + 42, X - 42:X + 42]
        costmap[np.where(costmap < 100)] = 0
        costmap[np.where(costmap != 0)] = 1
        
        return costmap

    def _get_observation(self):
        # observation is the 720 dim laser scan + one local goal in angle
        costmap = self._get_costmap()
        # for now we skip the local goal temperally
        # local_goal = local_goal / (2.0 * np.pi) # scale to (-0.5, 0.5)
        obs = costmap
        return obs

    def visual_costmap(self, costmap):
        from matplotlib import pyplot as plt
        import cv2

        costmap = costmap * 100
        costmap = np.transpose(costmap, axes=(1, 2, 0)) + 100
        costmap = np.repeat(costmap, 3, axis=2)
        plt.imshow(costmap)
        plt.show(block=False)
        plt.pause(.5)