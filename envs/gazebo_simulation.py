import numpy as np

try:  # make sure to create a fake environment without ros installed
    import rospy
    from std_srvs.srv import Empty
    from gazebo_msgs.msg import ModelState
    from gazebo_msgs.srv import SetModelState, GetModelState
    from geometry_msgs.msg import Quaternion, Twist
    from sensor_msgs.msg import LaserScan
    from std_msgs.msg import Bool
except ModuleNotFoundError:
    pass

def create_model_state(x, y, z, angle):
    # the rotation of the angle is in (0, 0, 1) direction
    model_state = ModelState()
    model_state.model_name = 'jackal'
    model_state.pose.position.x = x
    model_state.pose.position.y = y
    model_state.pose.position.z = z
    model_state.pose.orientation = Quaternion(0, 0, np.sin(angle/2.), np.cos(angle/2.))
    model_state.reference_frame = "world"

    return model_state


class GazeboSimulation():

    def __init__(self, init_position = [0, 0, 0]):
        self._pause = rospy.ServiceProxy('/gazebo/pause_physics', Empty)
        self._unpause = rospy.ServiceProxy('/gazebo/unpause_physics', Empty)
        self._reset = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)
        self._model_state_getter = rospy.ServiceProxy('/gazebo/get_model_state', GetModelState)

        self._init_model_state = create_model_state(init_position[0],init_position[1],0,init_position[2])
        
        self.collision_count = 0
        self._collision_sub = rospy.Subscriber("/collision", Bool, self.collision_monitor)
        
        self.bad_vel_count = 0
        self.vel_count = 0
        self._vel_sub = rospy.Subscriber("/jackal_velocity_controller/cmd_vel", Twist, self.vel_monitor)
        
    def vel_monitor(self, msg):
        """
        Count the number of velocity command and velocity command
        that is smaller than 0.2 m/s (hard coded here, count as self.bad_vel)
        """
        vx = msg.linear.x
        if vx <= 0:
            self.bad_vel_count += 1
        self.vel_count += 1
        
    def get_bad_vel_num(self):
        """
        return the number of bad velocity and reset the count
        """
        bad_vel = self.bad_vel_count
        vel = self.vel_count
        self.bad_vel_count = 0
        self.vel_count = 0
        return bad_vel, vel
        
    def collision_monitor(self, msg):
        if msg.data:
            self.collision_count += 1
    
    def get_hard_collision(self):
        # hard collision count since last call
        collided = self.collision_count > 0
        self.collision_count = 0
        return collided

    def pause(self):
        rospy.wait_for_service('/gazebo/pause_physics')
        try:
            self._pause()
        except rospy.ServiceException:
            print ("/gazebo/pause_physics service call failed")

    def unpause(self):
        rospy.wait_for_service('/gazebo/unpause_physics')
        try:
            self._unpause()
        except rospy.ServiceException:
            print ("/gazebo/unpause_physics service call failed")

    def reset(self):
        """
        /gazebo/reset_world or /gazebo/reset_simulation will
        destroy the world setting, here we used set model state
        to put the model back to the origin
        """
        rospy.wait_for_service("/gazebo/set_model_state")
        try:
            self._reset(self._init_model_state)
        except (rospy.ServiceException):
            rospy.logwarn("/gazebo/set_model_state service call failed")

    def get_laser_scan(self):
        data = None
        while data is None:
            try:
                data = rospy.wait_for_message('front/scan', LaserScan, timeout=5)
            except:
                pass
        return data

    def get_model_state(self):
        rospy.wait_for_service("/gazebo/get_model_state")
        try:
            return self._model_state_getter('jackal', 'world')
        except (rospy.ServiceException):
            rospy.logwarn("/gazebo/get_model_state service call failed")

    def reset_init_model_state(self, init_position = [0, 0, 0]):
        """Overwrite the initial model state

        Args:
            init_position (list, optional): initial model state in x, y, z. Defaults to [0, 0, 0].
        """
        self._init_model_state = create_model_state(init_position[0],init_position[1],0,init_position[2])