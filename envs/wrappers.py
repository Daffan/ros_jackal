import gym

class ShapingRewardWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)

    def reset(self):
        obs = self.env.reset()
        self.Y = self.env.gazebo_sim.get_model_state().pose.position.y
        return obs

    def step(self, action):
        obs, rew, done, info = self.env.step(action)
        position = self.env.gazebo_sim.get_model_state().pose.position
        rew += position.y - self.Y
        self.Y = position.y
        return obs, rew, done, info