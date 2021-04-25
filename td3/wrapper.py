from tianshou.env import DummyVectorEnv

class DummyVectorEnvSpace(DummyVectorEnv):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.observation_space = self.observation_space[0]
        self.action_space = self.action_space[0]