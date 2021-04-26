from tianshou.env import DummyVectorEnv

class DummyVectorEnvSpace(DummyVectorEnv):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.obs_space, *_ = self.observation_space
        self.act_space, *_ = self.action_space