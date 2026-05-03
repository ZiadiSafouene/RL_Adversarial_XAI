import gymnasium as gym
from gymnasium import spaces

class AdvEnv(gym.Env):
    def __init__(self):
        super(AdvEnv, self).__init__()
        # Define action and observation space
        pass

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        return None, {}

    def step(self, action):
        return None, 0, False, False, {}
