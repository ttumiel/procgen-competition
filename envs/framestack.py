import gym
from gym.wrappers import FrameStack
from ray.tune import registry
import numpy as np

from envs.procgen_env_wrapper import ProcgenEnvWrapper

class FrameGapStack(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        o = self.observation_space
        self.f,*w = o.shape
        self.f -= 1
        shape = 2,*w
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=shape, dtype=o.dtype)

    def observation(self, obs):
        return np.array(obs)[(0,self.f),]

# Register Env in Ray
registry.register_env(
    "stacked_procgen_env",  # This should be different from procgen_env_wrapper
    lambda config: FrameStack(ProcgenEnvWrapper(config), 4),
)

registry.register_env(
    "gap_stacked_procgen_env",  # This should be different from procgen_env_wrapper
    lambda config: FrameGapStack(FrameStack(ProcgenEnvWrapper(config), 4)),
)
