import gymnasium as gym
from gymnasium.core import WrapperActType

from gym_collab.envs import AICollabEnv
from gym_collab.envs.helper_wrappers import AtomicWrapper, AutomaticSensingWrapper, SimpleObservations


class MARLWrapper(gym.Wrapper):
    def __init__(self, env: AICollabEnv):
        super().__init__(env)
        self.env = AtomicWrapper(env)
        self.env = AutomaticSensingWrapper(self.env)
        self.env = SimpleObservations(self.env)

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)

    def step(self, action: WrapperActType):
        return self.env.step(action)
