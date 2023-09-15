import abc
import copy
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING

import gymnasium
from gymnasium.wrappers import RecordEpisodeStatistics, TimeLimit

from rm_marl.reward_machine import RewardMachine

if TYPE_CHECKING:
    from ..reward_machine import RewardMachine


class NumberStepsDiscountedRewardWrapper(gymnasium.Wrapper):
    def step(self, action):
        observation, reward, terminated, truncated, info = super().step(action)

        if terminated and info["episode"]["r"]:
            info["episode"]["r"] /= info["episode"]["l"]
        elif terminated:
            raise ValueError(info)

        return observation, reward, terminated, truncated, info


class LabelingFunctionWrapper(gymnasium.Wrapper):
    def __init__(self, env: gymnasium.Env):
        super().__init__(env)

        self.prev_obs = None

    @abc.abstractmethod
    def get_labels(self, obs: dict = None, prev_obs: dict = None):
        raise NotImplementedError("get_labels")

    def step(self, action):
        observation, reward, terminated, truncated, info = super().step(action)
        info["labels"] = self.get_labels(obs=observation, prev_obs=self.prev_obs)
        self.prev_obs = copy.deepcopy(observation)
        return observation, reward, terminated, truncated, info

    def reset(self, **kwargs):
        """Resets the environment with kwargs."""
        obs, info = super().reset(**kwargs)
        info["labels"] = self.get_labels(obs=obs, prev_obs=None)
        self.prev_obs = obs
        return obs, info


@dataclass
class RandomLabelingConfig:
    proba: float
    condition: callable
    env_update: callable


class RandomLabelingFunctionWrapper(gymnasium.Wrapper):


    def __init__(self, env: gymnasium.Env, random_events: dict):
        """
        random_event: {event: (probability, function(env))}
        """
        self.random_events = random_events or {}
        self.trace = []
        super().__init__(env)

    @property
    def flatten_trace(self):
        return tuple(e for es in self.trace for e in es)

    def get_labels(self, _obs: dict = None, _prev_obs: dict = None):
        # TODO verify this logic
        valid_events = [
            e 
            for e, c in self.random_events.items() 
            if c.condition(self) and self.np_random.random() <= c.proba
        ]
        return valid_events
        # if not valid_events:
        #     return []

        # event = self.np_random.choice(valid_events)
        # config = self.random_events[event]
        # if self.np_random.random() <= config.proba:
        #     return [event]

        # return []

    def step(self, action):
        observation, reward, terminated, truncated, info = super().step(action)
        labels, simulated_env_updates = info.get("labels", []), {}
        # FIXME
        # # two events can not happen at the same time
        # if not labels:
        random_labels = self.get_labels()
        for l in random_labels:
            simulated_env_updates[l] = self.random_events[l].env_update
            labels.append(l)
        
        self.trace.append(labels or [])
        info["labels"] = labels
        info["env_simulated_updates"] = simulated_env_updates
        return observation, reward, terminated, truncated, info

    def reset(self, **kwargs):
        self.trace.clear()
        return super().reset(**kwargs)

    @staticmethod
    def condition_true(e):
        return True


class AutomataWrapper(gymnasium.Wrapper):
    class LabelMode(Enum):
        ALL = 0
        RM = 1
        STATE = 2

    class TerminationMode(Enum):
        RM = 0
        ENV = 1
    
    def __init__(
        self, 
            env: gymnasium.Env, 
            rm: "RewardMachine", 
            label_mode: LabelMode = LabelMode.ALL,
            termination_mode: TerminationMode = TerminationMode.RM
    ):
        """
        label_mode:
            - ALL: returns all the labels returned by the labeling function
            - RM: returns only the labels present in the RM
            - STATE: returns only the lables that can be observed from the current state
        """
        assert hasattr(env, "get_labels"), "The LabelingFunctionWrapper is required"

        self.label_mode = label_mode
        self.termination_mode = termination_mode

        self.rm = rm
        self.u = self.rm.u0
        super().__init__(env)

    def filter_labels(self, labels, u):
        return [e for e in labels if self._is_valid_event(e, u)]

    def _is_valid_event(self, event, u):
        if self.label_mode == self.LabelMode.ALL:
            return True
        elif self.label_mode == self.LabelMode.RM:
            return event in self.rm.get_valid_events()
        elif self.label_mode == self.LabelMode.STATE:
            return event in self.rm.get_valid_events(u)

    def step(self, action):
        observation, reward, terminated, truncated, info = super().step(action)
        info["labels"] = self.filter_labels(info["labels"], self.u)
        simulated_updates = info.pop("env_simulated_updates", {})

        labels = copy.deepcopy(info["labels"])
        for e in info["labels"]:
            # apply simulated updates to the environment
            if e in simulated_updates:
                labels_update = simulated_updates[e](self.unwrapped)
                if labels_update:
                    labels = labels_update(labels)
        
        info["labels"] = labels
        u_next = self.rm.get_next_state(self.u, info["labels"])
        reward = self._get_reward(reward, u_next)
        self.u = u_next

        terminated = self._get_terminated(terminated)
        info["rm_state"] = self.u

        return observation, reward, terminated, truncated, info

    def _get_reward(self, reward, u_next):
        return reward
    
    def _get_terminated(self, terminated):
        if self.termination_mode == self.TerminationMode.ENV:
            return terminated
        else: # should be TerminationMode.RM
            return self.rm.is_state_terminal(self.u)

    def reset(self, **kwargs):
        obs, info = super().reset(**kwargs)
        self.u = self.rm.u0

        info["labels"] = self.filter_labels(info["labels"], self.u)
        simulated_updates = info.pop("env_simulated_updates", {})

        for e in info["labels"]:
            # apply simulated updates to the environment
            if e in simulated_updates:
                simulated_updates[e](self.unwrapped)
        
        u_next = self.rm.get_next_state(self.u, info["labels"])
        self.u = u_next
        
        info["rm_state"] = self.u
        return obs, info

class RMProgressReward():
    def __init__(self) -> None:
        self.max_u = 0

    def __call__(self, rm, u, u_next, reward):
        if int(u_next) > self.max_u:
            self.max_u = int(u_next)
            reward += 1
        elif int(u_next) < int(u):
            reward -= 1
        return reward

class RewardMachineWrapper(AutomataWrapper):

    def __init__(
            self, 
            env: gymnasium.Env, 
            rm: RewardMachine, 
            label_mode: AutomataWrapper.LabelMode = AutomataWrapper.LabelMode.ALL, 
            termination_mode: AutomataWrapper.TerminationMode = AutomataWrapper.TerminationMode.RM,
            reward_function: callable = None
    ):
        super().__init__(env, rm, label_mode, termination_mode)

        self.reward_function = reward_function or self._simple_reward_func

    @staticmethod
    def _simple_reward_func(rm, u, u_next, reward):
        return rm.get_reward(u, u_next)

    def _get_reward(self, reward, u_next):
        return self.reward_function(self.rm, self.u, u_next, reward)


class SingleAgentEnvWrapper(gymnasium.Wrapper):
    def __init__(self, env: gymnasium.Env, agent_id: str):
        self.agent_id = agent_id

        super().__init__(env)

        self.observation_space = gymnasium.spaces.Dict({
            aid: obs_space
            for aid, obs_space in self.observation_space.spaces.items()
            if aid == self.agent_id
        })

    def reset(self, **kwargs):
        obs, info = super().reset(**kwargs)
        obs = {aid: apos for aid, apos in obs.items() if aid == self.agent_id}
        return obs, info

    def step(self, action):
        action = {aid: a for aid, a in action.items() if aid == self.agent_id}

        observation, reward, terminated, truncated, info = super().step(action)

        observation = {
            aid: apos for aid, apos in observation.items() if aid == self.agent_id
        }

        return observation, reward, terminated, truncated, info

