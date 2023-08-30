"""
Simplifying wrappers to make the AI collab environment results easier to use.
"""
from dataclasses import dataclass
import random
from typing import Tuple, SupportsFloat, Any, Dict, Optional

import gymnasium as gym
import numpy as np
from gymnasium.core import ObsType, WrapperActType, WrapperObsType, ActType

from gym_collab.envs import AICollabEnv

empty_action = {
    "action": -1,
    "item": 0,
    "message": "empty",
    "num_cells_move": 1,
    "robot": 0,
}


# TODO: implement reset to get the occupancy map
class GridWorldEnv(gym.Wrapper):
    def __init__(self, env: AICollabEnv):
        super().__init__(env)
        self.env = env

    def step(
            self, action: WrapperActType
    ) -> Tuple[WrapperObsType, SupportsFloat, bool, bool, Dict[str, Any]]:

        next_observation, reward, terminated, truncated, info = self._execute_and_wait(action)

        if not terminated and not truncated:
            get_occupancy_map_code = 18
            action = empty_action.copy()
            action["action"] = get_occupancy_map_code
            occupancy_map_obs, reward_2, terminated, truncated, info = self._execute_and_wait(action)
            next_observation["frame"] = occupancy_map_obs["frame"]
            next_observation["action_status"] = np.logical_or(next_observation["action_status"],
                                                              occupancy_map_obs["action_status"])
            reward += reward_2

        return next_observation, reward, terminated, truncated, info

    def _execute_and_wait(self, action):
        next_observation, reward_sum, terminated, truncated, info = self.env.step(action)
        while not any(next_observation["action_status"][:4]) and not terminated and not truncated:
            wait_action_code = 26
            action1 = empty_action.copy()
            action1["action"] = wait_action_code
            next_observation, reward, terminated, truncated, info = self.env.step(action1)
            reward_sum += reward
        return next_observation, reward_sum, terminated, truncated, info


class ObservationTrackingEnv(gym.Wrapper):
    """
    Stores the last seen observation as obs field.
    This allows using the last observation in order to execute the next action,
    e.g. to choose which direction to pick up the object in.
    """

    def __init__(self, env: gym.Env):
        super().__init__(env)
        self.obs = None

    def reset(self, **kwargs) -> Tuple[WrapperObsType, Dict[str, Any]]:
        initial_observation, info = self.env.reset(**kwargs)
        self.obs = initial_observation
        return initial_observation, info

    def step(
            self, action: WrapperActType
    ) -> Tuple[WrapperObsType, SupportsFloat, bool, bool, Dict[str, Any]]:
        next_observation, reward, terminated, truncated, info = self.env.step(action)
        self.obs = next_observation
        return next_observation, reward, terminated, truncated, info


class SimplePickup(gym.ActionWrapper):
    """
    Introduces a new action: auto grab - number 27
    """

    def __init__(self, env: ObservationTrackingEnv):
        super().__init__(env)
        self.env = env
        self.auto_grab_code = 27

    def step(
            self, action: WrapperActType
    ) -> Tuple[WrapperObsType, SupportsFloat, bool, bool, Dict[str, Any]]:
        a = action
        if action == self.auto_grab_code:
            a = self._get_grab_action_code()
        return self.env.step(a)

    # Determines in which direction the agent should grab an item
    # Returns the action code for doing so
    def _get_grab_action_code(self):
        dirs_to_pick_direction = {
            (-1, 0): 8,  # grab up
            (0, 1): 9,  # grab right
            (1, 0): 10,  # grab down
            (0, -1): 11,  # grab left
        }

        pos_i, pos_j = _find_curr_agent_location(self.env.obs.frame)
        dirs = list(dirs_to_pick_direction.keys())
        # Always check directions in a different order.
        # This will result in a random object selection if there is more than 1
        # in the vicinity.
        random.shuffle(dirs)
        for step_i, step_j in dirs:
            new_i = pos_i + step_i
            new_j = pos_j + step_j
            if _inbounds((new_i, new_j), new_i, new_j) and self.env.obs.frame[new_i][new_j] == 2:
                return dirs_to_pick_direction[(step_i, step_j)]

        # Attempt to grab up if everything fails.
        # This will result in the same errors as an incorrect grab
        return dirs_to_pick_direction[(-1, 0)]


class SimplifiedActionSet(gym.ActionWrapper):
    """
    Actions:
        0 - do nothing
        1 - up
        2 - right
        3 - down
        4 - left
        5 - pick up
        6 - drop
        7 - ask for help
    """

    # TODO: Type
    def __init__(self, env):
        super().__init__(env)
        self.action_space = gym.spaces.Discrete(8)

    def action(self, action: WrapperActType) -> ActType:
        if not self.action_space.contains(action):
            raise ValueError(f"Invalid action: {action}. Action must be in the range [0, 7].")

        # Map the action back to the original action space
        mapped_action = self.map_action(action)
        return mapped_action

    def map_action(self, action: WrapperActType) -> ActType:
        new_to_old_map = {
            0: 26,  # do nothing
            1: 0,  # up
            2: 3,  # right
            3: 1,  # down
            4: 2,  # left
            5: 8,  # pick up TODO: implement pickup
            6: 16,  # drop
            7: 23,  # ask for help TODO: implement send message
        }
        return new_to_old_map[action]


def _find_curr_agent_location(occupancy_map: np.ndarray) -> (int, int):
    agent_code = 5
    return tuple(np.transpose(np.where(occupancy_map == agent_code)).squeeze())


def _inbounds(size: (int, int), i: int, j: int) -> bool:
    size_i, size_j = size
    return 0 <= i < size_i and 0 <= j < size_j


class NewObservationAdded(gym.ObservationWrapper):
    def __init__(self, env: GridWorldEnv):
        super().__init__(env)

    def observation(self, observation: ObsType) -> WrapperObsType:
        occupancy_map = observation['frame']
        curr_pos = _find_curr_agent_location(occupancy_map)

        return {**observation, "box_reached": self._box_adjacent(occupancy_map, curr_pos)}

    # Check all adjacent fields to the agent, check if any of them is a box
    @staticmethod
    def _box_adjacent(occupancy_map: np.ndarray, curr_pos: (int, int)):
        step_dir_i = [1, 1, 0, -1, -1, -1, 0, 1]
        step_dir_j = [0, 1, 1, 1, 0, -1, -1, -1]
        curr_i, curr_j = curr_pos

        for dir_i, dir_j in zip(step_dir_i, step_dir_j):
            new_j = curr_j + dir_j
            new_i = curr_i + dir_i
            size = tuple(occupancy_map.shape)
            if _inbounds(size, new_i, new_j) \
                    and occupancy_map[new_i][new_j] == 2:
                return True

        return False


@dataclass
class AgentInfo:
    pos: (int, int)
    need_help: bool


class SimpleObservations(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)

    def observation(self, observation: ObsType) -> WrapperObsType:
        obs = {
            "agent_id": -1,
            "agent_strength": -1,
            "nearby_object_weight": 0,  # TODO: implement when there is danger sensing
            "nearby_object_danger": 0,  # TODO: implement when there is danger sensing: 0 for no danger, 1 for danger
            "agent_info": [ # TODO: implement when we know how to get robot info
            ],
        },
        return obs

#TODO: reward shaping
# It would be sufficient just to put -1 whenever there is