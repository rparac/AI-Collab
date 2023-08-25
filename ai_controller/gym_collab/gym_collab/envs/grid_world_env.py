"""
Simplifying wrappers to make the AI collab environment results easier to use.
"""
from typing import Tuple, SupportsFloat, Any, Dict

import gymnasium as gym
import numpy as np
from gymnasium.core import ObsType, WrapperActType, WrapperObsType

from gym_collab.envs import AICollabEnv

empty_action = {
    "action": -1,
    "item": 0,
    "message": "empty",
    "num_cells_move": 1,
    "robot": 0,
}


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


class SimpleObservations(gym.ObservationWrapper):
    def __init__(self, env: GridWorldEnv):
        super().__init__(env)

    def observation(self, observation: ObsType) -> WrapperObsType:
        occupancy_map = observation['frame']
        curr_pos = self._find_curr_agent_location(occupancy_map)

        return {**observation, "box_reached": self._box_adjacent(occupancy_map, curr_pos)}

    @staticmethod
    def _find_curr_agent_location(occupancy_map: np.ndarray) -> (int, int):
        agent_code = 5
        return tuple(np.transpose(np.where(occupancy_map == agent_code)).squeeze())

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
            if SimpleObservations._inbounds(size, new_i, new_j) \
                    and occupancy_map[new_i][new_j] == 2:
                return True

        return False

    @staticmethod
    def _inbounds(size: (int, int), i: int, j: int) -> bool:
        size_i, size_j = size
        return 0 <= i < size_i and 0 <= j < size_j
