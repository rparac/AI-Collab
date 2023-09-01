"""
Simplifying wrappers to make the AI collab environment results easier to use.
"""
from dataclasses import dataclass
import random
from typing import Tuple, SupportsFloat, Any, Dict, List, Optional

import gymnasium as gym
import numpy as np
from gymnasium.core import ObsType, WrapperActType, WrapperObsType, ActType

from gym_collab.envs import AICollabEnv
from gym_collab.envs.utils import adjacent_cells_iterator, wrap_action_enum, _find_curr_agent_location
from .action import Action


class AtomicWrapper(gym.Wrapper):
    """
    Automatically waits until an action is executed.
    It does not allow sensing in parallel with move actions.
    """

    def __init__(self, env: AICollabEnv):
        super().__init__(env)
        self.env = env

    def step(self, action: WrapperActType) -> Tuple[WrapperObsType, SupportsFloat, bool, bool, Dict[str, Any]]:
        next_observation, reward_sum, terminated, truncated, info = self.env.step(action)

        while not any(next_observation["action_status"][:4]) and not terminated and not truncated:
            next_observation, reward, terminated, truncated, info = self.env.step(wrap_action_enum(Action.wait))
            reward_sum += reward
        return next_observation, reward_sum, terminated, truncated, info


class AutomaticSensingWrapper(gym.Wrapper):
    """
    Executes sensing actions after every step.
    In particular, it executes the following:
     - get_occupancy_map - so agent always has access to the environment
     - danger_sensing & check_item [for all adjacent object]
        - this results in nearby_obj_weight & nearby_obj_danger predicates which
          contain if the object nearby is dangerous and what is its weight (possibly
          0)
        - if there are multiple nearby objects, it randomly samples one to present to
          the agent
    """

    def __init__(self, env: AtomicWrapper):
        super().__init__(env)
        self.env = env

    def reset(
            self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None
    ) -> Tuple[WrapperObsType, Dict[str, Any]]:
        self.env.reset(seed=seed, options=options)

        occupancy_map_obs, reward, terminated, truncated, obs_info = self._execute_get_occupancy_map()

        if not terminated and not truncated:
            danger_sensing_obs, reward_2, terminated, truncated, info = self._execute_danger_sensing(
                occupancy_map_obs["frame"],
                obs_info["map_metadata"]
            )
            occupancy_map_obs["nearby_obj_weight"] = danger_sensing_obs["nearby_obj_weight"]
            occupancy_map_obs["nearby_obj_danger"] = danger_sensing_obs["nearby_obj_danger"]
            reward += reward_2
        return occupancy_map_obs, obs_info

    def step(
            self, action: WrapperActType
    ) -> Tuple[WrapperObsType, SupportsFloat, bool, bool, Dict[str, Any]]:

        next_observation, reward, terminated, truncated, info = self.env.step(action)

        if not terminated and not truncated:
            occupancy_map_obs, reward_2, terminated, truncated, obs_info = self._execute_get_occupancy_map()
            next_observation["frame"] = occupancy_map_obs["frame"]
            info["map_metadata"] = obs_info["map_metadata"]
            next_observation["action_status"] = np.logical_or(next_observation["action_status"],
                                                              occupancy_map_obs["action_status"])
            reward += reward_2

        if not terminated and not truncated:
            danger_sensing_obs, reward_2, terminated, truncated, danger_info = self._execute_danger_sensing(
                next_observation["frame"],
                info["map_metadata"]
            )
            next_observation["nearby_obj_weight"] = danger_sensing_obs["nearby_obj_weight"]
            next_observation["nearby_obj_danger"] = danger_sensing_obs["nearby_obj_danger"]
            reward += reward_2

        return next_observation, reward, terminated, truncated, info

    def _execute_get_occupancy_map(self):
        occupancy_map_obs, reward_2, terminated, truncated, info = self.env.step(
            wrap_action_enum(Action.get_occupancy_map))
        return occupancy_map_obs, reward_2, terminated, truncated, info

    def _execute_danger_sensing(self, occupancy_map, map_metadata):
        # Assume danger sensing always succeeds
        obs, _reward, terminated, truncated, info = self.env.step(wrap_action_enum(Action.danger_sensing))

        pos = _find_curr_agent_location(occupancy_map)
        grid_size = (len(occupancy_map), len(occupancy_map[0]))
        objects_nearby = list()
        for next_pos in adjacent_cells_iterator(pos, grid_size):
            idx = self._find_object_index(map_metadata, info['object_key_to_index'], next_pos)
            if idx != -1:
                check_item_code = 20
                action = {"action": check_item_code, "item": idx, "message": "empty", "num_cells_move": 1, "robot": 0}
                obs, _reward, terminated, truncated, info = self.env.step(action)
                objects_nearby.append(obs["item_output"])

        obs["nearby_obj_weight"] = 0
        obs["nearby_obj_danger"] = 0

        # Randomly choose which object to show the value for if there are more
        if len(objects_nearby) > 0:
            elem = random.choice(objects_nearby)
            # We ignore the probability for now.
            # Map 0 - unknown, 1 - not dangerous, 2 - dangerous -> 0 - not dangerous, 1 - dangerous
            obs["nearby_obj_danger"] = elem["item_danger_level"] - 1
            obs["nearby_obj_weight"] = elem["item_weight"]

        return obs, _reward, terminated, truncated, info

    # Gets valid index for check_item action or -1 if none exist
    @staticmethod
    def _find_object_index(map_metadata: Dict[str, List[List[Any]]], obj_key_to_idx: Dict[str, int],
                           pos: Tuple[int, int]) -> int:
        pos_i, pos_j = pos
        map_key = f"{pos_i}_{pos_j}"
        if map_key not in map_metadata:
            return -1

        object_key = map_metadata[map_key][0][0]
        return obj_key_to_idx[object_key]


class SimpleActions(gym.Wrapper):
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

    def __init__(self, env: AtomicWrapper):
        super().__init__(env)
        self.action_space = gym.spaces.Discrete(8)
        self.pick_up_code = 5
        self.ask_for_help_code = 7
        # Needs to keep track of observation to determine how to grab an object
        self.last_observation = None

    def reset(self, **kwargs) -> Tuple[WrapperObsType, Dict[str, Any]]:
        initial_observation, info = self.env.reset(**kwargs)
        self.last_observation = initial_observation
        return initial_observation, info

    def step(
            self, action: WrapperActType
    ) -> Tuple[WrapperObsType, SupportsFloat, bool, bool, Dict[str, Any]]:
        if not self.action_space.contains(action):
            raise ValueError(f"Invalid action: {action}. Action must be in the range [0, 7].")

        if action == self.pick_up_code:
            a = self._find_grab_action_code()
        elif action == self.ask_for_help_code:
            a = self._get_ask_for_help_action()
        else:
            a = self._map_action_code(action)
        next_observation, reward, terminated, truncated, info = self.env.step(a)
        self.last_observation = next_observation
        return next_observation, reward, terminated, truncated, info

    @staticmethod
    def _get_ask_for_help_action() -> ActType:
        send_message_action_code = 23
        broadcast_code = 0
        return {
            "action": send_message_action_code,
            "item": 0,
            "message": "I need help",  # TODO: add more information about the object
            "num_cells_move": 0,
            "robot": broadcast_code,
        }

    # Determines in which direction the agent should grab an item
    # Returns the action code for doing so
    def _find_grab_action_code(self) -> ActType:
        dirs_to_pick_direction = {
            (-1, 0): Action.grab_up,
            (0, 1): Action.grab_right,
            (1, 0): Action.grab_down,
            (0, -1): Action.grab_left,
        }

        frame = self.last_observation.frame
        pos = _find_curr_agent_location(frame)
        grid_size = (len(frame), len(frame[0]))
        neighbour_pos = list(adjacent_cells_iterator(pos, grid_size))
        if len(neighbour_pos) > 0:
            # Always check directions in a different order.
            # This will result in a random object selection if there is more than 1
            # in the vicinity.
            elem = random.choice(neighbour_pos)
            return wrap_action_enum(dirs_to_pick_direction[elem])

        # Attempt to grab up if everything fails.
        # This will result in the same errors as an incorrect grab
        up_grab_code = next(iter(dirs_to_pick_direction.values()))
        return wrap_action_enum(up_grab_code)

    @staticmethod
    def _map_action_code(action: WrapperActType) -> ActType:
        new_to_old_map = {
            0: Action.wait,  # do nothing
            1: Action.move_up,  # up
            2: Action.move_right,  # right
            3: Action.move_down,  # down
            4: Action.move_left,  # left
            6: Action.drop_object,  # drop
        }
        return wrap_action_enum(new_to_old_map[action])


@dataclass
class AgentInfo:
    pos: (int, int)
    need_help: bool


@dataclass
class ObjectInfo:
    pos: (int, int)
    carried_by: np.ndarray


class SimpleObservations(gym.Wrapper):
    def __init__(self, env: AutomaticSensingWrapper):
        super().__init__(env)
        self.env = env

    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None) -> Tuple[
        WrapperObsType, Dict[str, Any]]:
        obs, info = self.env.reset(seed=seed, options=options)
        return self._map_observation(obs, info), info

    def step(self, action: WrapperActType) -> Tuple[WrapperObsType, SupportsFloat, bool, bool, Dict[str, Any]]:
        obs, reward, terminated, truncated, info = self.env.step(action)
        return self._map_observation(obs, info), reward, terminated, truncated, info

    def _map_observation(self, observation: ObsType, info: Dict[str, Any]) -> WrapperObsType:
        curr_pos, other_pos = self._find_agent_positions(observation["frame"])

        agent_id = self._position_to_agent_id(curr_pos, info["map_metadata"])
        agent_infos = self._get_agent_infos(other_pos, info)
        num_agents = len(agent_infos) + 1

        obs = {
            "agent_id": agent_id,
            "agent_strength": observation["strength"],
            "nearby_obj_weight": observation["nearby_obj_weight"],
            "nearby_obj_danger": observation["nearby_obj_danger"],
            "agent_infos": agent_infos,
            "object_infos": self.get_object_infos(info["map_metadata"], num_agents),
        }
        return obs

    def _get_agent_infos(self, other_pos: List[Tuple[int, int]], info: Dict[str, Any]):
        id_pos_map = {
            self._position_to_agent_id(pos, info["map_metadata"]): pos
            for pos in other_pos
        }
        # TODO: implement when we know how to get robot help request,
        agent_infos = [AgentInfo(id_pos_map[i], need_help=False) for i in range(1, len(id_pos_map) + 1)]
        return agent_infos

    def _find_agent_positions(self, world_map: np.ndarray) -> (Tuple[int, int], List[Tuple[int, int]]):
        other_agent_id = 3
        curr_agent_id = 5

        positions = np.where(world_map == other_agent_id)
        # Transpose a pair of lists into list of pairs
        other_agent_pos = list(zip(*positions))

        positions = np.where(world_map == curr_agent_id)
        curr_agent_pos = next(zip(*positions))

        return curr_agent_pos, other_agent_pos

    def _position_to_agent_id(self, position: Tuple[int, int], map_metadata: Dict[str, List[Any]]) -> int:
        pos_i, pos_j = position
        pos_as_str = f"{pos_i}_{pos_j}"

        agent_env_id = map_metadata[pos_as_str][-1]

        return self.name_to_idx(agent_env_id)

    def name_to_idx(self, name: str) -> int:
        # ids are assigned from 'A'
        return ord(name) - ord('A')

    def get_object_infos(self, map_metadata: Dict[str, List[Any]], num_agents: int) -> List[ObjectInfo]:
        sol = []
        for str_pos, location_info in map_metadata.items():
            # Check if current positions holds object information
            if type(location_info[0]) == list:
                pos = self._map_key_to_pos(str_pos)
                carried_by = np.zeros(num_agents)
                # An agent is holding an object if both object and agent are at the same location
                if len(location_info) > 1:
                    agent_name = location_info[1]
                    carried_by[self.name_to_idx(agent_name)] = 1
                sol.append(ObjectInfo(pos=pos, carried_by=carried_by))
        return sol

    def _map_key_to_pos(self, key: str) -> (int, int):
        vals = key.split('_')
        assert len(vals) == 2

        return int(vals[0]), int(vals[1])
