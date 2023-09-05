"""
Simplifying wrappers to make the AI collab environment results easier to use.
"""
from dataclasses import dataclass
import random
from typing import Tuple, SupportsFloat, Any, Dict, List, Optional, Union

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

        while not any(next_observation["action_status"][:4]) and not terminated and not truncated \
                and action["action"] != Action.wait.value:
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
    - get_messages
    """

    def __init__(self, env: AtomicWrapper):
        super().__init__(env)
        self.env = env

    def reset(
            self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None
    ) -> Tuple[WrapperObsType, Dict[str, Any]]:
        self.env.reset(seed=seed, options=options)

        obs, info, _reward = self._update_with_sensing_info()
        return obs, info

    def step(
            self, action: WrapperActType
    ) -> Tuple[WrapperObsType, SupportsFloat, bool, bool, Dict[str, Any]]:

        next_observation, reward, terminated, truncated, info = self.env.step(action)
        next_observation, info, reward2 = self._update_with_sensing_info(terminated, truncated, next_observation, info)

        return next_observation, reward + reward2, terminated, truncated, info

    def _update_with_sensing_info(self, terminated=False, truncated=False, obs_to_update=None, info_to_update=None):
        reward = 0
        if not terminated and not truncated:
            occupancy_map_obs, reward_2, terminated, truncated, obs_info = self._execute_get_occupancy_map()
            if info_to_update is None and obs_to_update is None:
                info_to_update = obs_info
                obs_to_update = occupancy_map_obs
            else:
                info_to_update["frame"] = occupancy_map_obs["frame"]
                info_to_update["map_metadata"] = obs_info["map_metadata"]
                obs_to_update["action_status"] = np.logical_or(obs_to_update["action_status"],
                                                               occupancy_map_obs["action_status"])
            reward += reward_2
        if not terminated and not truncated:
            danger_sensing_obs, reward_2, terminated, truncated, danger_info = self._execute_danger_sensing(
                obs_to_update["frame"],
                info_to_update["map_metadata"]
            )
            obs_to_update["nearby_obj_weight"] = danger_sensing_obs["nearby_obj_weight"]
            obs_to_update["nearby_obj_danger"] = danger_sensing_obs["nearby_obj_danger"]
            reward += reward_2

        if not terminated and not truncated:
            get_messages_obs, reward_2, terminated, truncated, messages_info = self._execute_get_messages()
            info_to_update["messages"] = messages_info["messages"]
            reward += reward_2

        # Update for messages looping
        obs_info["robot_key_to_index"][self.env.robot_id] = self.env.action_space["robot"].n - 1
        return obs_to_update, info_to_update, reward

    def _execute_get_messages(self):
        get_message_obs, reward_2, terminated, truncated, info = self.env.step(
            wrap_action_enum(Action.get_messages))
        return get_message_obs, reward_2, terminated, truncated, info

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
            # Item danger confidence for a dangerous object - [[{'item_danger_confidence': [0.51858764], 'item_danger_level': 1, 'item_location': [3. 4.], 'item_time': [171.3572073], 'item_weight': 1}]]
            # Item danger for a benign object - {{'item_danger_confidence': [0.59551547], 'item_danger_level': 1, 'item_location': [4. 3.], 'item_time': [40.21824908], 'item_weight': 1}
            # Current guess is just if > 0.5
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

    def __init__(self, env: Union[AtomicWrapper, AutomaticSensingWrapper]):
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
        broadcast_code = 0
        return {
            "action": Action.send_message.value,
            "item": 0,
            "message": "Help",  # TODO: add more information about the object
            "num_cells_move": 0,
            "robot": broadcast_code,
        }

    # Determines in which direction the agent should grab an item
    # Returns the action code for doing so
    def _find_grab_action_code(self) -> ActType:
        def subtract(tuple1: Tuple[int, int], tuple2: Tuple[int, int]) -> Tuple[int, int]:
            return tuple1[0] - tuple2[0], tuple1[1] - tuple2[1]

        dirs_to_pick_direction = {
            (-1, 0): Action.grab_up,
            (0, 1): Action.grab_right,
            (1, 0): Action.grab_down,
            (0, -1): Action.grab_left,
        }

        frame = self.last_observation["frame"]
        pos = _find_curr_agent_location(frame)
        grid_size = (len(frame), len(frame[0]))

        neighbour_pos = [cell for cell in adjacent_cells_iterator(pos, grid_size) if frame[cell] == 2]
        if len(neighbour_pos) > 0:
            # Always check directions in a different order.
            # This will result in a random object selection if there is more than 1
            # in the vicinity.
            elem = random.choice(neighbour_pos)
            relative_dir = subtract(pos, elem)
            return wrap_action_enum(dirs_to_pick_direction[relative_dir])

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
    def __init__(self, env: Union[AutomaticSensingWrapper, SimpleActions]):
        super().__init__(env)
        self.env = env

    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None) -> Tuple[
        WrapperObsType, Dict[str, Any]]:
        obs, info = self.env.reset(seed=seed, options=options)
        return self._map_observation(obs, info), info

    def step(self, action: WrapperActType) -> Tuple[WrapperObsType, SupportsFloat, bool, bool, Dict[str, Any]]:
        obs, reward, terminated, truncated, info = self.env.step(action)
        return self._map_observation(obs, info), reward, terminated, truncated, info

    @staticmethod
    def _map_observation(observation: ObsType, info: Dict[str, Any]) -> WrapperObsType:
        curr_pos, other_pos = SimpleObservations._find_agent_positions(observation["frame"])

        agent_id = SimpleObservations._position_to_agent_id(curr_pos, info["map_metadata"])
        agent_infos = SimpleObservations._get_agent_infos(other_pos + [curr_pos], info)
        num_agents = len(agent_infos) + 1
        object_infos = SimpleObservations.get_object_infos(info["map_metadata"], num_agents)

        obs = {
            "agent_id": agent_id,
            "agent_strength": observation["strength"],
            "nearby_obj_weight": observation["nearby_obj_weight"],
            "nearby_obj_danger": observation["nearby_obj_danger"],
            "agent_infos": agent_infos,
            "object_infos": object_infos,
        }
        return obs

    @staticmethod
    def _get_agent_infos(other_pos: List[Tuple[int, int]], info: Dict[str, Any]):
        id_pos_map = {
            SimpleObservations._position_to_agent_id(pos, info["map_metadata"]): pos
            for pos in other_pos
        }
        need_help_ids = set()
        for message in info["messages"]:
            agent_name_id, msg_str, t = message
            agent_id = SimpleObservations._name_to_idx(agent_name_id)
            need_help_ids.add(agent_id)

        agent_infos = [AgentInfo(id_pos_map[i], need_help=i in need_help_ids) for i in range(0, len(id_pos_map))]
        return agent_infos

    @staticmethod
    def _find_agent_positions(world_map: np.ndarray) -> (Tuple[int, int], List[Tuple[int, int]]):
        other_agent_id = 3
        curr_agent_id = 5

        positions = np.where(world_map == other_agent_id)
        # Transpose a pair of lists into list of pairs
        other_agent_pos = list(zip(*positions))

        positions = np.where(world_map == curr_agent_id)
        curr_agent_pos = next(zip(*positions))

        return curr_agent_pos, other_agent_pos

    @staticmethod
    def _position_to_agent_id(position: Tuple[int, int], map_metadata: Dict[str, List[Any]]) -> int:
        pos_i, pos_j = position
        pos_as_str = f"{pos_i}_{pos_j}"

        agent_env_id = map_metadata[pos_as_str][-1]

        return SimpleObservations._name_to_idx(agent_env_id)

    @staticmethod
    def _name_to_idx(name: str) -> int:
        # ids are assigned from 'A'
        return ord(name) - ord('A')

    @staticmethod
    def get_object_infos(map_metadata: Dict[str, List[Any]], num_agents: int) -> List[ObjectInfo]:
        sol = []
        for str_pos, location_info in map_metadata.items():
            # Check if current positions holds object information
            if type(location_info[0]) == list:
                pos = SimpleObservations._map_key_to_pos(str_pos)
                carried_by = np.zeros(num_agents)
                # An agent is holding an object if both object and agent are at the same location
                if len(location_info) > 1:
                    agent_name = location_info[1]
                    carried_by[SimpleObservations._name_to_idx(agent_name)] = 1
                sol.append(ObjectInfo(pos=pos, carried_by=carried_by))
        return sol

    @staticmethod
    def _map_key_to_pos(key: str) -> (int, int):
        vals = key.split('_')
        assert len(vals) == 2

        return int(vals[0]), int(vals[1])
