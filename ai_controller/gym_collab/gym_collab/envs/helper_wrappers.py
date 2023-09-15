"""
Simplifying wrappers to make the AI collab environment results easier to use.
"""
import threading
from collections import defaultdict
from dataclasses import dataclass
import random
import asyncio
import concurrent.futures
from typing import Tuple, SupportsFloat, Any, Dict, List, Optional, Union

import gymnasium as gym
import numpy as np
from gymnasium.core import ObsType, WrapperActType, WrapperObsType, ActType

from gym_collab.envs import AICollabEnv
from gym_collab.envs.utils import adjacent_cells_iterator, wrap_action_enum, _find_curr_agent_location, \
    find_object_held_location
from .action import Action

true_msg_prefix = "True"
no_objects_msg = "<empty>"


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

        self.last_frame = None
        self.last_map_metadata = None
        self.dropped_in_safe_zone = set()

    def reset(
            self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None
    ) -> Tuple[WrapperObsType, Dict[str, Any]]:
        self.env.reset(seed=seed, options=options)

        self.dropped_in_safe_zone.clear()
        obs, info, _reward = self._update_with_sensing_info()
        self.last_frame = obs["frame"]
        self.last_map_metadata = info["map_metadata"]
        return obs, info

    def step(
            self, action: WrapperActType
    ) -> Tuple[WrapperObsType, SupportsFloat, bool, bool, Dict[str, Any]]:
        if action['action'] == Action.drop_object.value:
            a_pos, _ = SimpleObservations.find_agent_positions(self.last_frame)
            if a_pos in self.env.goal_coords:
                self.dropped_in_safe_zone.add(
                    SimpleObservations.get_object_id(a_pos, self.last_frame, self.last_map_metadata))

        next_observation, reward, terminated, truncated, info = self.env.step(action)
        next_observation, info, reward2 = self._update_with_sensing_info(terminated, truncated, next_observation, info)
        self.last_frame = next_observation["frame"]
        self.last_map_metadata = info["map_metadata"]

        self._send_message_if_holding_object(self.last_frame, self.last_map_metadata)

        return next_observation, reward + reward2, terminated, truncated, info

    def _update_with_sensing_info(self, terminated=False, truncated=False, obs_to_update=None, info_to_update=None):
        reward = 0
        if not terminated and not truncated:
            occupancy_map_obs, reward_2, terminated, truncated, obs_info = self._execute_get_occupancy_map()
            # Update for messages looping
            obs_info["robot_key_to_index"][self.env.robot_id] = self.env.action_space["robot"].n - 1
            if info_to_update is None and obs_to_update is None:
                info_to_update = obs_info
                obs_to_update = occupancy_map_obs
            else:
                obs_to_update["frame"] = occupancy_map_obs["frame"]
                info_to_update["map_metadata"] = obs_info["map_metadata"]
                obs_to_update["action_status"] = np.logical_or(obs_to_update["action_status"],
                                                               occupancy_map_obs["action_status"])
                # Wait doesn't update this so it is safer to get it from occupancy map
                obs_to_update["strength"] = occupancy_map_obs["strength"]
            reward += reward_2
        if not terminated and not truncated:
            danger_sensing_obs, reward_2, terminated, truncated, danger_info = self._execute_danger_sensing(
                obs_to_update["frame"],
                info_to_update["map_metadata"]
            )
            obs_to_update["nearby_obj_weight"] = danger_sensing_obs["nearby_obj_weight"]
            obs_to_update["nearby_obj_danger"] = danger_sensing_obs["nearby_obj_danger"]
            obs_to_update["nearby_obj_was_dropped"] = danger_sensing_obs["nearby_obj_was_dropped"]
            reward += reward_2

        if not terminated and not truncated:
            get_messages_obs, reward_2, terminated, truncated, messages_info = self._execute_get_messages()
            info_to_update["messages"] = messages_info["messages"]
            reward += reward_2

        return obs_to_update, info_to_update, reward

    def _send_message_if_holding_object(self, occupancy_map, map_metadata):
        pos = _find_curr_agent_location(occupancy_map)
        # Send a message that no objects are present if none are
        send_action = self._get_send_message_action(no_objects_msg)
        if self.env.unwrapped.extra.get('carrying_object', '') != '':
            object_id = SimpleObservations.get_object_id(pos, occupancy_map, map_metadata)
            send_action = self._get_send_message_action(object_id)

        # Don't care about the output of sending a message.
        # We should already have all the information
        self.env.step(send_action)

    @staticmethod
    def _get_send_message_action(object_id: str) -> ActType:
        broadcast_code = 0
        return {
            "action": Action.send_message.value,
            "item": 0,
            "message": object_id,  # TODO: add more information about the object
            "num_cells_move": 0,
            "robot": broadcast_code,
        }

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
        # The line above commented out if we will allow other pick-up
        # for next_pos in adjacent_cells_iterator(pos, grid_size):
        for next_pos in [pos]:
            idx = self._find_object_index(map_metadata, info['object_key_to_index'], next_pos)
            # TODO: This may be different if there are multiple objects
            if idx != -1 and self.env.unwrapped.extra.get('carrying_object', '') == '':
                check_item_code = 20
                action = {"action": check_item_code, "item": idx, "message": "empty", "num_cells_move": 1, "robot": 0}
                obs, _reward, terminated, truncated, info = self.env.step(action)
                objects_nearby.append(obs["item_output"])

        obs["nearby_obj_weight"] = 0
        obs["nearby_obj_danger"] = 0
        obs["nearby_obj_was_dropped"] = 0

        # drop
        # self.pos in self.goal_coords

        # Randomly choose which object to show the value for if there are more
        if len(objects_nearby) > 0:
            elem = random.choice(objects_nearby)
            # We ignore the probability for now.
            # Item danger confidence for a dangerous object - [[{'item_danger_confidence': [0.51858764], 'item_danger_level': 1, 'item_location': [3. 4.], 'item_time': [171.3572073], 'item_weight': 1}]]
            # Item danger for a benign object - {{'item_danger_confidence': [0.59551547], 'item_danger_level': 1, 'item_location': [4. 3.], 'item_time': [40.21824908], 'item_weight': 1}
            # Current guess is just if > 0.5
            # Map 0 - unknown, 1 - not dangerous, 2 - dangerous -> 0 - not dangerous, 1 - dangerous
            # TODO: fix danger sensing
            obs["nearby_obj_danger"] = 1  # elem["item_danger_level"] - 1
            obs["nearby_obj_weight"] = elem["item_weight"]
            item_loc_xy = elem["item_location"]
            obj_pos = (int(item_loc_xy[1]), int(item_loc_xy[0]))
            obj_id = SimpleObservations.get_object_id(obj_pos, occupancy_map, map_metadata)
            obs["nearby_obj_was_dropped"] = obj_id in self.dropped_in_safe_zone

        return obs, _reward, terminated, truncated, info

    # Gets valid index for check_item action or -1 if none exist
    @staticmethod
    def _find_object_index(map_metadata: Dict[str, List[List[Any]]], obj_key_to_idx: Dict[str, int],
                           pos: Tuple[int, int]) -> int:
        pos_x, pos_y = pos
        map_key = f"{pos_y}_{pos_x}"
        # No key or sensed an agent
        if map_key not in map_metadata or not isinstance(map_metadata[map_key][0], list):
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
    Important: up/right/down/left correspond to directions in the occupancy
    map which is for some reason mirroed image of the simulator
    """

    def __init__(self, env: Union[AtomicWrapper, AutomaticSensingWrapper]):
        super().__init__(env)
        self.num_agents = env.action_space["robot"].n
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
        # pos = _find_curr_agent_location(frame)
        # grid_size = (len(frame), len(frame[0]))
        #
        # neighbour_pos = [cell for cell in adjacent_cells_iterator(pos, grid_size) if frame[cell] == 2]
        # if len(neighbour_pos) > 0:
        #     # Always check directions in a different order.
        #     # This will result in a random object selection if there is more than 1
        #     # in the vicinity.
        #     elem = random.choice(neighbour_pos)
        #     relative_dir = subtract(pos, elem)
        #     return wrap_action_enum(dirs_to_pick_direction[relative_dir])

        # Attempt to grab up if everything fails.
        # This will result in the same errors as an incorrect grab
        up_grab_code = Action.grab_current_pos  # grab_down #grab_current_pos
        # up_grab_code = next(iter(dirs_to_pick_direction.values()))
        return wrap_action_enum(up_grab_code)

    @staticmethod
    def _map_action_code(action: WrapperActType) -> ActType:
        new_to_old_map = {
            0: Action.wait,  # do nothing
            3: Action.move_up,  # up
            4: Action.move_right,  # right
            1: Action.move_down,  # down
            2: Action.move_left,  # left
            6: Action.drop_object,  # drop
        }
        return wrap_action_enum(new_to_old_map[action])


class SimpleObservations(gym.Wrapper):
    def __init__(self, env: Union[AutomaticSensingWrapper, SimpleActions]):
        super().__init__(env)
        self.env = env

        assert isinstance(self.env.observation_space, gym.spaces.Dict)

        pos_max = self.env.observation_space['frame'].shape
        num_agents = self.env.num_agents
        self.object_info_space = gym.spaces.Dict({
            "pos": gym.spaces.MultiDiscrete(list(pos_max), dtype=np.int32),
            "carried_by": gym.spaces.MultiBinary(num_agents),
            "was_dropped": gym.spaces.Discrete(2),
        })

        # TODO: there is no easy way to obtain object ids at the moment
        #  Do a map sweep to get them. It is not too expensive if done once
        self.object_info_keys = ['D0_2']  # , 'D1_1']

        self.unflatten_observation_space = gym.spaces.Dict({
            "agent_id": gym.spaces.Discrete(num_agents),
            "agent_strength": gym.spaces.Discrete(num_agents, start=1),
            "agent_pos": gym.spaces.MultiDiscrete(list(pos_max), dtype=np.int32),
            "nearby_obj_weight": gym.spaces.Discrete(num_agents + 2),
            "nearby_obj_danger": gym.spaces.Discrete(2),  # is or isn't dangerous,
            "nearby_obj_was_dropped": gym.spaces.Discrete(2),  # whether it has already been dropped in the zone
            **{o: gym.spaces.utils.flatten_space(self.object_info_space) for o in self.object_info_keys}
        })

        self.observation_space = self.unflatten_observation_space

        self.agent_id = None
        self.other_agent_ids = None

        # key -> agent_idx
        self.get_object_carrier = defaultdict(lambda: -1)

    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None) -> Tuple[
        WrapperObsType, Dict[str, Any]]:
        obs, info = self.env.reset(seed=seed, options=options)

        curr_pos, other_pos = SimpleObservations.find_agent_positions(obs["frame"])
        agent_id = SimpleObservations._pos_to_agent_id(curr_pos, info["map_metadata"])
        other_ids = [SimpleObservations._pos_to_agent_id(pos, info["map_metadata"]) for pos in other_pos]
        self.agent_id = agent_id
        self.other_agent_ids = other_ids

        return self._map_observation(obs, info), info

    def step(self, action: WrapperActType) -> Tuple[WrapperObsType, SupportsFloat, bool, bool, Dict[str, Any]]:
        obs, reward, terminated, truncated, info = self.env.step(action)
        return self._map_observation(obs, info), reward, terminated, truncated, info

    def _map_observation(self, observation: ObsType, info: Dict[str, Any]) -> WrapperObsType:
        self._update_carried_info(info)
        # agent_infos = self._get_agent_infos(self.other_agent_ids + [self.agent_id], info)
        object_infos = self.get_object_infos(observation["frame"], info["map_metadata"], self.num_agents)

        obs_new = {
            "agent_id": self.agent_id,
            "agent_strength": observation["strength"],
            "agent_pos": np.array(self.agent_name_to_pos(self.idx_to_name(self.agent_id), info["map_metadata"])),
            "nearby_obj_weight": observation["nearby_obj_weight"],
            "nearby_obj_danger": observation["nearby_obj_danger"],
            "nearby_obj_was_dropped": observation["nearby_obj_was_dropped"],
            **object_infos,
        }
        return obs_new

    def _update_carried_info(self, info):
        for message in info["messages"]:
            agent_name, msg_str, t = message
            agent_id = SimpleObservations.name_to_idx(agent_name)
            if msg_str == no_objects_msg:
                self.get_object_carrier = defaultdict(lambda: -1, {k: v for k, v in self.get_object_carrier.items() if
                                                                   v != agent_id})
            else:
                object_id = msg_str
                self.get_object_carrier[object_id] = agent_id

    @staticmethod
    def _get_agent_infos(agent_ids: List[int], info: Dict[str, Any]):
        need_help_ids = set()
        for message in info["messages"]:
            agent_name_id, msg_str, t = message
            agent_id = SimpleObservations.name_to_idx(agent_name_id)
            need_help_ids.add(agent_id)

        id_pos_map = {
            id_: SimpleObservations.agent_name_to_pos(SimpleObservations.idx_to_name(id_), info["map_metadata"])
            for id_ in agent_ids
        }

        # TODO: put back when multi-agent casew works
        agent_infos = {SimpleObservations.idx_to_name(i):
                           SimpleObservations.agent_info(id_pos_map[i], need_help=i in need_help_ids)
                       for i in range(0, len(id_pos_map))}
        return agent_infos

    @staticmethod
    def agent_info(pos: (int, int), need_help: bool):
        return {
            "pos": np.array(pos),
            # "need_help": need_help,
        }

    @staticmethod
    def object_info(pos: (int, int), carried_by: np.ndarray, was_dropped: bool):
        return {
            "pos": np.array(pos),
            "carried_by": carried_by,
            "was_dropped": was_dropped,
        }

    @staticmethod
    def find_agent_positions(world_map: np.ndarray) -> (Tuple[int, int], List[Tuple[int, int]]):
        other_agent_id = 3
        curr_agent_id = 5

        # we are using (x,y) positions
        positions = np.where(world_map == other_agent_id)[::-1]
        # Transpose a pair of lists into list of pairs
        other_agent_pos = list(zip(*positions))

        # we are using (x,y) positions
        positions = np.where(world_map == curr_agent_id)[::-1]
        curr_agent_pos = next(zip(*positions))

        return curr_agent_pos, other_agent_pos

    @staticmethod
    def _pos_to_agent_id(position: Tuple[int, int], map_metadata: Dict[str, List[Any]]) -> int:
        pos_x, pos_y = position
        pos_as_str = f"{pos_y}_{pos_x}"

        agent_env_id = map_metadata[pos_as_str][-1]

        return SimpleObservations.name_to_idx(agent_env_id)

    @staticmethod
    def agent_name_to_pos(agent_id: str, map_metadata: Dict[str, List[Any]]):
        for pos_str, val in map_metadata.items():
            if agent_id in val:
                return SimpleObservations._map_key_to_pos(pos_str)
        return -1, -1

    @staticmethod
    def name_to_idx(name: str) -> int:
        # names are of the form "A{idx + 1}"
        return int(name[1:]) - 1

    @staticmethod
    def idx_to_name(idx: int) -> str:
        return f"A{idx + 1}"

    @staticmethod
    def get_object_id(pos: Tuple[int, int], occupancy_map: np.ndarray, map_metadata: Dict[str, List[any]]) -> str:
        map_key = f"{pos[1]}_{pos[0]}"
        map_info = map_metadata[map_key]
        if type(map_info[0]) != list:
            # edge case, the object is not in the same position as the agent
            pos_new = find_object_held_location(occupancy_map)
            map_info = map_metadata[f"{pos_new[1]}_{pos_new[0]}"]
            assert type(map_info[0]) == list
        id_, weight, danger_level = map_info[0]
        return f"{'D' if danger_level == 2 else 'S'}{id_}_{weight}"

    def get_object_infos(self, occupancy_map: np.ndarray, map_metadata: Dict[str, List[Any]], num_agents: int) -> Dict[
        str, Any]:
        sol = {}
        # self.env.unwrapped.extra['carrying_object'] -> return object id
        for str_pos, location_info in map_metadata.items():
            # Check if current positions holds object information
            if type(location_info[0]) == list:
                pos = SimpleObservations._map_key_to_pos(str_pos)
                sim_obj_id = location_info[0][0]
                object_id = SimpleObservations.get_object_id(pos, occupancy_map, map_metadata)
                carried_by = np.zeros(num_agents)

                object_carrier_agent = self.get_object_carrier[object_id]
                if object_carrier_agent != -1:
                    carried_by[object_carrier_agent] = 1

                # Check if the current agent is carrying the object.
                # Note that this is only done to speed up information propagation.
                # The agent will also be able to read it through a message
                if sim_obj_id == self.env.unwrapped.extra.get('carrying_object', ''):
                    carried_by[self.agent_id] = 1
                sol[object_id] = gym.spaces.flatten(self.object_info_space,
                                                    SimpleObservations.object_info(pos=pos, carried_by=carried_by,
                                                                                   was_dropped=object_id in self.dropped_in_safe_zone))
        return sol

    @staticmethod
    def _map_key_to_pos(key: str) -> (int, int):
        vals = key.split('_')
        assert len(vals) == 2

        return int(vals[1]), int(vals[0])


class AgentNameWrapper(gym.Wrapper):
    """
    Wrap an observation and action in a dictionary with agent_name.
    This is to have consistency with Leo's work, where environments contain information about
    all agents which are then projected to each agent
    """

    def __init__(self, env: SimpleObservations):
        super().__init__(env)

        self.unflatten_observation_space = env.observation_space
        self.observation_space = gym.spaces.Dict({
            SimpleObservations.idx_to_name(i): gym.spaces.utils.flatten_space(env.observation_space)
            for i in range(self.env.num_agents)
        })
        # self.action_space = gym.spaces.Dict({
        #     SimpleObservations.idx_to_name(i): env.action_space
        #     for i in range(self.env.num_agents)
        # })
        self.action_space = self.env.action_space
        self.env = env

    def reset(self, **kwargs) -> Tuple[WrapperObsType, Dict[str, Any]]:
        old_observation, info = self.env.reset(**kwargs)
        return self.observation(old_observation), info

    def step(self, action: WrapperActType) -> Tuple[WrapperObsType, SupportsFloat, bool, bool, Dict[str, Any]]:
        old_action = self.action(action)
        old_obs, reward, terminated, truncated, info = self.env.step(old_action)
        return self.observation(old_obs), reward, terminated, truncated, info

    def observation(self, observation: ObsType) -> WrapperObsType:
        agent_idx = observation["agent_id"]
        return {
            SimpleObservations.idx_to_name(agent_idx): gym.spaces.utils.flatten(self.unflatten_observation_space,
                                                                                observation)
        }

    def action(self, action: WrapperActType) -> ActType:
        assert isinstance(action, dict) and len(action) == 1
        return action[SimpleObservations.idx_to_name(self.env.agent_id)]


class EnvComposition(gym.Wrapper):
    def __init__(self, env_dict):
        first_env = next(iter(env_dict.values()))
        super().__init__(first_env)

        self.env_dict = env_dict

        self.observation_space = gym.spaces.Dict({
            k: gym.spaces.utils.flatten_space(env.observation_space)
            for k, env in self.env_dict.items()
        })
        self.action_space = first_env.action_space
        # self.action_space = gym.spaces.Dict({
        #     k: env.action_space
        #     for k, env in self.env_dict.items()
        # })

        self.prev_results = {
            k: (None, 0, False, False, None)
            for k in self.env_dict.keys()
        }

        self.tasks = {}
        self.loop = asyncio.get_event_loop()
        self.executor = concurrent.futures.ThreadPoolExecutor()

        self.can_execute_next = {}

    def reset(self, **kwargs):
        observations, info = {}, None

        dict_keys = list(self.env_dict.keys())
        dict_values = [self.env_dict[k] for k in dict_keys]
        with concurrent.futures.ThreadPoolExecutor() as executor:
            reset_obs_info = list(executor.map(lambda env_: env_.reset(**kwargs), dict_values))

        for env_id, (obs, info_) in zip(dict_keys, reset_obs_info):
            observations[env_id] = gym.spaces.utils.flatten(self.env_dict[env_id].unflatten_observation_space, obs)
            info = info_

        self.prev_results = {
            k: (observations[k], 0, False, False, info)
            for k in self.env_dict.keys()
        }
        # Keeps track whether the next step can be executed,
        # i.e. if the action had access to the latest information
        self.can_execute_next = {
            k: True
            for k in self.env_dict.keys()
        }
        return observations, info

    # def step(self, action: WrapperActType) -> Tuple[WrapperObsType, SupportsFloat, bool, bool, Dict[str, Any]]:
    #     def execute_step(env_act):
    #         env_, act_ = env_act
    #         return env_.step(act_)
    #
    #     observations, info = {}, None
    #     reward_sum = 0
    #     terminated, truncated = False, False
    #
    #     env_act_pairs = [(self.env_dict[aid], act_) for aid, act_ in action.items()]
    #     with concurrent.futures.ThreadPoolExecutor() as executor:
    #         step_outputs = list(executor.map(execute_step, env_act_pairs))
    #
    #     for aid, (obs, reward, terminated, truncated, info) in zip(action.keys(), step_outputs):
    #         curr_env = self.env_dict[aid]
    #         observations[aid] = gym.spaces.utils.flatten(curr_env.unflatten_observation_space, obs)
    #         reward_sum += reward
    #
    #     return observations, reward_sum, terminated, truncated, info

    def step(self, action: WrapperActType) -> Tuple[WrapperObsType, SupportsFloat, bool, bool, Dict[str, Any]]:
        def execute_step(env_, act_):
            out = env_.step(act_)
            return out

        for aid, act_ in action.items():
            env = self.env_dict[aid]
            # Only start a new task if one doesn't already exist for this environment
            if self.can_execute_next[aid]:
                future = self.executor.submit(execute_step, env, act_)
                self.can_execute_next[aid] = False
                self.tasks[aid] = future

        # Update the results
        for aid, task in self.tasks.items():
            if task.done():
                self.can_execute_next[aid] = True
                obs, reward, term, trunc, info = task.result()
                curr_env = self.env_dict[aid]
                flattened_obs = gym.spaces.utils.flatten(curr_env.unflatten_observation_space, obs)
                self.prev_results[aid] = flattened_obs, reward, term, trunc, info

        # Combine the previous results
        observations, reward_sum, terminated, truncated, info = {}, 0, False, False, {}
        for aid, (obs, reward, terminated_, truncated_, info_) in self.prev_results.items():
            observations[aid] = obs
            reward_sum += reward
            terminated |= terminated_
            truncated |= truncated_
            info = info_

        return observations, reward_sum, terminated, truncated, info
