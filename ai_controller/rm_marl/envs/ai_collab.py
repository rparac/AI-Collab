import itertools
import copy
import random
from enum import Enum, IntEnum
from typing import Tuple, List, Union
from collections import defaultdict

import gymnasium as gym
import numpy as np
import pygame

from .base_grid import BaseGridEnv
from .wrappers import LabelingFunctionWrapper, SingleAgentEnvWrapper
from ..utils.grid_world import adjacent_cells_iterator, inbounds

FIX_ORIENTATION = False
FIX_NEW_PICKUP = True
assert not (FIX_ORIENTATION and FIX_NEW_PICKUP), "the fixes are exclusive"

class AICollab(BaseGridEnv):
    class Actions(IntEnum):
        none = 0
        up = 1
        right = 2
        down = 3
        left = 4
        pick_up = 5
        drop = 6
        # ask_for_help = 7

    def __init__(
            self,
            render_mode=None, size=20, positions=None, file=None, initial_pos_seed=0,
            max_steps=5000,
            num_agents=1,
            randomize_objects_location=False
    ):
        super().__init__(render_mode, size, positions, file)
        self.initial_pos_seed = initial_pos_seed
        self.max_steps = max_steps
        self.num_agents = num_agents
        self.randomize_objects_location = randomize_objects_location

        self.dropped_in_safe_zone = set()
        self.object_carried = defaultdict(lambda: None)
        self.num_steps = 0

        self.agent_ids = set(self.agent_locations.keys())
        self.object_ids = set(self.object_infos.keys())
        self.goal_coords = self.postions_by_type("ZZ")["ZZ"]

        self.agent_infos = {
            aid: {
                "pos": self._get_agent_start_pos(aid),
                # "need_help": False
            }
            for aid in self.agent_ids
        }

        if FIX_ORIENTATION:
            self._orientation = {
                aid: self.Actions.up
                for aid in self.agent_ids
            }
        else:
            self._orientation = self.Actions.up

        pos_max = (self.size, self.size)
        self.object_info_space = gym.spaces.Dict({
            "pos": gym.spaces.MultiDiscrete(list(pos_max), dtype=np.int32),
            "carried_by": gym.spaces.MultiBinary(num_agents),
            "was_dropped": gym.spaces.Discrete(2),
        })

        self.unflatten_observation_space = gym.spaces.Dict({
            "agent_id": gym.spaces.Discrete(num_agents),
            "agent_strength": gym.spaces.Discrete(num_agents, start=1),
            "agent_pos": gym.spaces.MultiDiscrete(list(pos_max), dtype=np.int32),
            "nearby_obj_weight": gym.spaces.Discrete(num_agents + 2),
            "nearby_obj_danger": gym.spaces.Discrete(2),  # is or isn't dangerous,
            "nearby_obj_was_dropped": gym.spaces.Discrete(2),  # whether it has already been dropped in the zone
            **{o: gym.spaces.utils.flatten_space(self.object_info_space) for o in self.object_ids}
        })

        # SB3 doesn't like nested obs space so we flatten it
        self.observation_space = gym.spaces.Dict({
            a: gym.spaces.utils.flatten_space(self.unflatten_observation_space)
            for a in self.agent_ids
        })
        self.action_space = gym.spaces.Discrete(len(self.Actions))

    def _get_obs(self):
        sol = {}
        for i in range(self.num_agents):
            a_name = self.id_to_name(i)

            nearby_obj_weight, nearby_obj_danger, nearby_object_was_dropped = self.nearby_obj_info(a_name)
            sol.update({
                a_name: gym.spaces.utils.flatten(self.unflatten_observation_space, {
                    "agent_id": self.name_to_id(a_name),
                    "agent_strength": self.strength(a_name),
                    "agent_pos": self.agent_infos[a_name]["pos"],
                    "nearby_obj_weight": nearby_obj_weight,
                    "nearby_obj_danger": nearby_obj_danger,
                    "nearby_obj_was_dropped": nearby_object_was_dropped,
                    **{o: gym.spaces.flatten(self.object_info_space, i) for o, i in self.object_infos.items()},
                })
            })

        return sol

    def unflatten_obs(self, flatten_obs):
        """
        utility method to decode the obs
        """
        unflatten_obs = {
            a: gym.spaces.utils.unflatten(self.unflatten_observation_space, flatten_obs[a])
            for a in self.agent_ids
        }

        sol = {}
        for a, obs in unflatten_obs.items():
            sol[a] = {}
            for k, o in obs.items():
                if k in self.object_ids:
                    sol[a].update({k: gym.spaces.utils.unflatten(self.object_info_space, o)})
                else:
                    sol[a].update({k: o})
        return sol

    def _get_info(self):
        return {}

    def _draw_component(self, label, pos, canvas):
        if label[0] == "W":
            pygame.draw.rect(
                canvas,
                self.COLOR_MAPPING["Y"],
                pygame.Rect(
                    self.pix_square_size * pos,
                    (self.pix_square_size, self.pix_square_size),
                ),
            )
        elif label[0] == "D":
            pygame.draw.circle(
                canvas,
                self.COLOR_MAPPING["R"],
                (pos + 0.5) * self.pix_square_size,
                self.pix_square_size / 6,
            )
        elif label[0] == "S":
            pygame.draw.circle(
                canvas,
                self.COLOR_MAPPING["G"],
                (pos + 0.5) * self.pix_square_size,
                self.pix_square_size / 6,
            )
        elif label[0] == "Z":  # Zone
            pygame.draw.rect(
                canvas,
                self.COLOR_MAPPING["B"],
                pygame.Rect(
                    self.pix_square_size * pos,
                    (self.pix_square_size, self.pix_square_size),
                ),
            )
        elif label[0] in ["A"]:
            pygame.draw.circle(
                canvas,
                (255,105,180) if label[-1] == "1" else (0, 255, 127),
                (pos + 0.5) * self.pix_square_size,
                self.pix_square_size / 3,
            )
            # bug on mac
            # font = pygame.font.SysFont(None, 30)
            # img = font.render(label, True, self.COLOR_MAPPING["K"])
            # canvas.blit(img, self.pix_square_size * (pos + 0.25))

    def _can_enter(self, new_pos, include_objs=False):
        if FIX_NEW_PICKUP:
            walls = self.postions_by_type("W").items()
            agents = self.postions_by_type("A").items()
            chain = itertools.chain(
                walls, agents,
                self.postions_by_type("S").items(),
                self.postions_by_type("D").items()
            ) if include_objs else itertools.chain(walls, agents)
            for label, positions in chain:
                if tuple(new_pos) in positions:
                    return False
            return True
        ###########
        else:
            walls = self.postions_by_type("W").items()
            agents = self.postions_by_type("A").items()
            s_items = self.postions_by_type("S").items()
            d_items = self.postions_by_type("D").items()
            for label, positions in itertools.chain(walls, agents, s_items, d_items):
                if tuple(new_pos) in positions:
                    return False
            return True

    def reset(self, seed=None, options=None):
        if FIX_ORIENTATION:
            self._orientation = {
                aid: self.Actions.up
                for aid in self.agent_ids
            }
        else:
            self._orientation = self.Actions.up
        self.object_carried.clear()
        self.num_steps = 0
        self.agent_infos = {
            self.id_to_name(i): {
                "pos": self._get_agent_start_pos(self.id_to_name(i)),
                # "need_help": False
            }
            for i in range(self.num_agents)
        }
        self.dropped_in_safe_zone.clear()

        # override reset to allow randomize postions
        gym.Env.reset(self, seed=seed)

        self.active_flags.clear()
        self.positions.clear()
        for p, l in self.original_positions.items():
            if self.randomize_objects_location and l[0] in ("S", "D"):
                continue
            self.positions[p].append(l)

        if self.randomize_objects_location:
            for p, l in self.original_positions.items():
                if l[0] in ("S", "D"):
                    p = self._randomize_object_location(p)
                    self.positions[p].append(l)

        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, info

    def _randomize_object_location(self, original_p):
        p = (self._np_random.integers(1, self.size-1), self._np_random.integers(1, self.size-1))
        while self.positions[p]:
            p = (self._np_random.integers(1, self.size-1), self._np_random.integers(1, self.size-1))
        return p

    def step(self, actions):
        agent_reward = {
            aid: 0
            for aid in actions.keys()
        }

        previous_agent_infos = copy.deepcopy(self.agent_infos)
        for aid, action in actions.items():
            # self.cancel_help_request(aid)

            act = self.Actions(action)
            if act == self.Actions.pick_up:
                self.pick_up_step(aid)
            elif act == self.Actions.drop:
                agent_reward[aid] = self.drop_step(aid)
            # elif act == self.Actions.ask_for_help:
            #     self.ask_for_help(aid)
            elif act < self.Actions.pick_up:
                if act != self.Actions.none:
                    if FIX_ORIENTATION:
                        self._orientation[aid] = act
                    else:
                        self._orientation = act
                direction = self._act_to_direction(act)
                new_agent_pos = np.clip(
                    self.agent_infos[aid]["pos"] + direction, 0, self.size - 1
                )
                self.move_agent_to(aid, tuple(new_agent_pos))

        for aid in self.agent_ids:
            if aid in actions: # and actions[aid] < self.Actions.pick_up:
                if FIX_NEW_PICKUP:
                    if not self.object_carried[aid]:
                        continue
                    if self.strength(aid) >= self.get_object_weight(self.object_carried[aid]):
                        self.positions[tuple(previous_agent_infos[aid]["pos"])].remove(self.object_carried[aid])
                        self.positions[tuple(self.agent_infos[aid]["pos"])].append(self.object_carried[aid])
                    elif any(elem in self.positions[tuple(previous_agent_infos[aid]["pos"])] 
                             for elem in self.object_ids 
                             if elem != self.object_carried[aid]):
                        # in theory there is a way 2 agents could be on the same cell
                        self.move_agent_to(aid, tuple(previous_agent_infos[aid]["pos"]))
                    else:
                        agent_reward[aid] = self.drop_step(aid, tuple(previous_agent_infos[aid]["pos"]))
                #####
                else:
                    if self.object_carried[aid] and self.strength(aid) >= self.get_object_weight(self.object_carried[aid]):
                        self.positions[tuple(previous_agent_infos[aid]["pos"])].remove(self.object_carried[aid])
                        self.positions[tuple(self.agent_infos[aid]["pos"])].append(self.object_carried[aid])
                    else:
                        self.drop_step(aid)

        # All dangerous objects have been moved to the zone
        terminated = all(
            d in self.dropped_in_safe_zone 
            for d in self.postions_by_type("D").keys()
        )

        observation = self._get_obs()
        info = self._get_info()
        if self.render_mode == "human":
            self._render_frame()

        self.num_steps += 1
        reward = sum(ar or -0.01 for ar in agent_reward.values())
        reward /= 20

        return observation, reward, terminated, self.num_steps >= self.max_steps, info

    def pick_up_step(self, a_name: str):
        if FIX_NEW_PICKUP:
            if self.object_carried[a_name]:
                return
            
            curr_pos = self.agent_infos[a_name]["pos"]
            object_picked_up = self._pick_by(self.positions[tuple(curr_pos)], lambda x: x[0] in ["D", "S"])
            if object_picked_up:
                self.object_carried[a_name] = object_picked_up
        #######
        else:
            curr_pos = self.agent_infos[a_name]["pos"]
            neighbour_pos = [cell for cell in adjacent_cells_iterator(curr_pos, self.size) if
                                any(
                                    (self.get_object_weight(elem) <= self.strength(a_name)) and
                                    (elem not in self.object_carried.values())
                                    for elem in self.positions[cell] 
                                    if elem[0] in ["D", "S"]
                                )
                             ]
            if len(neighbour_pos) > 0:
                # Always check directions in a different order.
                # This will result in a random object selection if there is more than 1
                # in the vicinity.
                chosen_object_pos = random.choice(neighbour_pos)
                relative_dir = chosen_object_pos - curr_pos
                if FIX_ORIENTATION:
                    self._orientation[a_name] = self._dir_to_action(tuple(relative_dir))
                else:
                    self._orientation = self._dir_to_action(tuple(relative_dir))

                self.positions[tuple(curr_pos)].remove(a_name)
                self.object_carried[a_name] = self._pick_by(self.positions[tuple(chosen_object_pos)], lambda x: x[0] in ["D", "S"])
                self.positions[tuple(chosen_object_pos)].append(a_name)
                self.agent_infos[a_name]["pos"] = np.array(chosen_object_pos)

    @staticmethod
    def get_object_weight(obj_id):
        # return int(obj_id[-1])
        return int(obj_id.split("_")[-1])

    @staticmethod
    def _pick_by(l: List[str], cond):
        for x in l:
            if cond(x):
                return x
        return None

    def drop_step(self, a_name: str, agent_pos: tuple = None):
        if FIX_NEW_PICKUP:
            if not self.object_carried[a_name]:
                return 0

            curr_pos = agent_pos or self.agent_infos[a_name]["pos"]
            if any(
                elem != self.object_carried[a_name] 
                for elem in self.positions[tuple(curr_pos)]
                if elem[0] in ("D", "S")
            ):
                # can't drop an object if there is already another object in the location
                return 0

            self.object_carried[a_name] = None
            if tuple(curr_pos) in self.goal_coords:
                obj_dropped = self._pick_by(self.positions[tuple(curr_pos)], lambda x: x[0] in ["D", "S"])
                if obj_dropped and obj_dropped not in self.dropped_in_safe_zone:
                    reward = -5 if obj_dropped[0] == "S" else 10
                    self.dropped_in_safe_zone.add(obj_dropped)
                    return reward
            return 0
        ##########
        else:
            # handle orientation
            # drop in the cell that should be adjacent to object's orientation
            if FIX_ORIENTATION:
                dir = self._act_to_direction(self._orientation[a_name])
            else:
                dir = self._act_to_direction(self._orientation)
            old_location = self.agent_infos[a_name]["pos"]
            new_agent_pos = old_location - dir
            if inbounds((self.size, self.size), new_agent_pos[0], new_agent_pos[1]) and self.object_carried[a_name]:
                self.positions[tuple(old_location)].remove(a_name)
                self.positions[tuple(new_agent_pos)].append(a_name)
                self.agent_infos[a_name]["pos"] = np.array(new_agent_pos)
                self.object_carried[a_name] = None

                if tuple(old_location) in self.goal_coords:
                    obj_dropped = self._pick_by(self.positions[tuple(old_location)], lambda x: x[0] in ["D", "S"])
                    if obj_dropped not in self.dropped_in_safe_zone:
                        reward = -5 if obj_dropped[0] == "S" else 10
                        self.dropped_in_safe_zone.add(obj_dropped)
                        return reward

            return -0.01

    # @property
    # def agent_locations(self):
    #     return {
    #         l: np.array(p) for p, ls in self.positions.items() for l in ls if l.startswith("A")
    #     }

    def _dir_to_action(self, dir: Tuple[int, int]):
        self._dict = {
            tuple([0, -1]): self.Actions.up,
            tuple([1, 0]): self.Actions.right,
            tuple([0, 1]): self.Actions.down,
            tuple([-1, 0]): self.Actions.left,
            tuple([0, 0]): self.Actions.none,
        }
        return self._dict[dir]

    def _act_to_direction(self, action: Actions):
        self._dict = {
            self.Actions.up: np.array([0, -1]),
            self.Actions.right: np.array([1, 0]),
            self.Actions.down: np.array([0, 1]),
            self.Actions.left: np.array([-1, 0]),
            self.Actions.none: np.array([0, 0]),
        }
        return self._dict[action]

    def strength(self, a_name: str) -> int:
        curr_pos = tuple(self.agent_infos[a_name]["pos"])
        strength = 0
        for adj_cell in adjacent_cells_iterator(curr_pos, self.size, eight_dir=True):
            for agent in self.agent_infos.keys():
                if agent in self.positions[adj_cell]:
                    strength += 1

        return strength

    def nearby_obj_info(self, a_name: str):
        if FIX_NEW_PICKUP:
            curr_pos = tuple(self.agent_infos[a_name]["pos"])
            for entity in self.positions[curr_pos]:
                if entity[0] in ['S', 'D'] and entity not in self.object_carried.values():
                    weight = self.get_object_weight(entity)
                    danger = int(entity[0] == 'D')
                    was_dropped = entity in self.dropped_in_safe_zone
                    return weight, danger, was_dropped
            return 0, 0, 0
        ###
        else:
            curr_pos = tuple(self.agent_infos[a_name]["pos"])
            for adj_cell in adjacent_cells_iterator(curr_pos, self.size):
                if adj_cell in self.goal_coords:
                    continue
                for entity in self.positions[adj_cell]:
                    if entity[0] in ['S', 'D'] and entity not in self.object_carried.values():
                        weight = self.get_object_weight(entity)
                        danger = int(entity[0] == 'D')
                        was_dropped = entity in self.dropped_in_safe_zone
                        return weight, danger, was_dropped
            return 0, 0, 0

    @property
    def object_infos(self):
        sol = {}
        s_items = self.postions_by_type("S").items()
        d_items = self.postions_by_type("D").items()
        for label, positions in tuple(s_items) + tuple(d_items):
            if FIX_NEW_PICKUP:
                carried_by = np.zeros(len(self.agent_ids))
                for aid in self.agent_ids:
                    if self.object_carried[aid] == label:
                        carried_by[self.name_to_id(aid)] = 1

                sol[label] = {
                    "pos": np.array(positions[0]),
                    "carried_by": carried_by,
                    "was_dropped": label in self.dropped_in_safe_zone
                }
            ###
            else:
                for position in positions:
                    held_by = self._pick_by(self.positions[position], lambda x: x in self.agent_ids)
                    carried_by = np.zeros(len(self.agent_ids))
                    if held_by:
                        carried_by[self.name_to_id(held_by)] = 1

                    sol[label] = {
                        # "pos": np.array(position),
                        "carried_by": carried_by,
                        "was_dropped": label in self.dropped_in_safe_zone
                    }
        return sol

    # def carries_object(self, a_name: str):
    #     return len(self.positions[tuple(self.agent_infos[a_name]["pos"])]) > 1

    @staticmethod
    def name_to_id(a_name: str):
        return int(a_name[1]) - 1

    @staticmethod
    def id_to_name(a_id: int):
        return f"A{a_id + 1}"

    def _get_agent_start_pos(self, a_name: str):
        return np.array(self.postions_by_type(a_name, pos={k: [v] for k, v in self.original_positions.items()})[a_name][0])

    def cancel_help_request(self, a_name: str):
        self.agent_infos[a_name] = {
            "pos": self.agent_infos[a_name]["pos"],
            # "need_help": 0,
        }

    def ask_for_help(self, a_name: str):
        self.agent_infos[a_name] = {
            "pos": self.agent_infos[a_name]["pos"],
            # "need_help": 1,
        }

    def move_agent_to(self, aid, new_pos):
        if self._can_enter(new_pos, include_objs=self.object_carried[aid]):
            old_pos = tuple(self.agent_infos[aid]["pos"])
            self.positions[old_pos].remove(aid)
            self.positions[tuple(new_pos)].append(aid)
            self.agent_infos[aid]["pos"] = np.array(new_pos)

    @staticmethod
    def is_A1_close_to_object(env):
        object_pos = tuple(env.postions_by_type("D2_2")["D2_2"][0])
        agent_pos = tuple(env.agent_infos["A1"]["pos"])

        adj_cells = [
            cell 
            for cell in adjacent_cells_iterator(object_pos, env.size-1, eight_dir=True, include_current=False)
        ]

        return agent_pos in adj_cells

    @staticmethod
    def is_A1_carrying(env):
        return env.object_carried["A1"] == "D2_2"
    
    @staticmethod
    def is_A2_carrying(env):
        return env.object_carried["A2"] == "D2_2"
    
    @staticmethod
    def is_A1_on_obj(env):
        agent_pos = tuple(env.agent_infos["A1"]["pos"]) 
        object_pos = tuple(env.postions_by_type("D2_2")["D2_2"][0])
        return agent_pos == object_pos
    
    @staticmethod
    def is_A1_near_A2(env):
        return env.strength("A1") > 1
    
    @staticmethod
    def is_A1_in_zone(env):
        zone = env.postions_by_type("ZZ")["ZZ"]
        a1_pos = tuple(env.agent_infos["A1"]["pos"])
        return a1_pos in zone

    @staticmethod
    def A1_carries_object(env):
        object_pos = env.postions_by_type("D2_2")["D2_2"][0]
        agent_pos = tuple(env.agent_infos["A1"]["pos"])
        env.positions[object_pos].remove("D2_2")
        env.positions[agent_pos].append("D2_2")
        env.object_carried["A1"] = "D2_2"

    @staticmethod
    def move_A1_towards_zone(env):
        zone = env.postions_by_type("ZZ")["ZZ"]
        a1_pos = np.array(env.agent_infos["A1"]["pos"])
        zone_pos_ix = sorted([
            (i, abs(a1_pos[0] - z[0]) + abs(a1_pos[1] - z[1])) # manhattan distance
            for i, z in enumerate(zone)
        ], key=lambda i: i[1])[-1][0]
        zone_pos = zone[zone_pos_ix]
        dir = np.sign(a1_pos - zone_pos)
        if sum(dir) == 2:
            dir[-1] = 0
        move_to = a1_pos - dir
        env.move_agent_to("A1", move_to)

        if env.object_carried["A1"] and env.strength("A1") >= env.get_object_weight(env.object_carried["A1"]):
            env.positions[tuple(a1_pos)].remove(env.object_carried["A1"])
            env.positions[tuple(move_to)].append(env.object_carried["A1"])
        else:
            env.object_carried["A1"] = None
            return lambda ls: [l for l in ls if l not in ("a1l", "a2h", "_X")]

    @staticmethod
    def move_A2_towards_zone(env):
        zone = env.postions_by_type("ZZ")["ZZ"]
        a2_pos = np.array(env.agent_infos["A2"]["pos"])
        zone_pos_ix = sorted([
            (i, abs(a2_pos[0] - z[0]) + abs(a2_pos[1] - z[1])) # manhattan distance
            for i, z in enumerate(zone)
        ], key=lambda i: i[1])[-1][0]
        zone_pos = zone[zone_pos_ix]
        dir = np.sign(a2_pos - zone_pos)
        if sum(dir) == 2:
            dir[-1] = 0
        move_to = a2_pos - dir
        env.move_agent_to("A2", move_to)

    @staticmethod
    def move_A1_towards_obj(env):
        a1_pos = np.array(env.agent_infos["A1"]["pos"])
        object_pos = tuple(env.postions_by_type("D2_2")["D2_2"][0])
        dir = np.sign(a1_pos - object_pos)
        if sum(dir) == 2:
            dir[-1] = 0
        move_to = a1_pos - dir
        env.move_agent_to("A1", move_to)
    
    @staticmethod
    def move_A2_towards_A1(env):
        a1_pos = np.array(env.agent_infos["A1"]["pos"])
        a2_pos = np.array(env.agent_infos["A2"]["pos"])
        dir = np.sign(a2_pos - a1_pos)
        if sum(dir) == 2:
            dir[-1] = 0
        move_to = a2_pos - dir
        env.move_agent_to("A2", move_to)


class AICollabLabellingFunctionWrapper(LabelingFunctionWrapper):

    def get_labels(self, obs: dict = None, prev_obs: dict = None):
        """Returns a modified observation."""
        labels = []

        for aid, flatten_obs in obs.items():

            unwrapped_obs = gym.spaces.unflatten(self.env.unflatten_observation_space, flatten_obs)
            label_agent_id = aid.lower()
            agent_ix = unwrapped_obs["agent_id"]

            if unwrapped_obs["agent_strength"] > 1:
                labels.append(f'{label_agent_id}h')

            if any(
                obj_info["carried_by"][agent_ix] > 0
                for obj_info in self.env.unwrapped.object_infos.values()
            ):
                labels.append(f'{label_agent_id}l')
            elif unwrapped_obs["nearby_obj_weight"] > 0 and unwrapped_obs["nearby_obj_danger"] == 1:
                labels.append(f'{label_agent_id}d')

            pos = tuple(unwrapped_obs["agent_pos"])
            if pos in self.env.unwrapped.goal_coords:
                labels.append(f'{label_agent_id}z')

        return labels


class OriginalAICollabLabellingFunctionWrapper(LabelingFunctionWrapper):
    """
    Labelling function wrapper for Julian's environment which doesn't have
    the space flattened out (will probably change)
    """

    def get_labels(self, obs: dict = None, prev_obs: dict = None):
        """Returns a modified observation."""
        labels = []

        for aid, flatten_obs in obs.items():

            unwrapped_obs = gym.spaces.unflatten(self.env_dict[aid].unflatten_observation_space, flatten_obs)
            label_agent_id = aid.lower()
            agent_ix = unwrapped_obs["agent_id"]

            object_info = {k: gym.spaces.unflatten(self.object_info_space, unwrapped_obs[k]) for k in
                           self.env_dict[aid].object_info_keys}

            if unwrapped_obs["agent_strength"] > 1:
                labels.append(f'{label_agent_id}h')

            if any(
                    v["carried_by"][agent_ix] > 0
                    for k_, v in object_info.items()
            ):
                labels.append(f'{label_agent_id}l')
            elif unwrapped_obs["nearby_obj_weight"] > 0 and unwrapped_obs["nearby_obj_danger"] == 1:
                labels.append(f'{label_agent_id}d')

            pos = tuple(unwrapped_obs['agent_pos'])
            if pos in self.env_dict[aid].unwrapped.goal_coords:
                labels.append(f'{label_agent_id}z')

        return labels

