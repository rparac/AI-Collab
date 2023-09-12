import itertools
import random
from collections import defaultdict
from enum import IntEnum
from typing import Tuple, List

import gymnasium as gym
import numpy as np
import pygame

from .base_grid import BaseGridEnv
from .wrappers import LabelingFunctionWrapper
from ..utils.grid_world import adjacent_cells_iterator, inbounds


# TODO: there seems to be a drop bug when the agent is obstructed.
#  It should do nothing in that case, not climb the wall

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
    ):
        super().__init__(render_mode, size, positions, file)
        self.initial_pos_seed = initial_pos_seed
        self.max_steps = max_steps
        self.num_agents = num_agents

        self.agent_infos = {
            self.id_to_name(i): {
                "pos": self._get_agent_start_pos(self.id_to_name(i)),
                # "need_help": False
            }
            for i in range(num_agents)
        }

        self.dropped_in_safe_zone = set()
        self.agent_ids = set(self.agent_locations.keys())
        self.object_ids = set(self.object_infos.keys())

        pos_max = (self.size, self.size)
        self.agent_info_space = gym.spaces.Dict({
            "pos": gym.spaces.MultiDiscrete(list(pos_max), dtype=np.int32),
            # "need_help": gym.spaces.Discrete(2),
        })
        self.object_info_space = gym.spaces.Dict({
            # "pos": gym.spaces.MultiDiscrete(list(pos_max), dtype=np.int32),
            # "carried_by": gym.spaces.MultiBinary(num_agents),
            "was_dropped": gym.spaces.Discrete(2),
        })

        self.unflatten_observation_space = gym.spaces.Dict({
            "agent_id": gym.spaces.Discrete(num_agents),
            "agent_strength": gym.spaces.Discrete(num_agents, start=1),
            "nearby_obj_weight": gym.spaces.Discrete(num_agents + 2),
            "nearby_obj_danger": gym.spaces.Discrete(2),  # is or isn't dangerous,
            "nearby_obj_was_dropped": gym.spaces.Discrete(2),  # whether it has already been dropped in the zone
            # TODO: handle other agents nearby
            # **{a: gym.spaces.utils.flatten_space(self.agent_info_space) for a in self.agent_ids},
            **{"A": gym.spaces.utils.flatten_space(self.agent_info_space) for a in self.agent_ids},
            **{o: gym.spaces.utils.flatten_space(self.object_info_space) for o in self.object_ids}
        })

        # SB3 doesn't like nested obs space so we flatten it
        self.observation_space = gym.spaces.Dict({
            a: gym.spaces.utils.flatten_space(self.unflatten_observation_space)
            for a in self.agent_ids
        })
        self.action_space = gym.spaces.Discrete(len(self.Actions))

        self._orientation = self.Actions.up
        self.object_carried = defaultdict(lambda: None)

        self.num_steps = 0

        self.goal_coords = self.postions_by_type("ZZ")["ZZ"]

    def _get_obs(self):
        sol = {}
        for i in range(self.num_agents):
            a_name = self.id_to_name(i)

            nearby_obj_weight, nearby_obj_danger, nearby_object_was_dropped = self.nearby_obj_info(a_name)
            sol.update({
                a_name: gym.spaces.utils.flatten(self.unflatten_observation_space, {
                    "agent_id": self.name_to_id(a_name),
                    "agent_strength": self.strength(a_name),
                    "nearby_obj_weight": nearby_obj_weight,
                    "nearby_obj_danger": nearby_obj_danger,
                    "nearby_obj_was_dropped": nearby_object_was_dropped,
                    # TODO: handle help
                    # **{a: gym.spaces.flatten(self.agent_info_space, self.agent_infos[a]) for a in self.agent_ids},
                    **{"A": gym.spaces.flatten(self.agent_info_space, self.agent_infos[a_name])},
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
                if k in self.agent_ids:
                    sol[a].update({k: gym.spaces.utils.unflatten(self.agent_info_space, o)})
                elif k in self.object_ids:
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
                (0, 0, 0),
                (pos + 0.5) * self.pix_square_size,
                self.pix_square_size / 3,
            )
            # bug on mac
            # font = pygame.font.SysFont(None, 30)
            # img = font.render(label, True, self.COLOR_MAPPING["K"])
            # canvas.blit(img, self.pix_square_size * (pos + 0.25))

    def _can_enter(self, new_pos):
        walls = self.postions_by_type("W").items()
        agents = self.postions_by_type("A").items()
        s_items = self.postions_by_type("S").items()
        d_items = self.postions_by_type("D").items()
        for label, positions in itertools.chain(walls, agents, s_items, d_items):
            if tuple(new_pos) in positions:
                return False
        return True

    def reset(self, seed=None, options=None):
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
        obs, info = super().reset(seed=seed, options=options)
        return obs, info

    def step(self, actions):
        reward = -0.01

        for aid, action in actions.items():
            # self.cancel_help_request(aid)

            act = self.Actions(action)
            if act == self.Actions.pick_up:
                self.pick_up_step(aid)
            elif act == self.Actions.drop:
                reward = self.drop_step(aid)
            # elif act == self.Actions.ask_for_help:
            #     self.ask_for_help(aid)
            elif act < self.Actions.pick_up:
                if act != self.Actions.none:
                    self._orientation = act
                direction = self._act_to_direction(act)
                new_agent_pos = np.clip(
                    self.agent_infos[aid]["pos"] + direction, 0, self.size - 1
                )
                if self._can_enter(new_agent_pos):
                    old_pos = tuple(self.agent_infos[aid]["pos"])
                    self.positions[old_pos].remove(aid)
                    self.positions[tuple(new_agent_pos)].append(aid)
                    self.agent_infos[aid]["pos"] = np.array(new_agent_pos)
                    if self.object_carried[aid]:
                        self.positions[old_pos].remove(self.object_carried[aid])
                        self.positions[tuple(new_agent_pos)].append(self.object_carried[aid])

        # All dangerous objects have been moved to the zone
        terminated = (
            # a agent can be carrying a green object
            # not any(self.object_carried.values()) and
            all(d in self.dropped_in_safe_zone for d in self.postions_by_type("D").keys())
        )

        observation = self._get_obs()
        info = self._get_info()
        if self.render_mode == "human":
            self._render_frame()

        self.num_steps += 1
        reward /= 10

        return observation, reward, terminated, self.num_steps >= self.max_steps, info

    def pick_up_step(self, a_name: str):
        curr_pos = self.agent_infos[a_name]["pos"]
        neighbour_pos = [cell for cell in adjacent_cells_iterator(curr_pos, self.size) if
                         any(elem[0] in ["D", "S"] and self._get_object_weight(elem) <= self.strength(a_name) for elem
                             in self.positions[cell]) and
                         all(elem not in self.agent_infos.keys() for elem in self.positions[cell])
                         ]
        if len(neighbour_pos) > 0:
            # Always check directions in a different order.
            # This will result in a random object selection if there is more than 1
            # in the vicinity.
            chosen_object_pos = random.choice(neighbour_pos)
            relative_dir = chosen_object_pos - curr_pos
            self._orientation = self._dir_to_action(tuple(relative_dir))

            self.positions[tuple(curr_pos)].remove(a_name)
            self.object_carried[a_name] = self._pick_by(self.positions[tuple(chosen_object_pos)],
                                                        lambda x: x[0] in ["D", "S"])
            self.positions[tuple(chosen_object_pos)].append(a_name)
            self.agent_infos[a_name]["pos"] = np.array(chosen_object_pos)

    @staticmethod
    def _get_object_weight(obj_id):
        # return int(obj_id[-1])
        return int(obj_id.split("_")[-1])

    @staticmethod
    def _pick_by(l: List[str], cond):
        for x in l:
            if cond(x):
                return x
        return None

    def drop_step(self, a_name: str):
        # handle orientation
        # drop in the cell that should be adjacent to object's orientation
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
        strength = 1
        for adj_cell in adjacent_cells_iterator(curr_pos, self.size, eight_dir=True):
            for agent in self.agent_infos.keys():
                if agent in self.positions[adj_cell]:
                    strength += 1

        return strength

    def nearby_obj_info(self, a_name: str):
        curr_pos = tuple(self.agent_infos[a_name]["pos"])
        for adj_cell in adjacent_cells_iterator(curr_pos, self.size):
            if adj_cell in self.goal_coords:
                continue
            for entity in self.positions[adj_cell]:
                if entity[0] in ['S', 'D'] and entity not in self.object_carried.values():
                    weight = self._get_object_weight(entity)
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
            for position in positions:
                held_by = self._pick_by(self.positions[position], lambda x: x in self.agent_infos.keys())
                carried_by = np.zeros(len(self.agent_infos.keys()))
                if held_by:
                    carried_by[self.name_to_id(held_by)] = 1

                sol[label] = {
                    "pos": np.array(position),
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
        return np.array(
            self.postions_by_type(a_name, pos={k: [v] for k, v in self.original_positions.items()})[a_name][0])

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


class AICollabLabellingFunctionWrapper(LabelingFunctionWrapper):

    def get_labels(self, obs: dict = None, prev_obs: dict = None):
        """Returns a modified observation."""
        labels = []

        for aid, flatten_obs in obs.items():

            unwrapped_obs = gym.spaces.unflatten(self.env.unflatten_observation_space, flatten_obs)
            label_agent_id = aid.lower()
            agent_ix = unwrapped_obs["agent_id"]

            if any(
                    obj_info["carried_by"][agent_ix] > 0
                    for obj_info in self.env.unwrapped.object_infos.values()
            ):
                labels.append(f'{label_agent_id}l')
            elif unwrapped_obs["nearby_obj_weight"] > 0 and unwrapped_obs["nearby_obj_danger"] == 1:
                labels.append(f'{label_agent_id}d')

            # TODO: handle other agents nearby
            agent_info = gym.spaces.unflatten(self.agent_info_space, unwrapped_obs["A"])
            pos = tuple(agent_info["pos"])
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

        unwrapped_obs = obs['A']
        agent_id = unwrapped_obs["agent_id"]
        agent_name = self.env.unwrapped.id_to_name(agent_id)

        if unwrapped_obs["nearby_obj_weight"] > 0 and unwrapped_obs["nearby_obj_danger"] == 1:
            labels.append('a1d')

        if any(
                obj_info["carried_by"][agent_id]
                for obj_info in tuple(v for k, v in unwrapped_obs.items() if k in self.env.unwrapped.object_ids)
        ) > 0:
            labels.append('a1l')

        pos = unwrapped_obs[agent_name]["pos"]
        if pos in self.env.unwrapped.goal_coords:
            labels.append('a1z')

        return labels
