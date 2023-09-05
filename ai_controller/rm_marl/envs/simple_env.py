import itertools
import random

import gym
import numpy as np
import pygame

from rm_marl.envs.base_grid import BaseGridEnv
from rm_marl.envs.wrappers import LabelingFunctionWrapper


class SimpleEnv(BaseGridEnv):
    def __init__(self, render_mode=None, size=6, positions=None, file=None, initial_pos_seed=0, max_steps=250):
        super().__init__(render_mode, size, positions, file)
        random.seed(initial_pos_seed)
        self._b1_pressed = False
        self.initial_pos_seed = initial_pos_seed
        self.current_step = 0
        self.max_steps = max_steps

        self.unflatten_observation_space = gym.spaces.Dict({
            id_: gym.spaces.Dict({
                "location": gym.spaces.Box(0, size - 1, shape=(2,), dtype=int),
                **{id_: gym.spaces.Box(0, size - 1, shape=(2,), dtype=int)
                   for id_ in self.button_locations.keys()}
            })
            for id_ in self.agent_locations.keys()
        })
        #SB3 doesn't like nested obs space so we flatten it
        self.observation_space = gym.spaces.Dict({
            k: gym.spaces.utils.flatten_space(v)
            for k, v in self.unflatten_observation_space.items()
        })

    def _get_obs(self):
        return {
            agent_id: gym.spaces.utils.flatten(
                self.unflatten_observation_space[agent_id], 
                {"location": loc, **self.button_locations}
            )
            for agent_id, loc in self.agent_locations.items()
        }

    def _get_info(self):
        return {}

    @property
    def button_locations(self):
        return {
            l: np.array(p) for p, ls in self.positions.items() for l in ls if l.startswith("B")
        }

    def _draw_component(self, label, pos, canvas):
        if label[0] == "W":
            pygame.draw.rect(
                canvas,
                self.COLOR_MAPPING[label[-1]],
                pygame.Rect(
                    self.pix_square_size * pos,
                    (self.pix_square_size, self.pix_square_size),
                ),
            )
        elif label[0] == "B":
            pygame.draw.circle(
                canvas,
                self.COLOR_MAPPING[label[-1]],
                (pos + 0.5) * self.pix_square_size,
                self.pix_square_size / 3,
            )
        elif label[0] == "A":
            font = pygame.font.SysFont(None, 50)
            img = font.render(label, True, self.COLOR_MAPPING["K"])
            canvas.blit(img, self.pix_square_size * (pos + 0.25))

    def _can_enter(self, new_pos):
        for label, positions in self.postions_by_type("W").items():
            if tuple(new_pos) in positions:
                return False
        return True

    def _clear_pos(self):
        self.original_positions = {
            k: v
            for k, v in self.original_positions.items() if v != "BY" and v != "BR"
        }

    def _sample_tuple(self):
        row = random.randint(1, 5)
        col = random.randint(1, 5)
        # Must be different from the agent location
        while row == 1 and col == 1:
            row = random.randint(1, 5)
            col = random.randint(1, 5)
        return row, col

    def reset(self, seed=None, options=None):
        # self._clear_pos()
        # by_pos = self._sample_tuple()
        # br_pos = self._sample_tuple()
        # while br_pos == by_pos:
        #     br_pos = self._sample_tuple()
        # self.original_positions[by_pos] = 'BY'
        # self.original_positions[br_pos] = 'BR'
        self.current_step = 0

        obs, info = super().reset(seed=seed)
        self._b1_pressed = False
        return obs, info

    def _is_button_pressed(self, button_label):
        for agent_loc in self.agent_locations.values():
            for pos in self.postions_by_type(button_label).values():
                if tuple(agent_loc) in pos:
                    return True
        return False

    def step(self, actions):
        self.current_step += 1
        for aid, action in actions.items():
            direction = self._action_to_direction[self.Actions(action)]
            new_agent_pos = np.clip(
                self.agent_locations[aid] + direction, 0, self.size - 1
            )
            if self._can_enter(new_agent_pos):
                self.positions[tuple(self.agent_locations[aid])].remove(aid)
                self.positions[tuple(new_agent_pos)].append(aid)

        terminated = False
        if self._is_button_pressed("BY"):
            self._b1_pressed = True
        if self._b1_pressed and self._is_button_pressed("BR"):
            terminated = True
        reward = 1 if terminated else -0.01  # Binary sparse rewards

        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, reward, terminated, self.current_step > self.max_steps, info


class SimpleEnvLabellingFunctionWrapper(LabelingFunctionWrapper):
    def get_labels(self, obs: dict = None, prev_obs: dict = None):
        """Returns a modified observation."""
        agent_locations = obs or self.agent_locations
        prev_agent_locations = prev_obs or self.prev_agent_locations
        labels = []

        by_positions = self.postions_by_type("B").get("BY", [])
        br_positions = self.postions_by_type("B").get("BR", [])

        if self._agent_has_moved_to(
                "A1", prev_agent_locations, agent_locations, by_positions
        ):
            labels.append("by")

        if self._agent_has_moved_to(
                "A1", prev_agent_locations, agent_locations, br_positions
        ):
            labels.append("br")

        return labels

    @staticmethod
    def _agent_has_moved_to(agent, from_loc, to_loc, positions):
        return (
                tuple(from_loc[agent]) not in positions
                and tuple(to_loc[agent]) in positions
        )
