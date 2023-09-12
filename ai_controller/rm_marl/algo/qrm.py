from collections import defaultdict
from typing import Optional

import gymnasium.spaces
import numpy as np
from gymnasium.utils import seeding

from ._base import Algo


class QRM(Algo):
    _np_random: Optional[np.random.Generator] = None

    def __init__(
            self,
            action_space: "gymnasium.spaces.Space" = None,
            initial_epsilon: float = 0.0,
            final_epsilon: float = 0.0,
            exploration_steps: int = 1000,
            temperature: float = 50.0,
            alpha: float = 0.8,
            gamma: float = 0.9,
            seed: int = 123,
    ):
        assert isinstance(action_space, gymnasium.spaces.Discrete)
        self.action_space = action_space
        self.initial_epsilon = initial_epsilon
        self.final_epsilon = final_epsilon
        self.exploration_steps = exploration_steps
        self.temperature = temperature
        self.alpha = alpha
        self.gamma = gamma
        self._seed = seed

        self.q = defaultdict(self._q_sa_constructor)

        self.step_count = 0

        self.reset(seed=seed)

    def _q_a_constructor(self):
        return np.zeros((self.action_space.n))

    def _q_sa_constructor(self):
        return defaultdict(self._q_a_constructor)

    def reset(self, seed: Optional[int] = None):
        seed = self._seed if seed is None else seed
        if seed is not None:
            self._np_random, seed = seeding.np_random(seed)

        self.q.clear()

    @staticmethod
    def _recursive_to_hashable_state_(state):
        if type(state) == np.ndarray:
            return tuple(state)
        if type(state) in [list, tuple]:
            return tuple(QRM._recursive_to_hashable_state_(val) for val in state)
        if type(state) in [int, float, bool]:
            return state

        return tuple(
            sorted({k: QRM._recursive_to_hashable_state_(v) for k, v in state.items()}.items(), key=lambda i: i[0])
        )

    @staticmethod
    def _to_hashable_state_(state):
        # return QRM._recursive_to_hashable_state_(state)
        return tuple(
            sorted({k: tuple(v) for k, v in state.items()}.items(), key=lambda i: i[0])
        )

    def learn(self, state, u, action, reward, done, next_state, next_u):
        next_q = np.amax(self.q[next_u][self._to_hashable_state_(next_state)])
        target_q = reward + (1 - int(done)) * self.gamma * next_q

        current_q = self.q[u][self._to_hashable_state_(state)][action]

        loss = np.abs(current_q - target_q)

        # Bellman update
        self.q[u][self._to_hashable_state_(state)][action] = (
                                                                     1 - self.alpha
                                                             ) * current_q + self.alpha * target_q

        return loss

    # TODO: implement self.train()
    def action(self, state, u, greedy: bool = False, training: bool = True):
        if training:
            self.step_count += 1

        if training and self._np_random.random() < self.epsilon:
            action = self._np_random.choice(range(self.action_space.n))
        elif not greedy:
            pr_sum = np.sum(
                np.exp(self.q[u][self._to_hashable_state_(state)] * self.temperature)
            )
            pr = (
                    np.exp(self.q[u][self._to_hashable_state_(state)] * self.temperature)
                    / pr_sum
            )

            # If any q-values are so large that the softmax function returns infinity,
            # make the corresponding actions equally likely
            if any(np.isnan(pr)):
                print("BOLTZMANN CONSTANT TOO LARGE IN ACTION-SELECTION SOFTMAX.")
                temp = np.array(np.isnan(pr), dtype=float)
                pr = temp / np.sum(temp)

            cdf = np.insert(np.cumsum(pr), 0, 0)

            randn = self._np_random.random()
            for a in range(self.action_space.n):
                if randn >= cdf[a] and randn <= cdf[a + 1]:
                    action = a
                    break
        else:
            best_actions = np.where(
                self.q[u][self._to_hashable_state_(state)]
                == np.max(self.q[u][self._to_hashable_state_(state)])
            )[0]
            action = self._np_random.choice(best_actions)

        return action

    @property
    def epsilon(self):
        if self.step_count >= self.exploration_steps:
            return self.final_epsilon
        else:
            slope = (self.final_epsilon - self.initial_epsilon) / self.exploration_steps
            return self.initial_epsilon + self.step_count * slope
