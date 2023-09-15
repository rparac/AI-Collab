from typing import Iterator, Tuple

import numpy as np
from gymnasium.core import ActType

from .action import Action


def adjacent_cells_iterator(curr_pos: (int, int), grid_size: (int, int)) -> Iterator[Tuple[int, int]]:
    pos_i, pos_j = curr_pos
    dirs_i = [0, 1, 0, -1]
    dirs_j = [-1, 0, 1, 0]
    for dir_i, dir_j in zip(dirs_i, dirs_j):
        next_i, next_j = pos_i + dir_i, pos_j + dir_j
        if _inbounds(grid_size, next_i, next_j):
            yield next_i, next_j


def _inbounds(size: (int, int), i: int, j: int) -> bool:
    size_i, size_j = size
    return 0 <= i < size_i and 0 <= j < size_j


# Wraps action code into an action
# Use when there is only action code fully determines the
# action, e.g. in move, danger_sensing, grab
def wrap_action_enum(action_code: Action) -> ActType:
    return {
        "action": action_code.value,
        "item": 0,
        "message": "empty",
        "num_cells_move": 1,
        "robot": 0,
    }


def create_check_item_action(item_idx: int) -> ActType:
    return {
        "action": Action.check_item.value,
        "item": item_idx,
        "message": "empty",
        "num_cells_move": 1,
        "robot": 0,
    }


def _find_curr_agent_location(occupancy_map: np.ndarray) -> (int, int):
    agent_code = 5
    return tuple(np.transpose(np.where(occupancy_map == agent_code)[::-1]).squeeze())

def find_object_held_location(occupancy_map: np.ndarray) -> (int, int):
    object_held_code = 4
    return tuple(np.transpose(np.where(occupancy_map == object_held_code)[::-1]).squeeze())
