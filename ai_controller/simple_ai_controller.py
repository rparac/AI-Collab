"""
AI controller whose sole purpose is to move left 3 spaces and then move down 4 spaces.
"""
from typing import Type

import gym_collab
import time

import gymnasium as gym

from gym_collab.envs import AICollabEnv
from gym_collab.envs.grid_world_env import GridWorldEnv, SimpleObservations

input_to_dir = {
    "w": 0,  # up
    "a": 2,  # left
    "s": 1,  # down
    "d": 3,  # "right"
}


def create_ai_collab_env() -> AICollabEnv:
    ai_collab_env = gym.make('gym_collab/AICollabWorld-v0', use_occupancy=True, view_radius=50, skip_frames=10,
                             client_number=1,
                             host='0.0.0.0', port=8080, address="https://localhost:5683", cert_file=None, key_file=None)
    return ai_collab_env  # type: ignore[override]


env = GridWorldEnv(create_ai_collab_env())
env = SimpleObservations(env)

observation, info = env.reset()

while True:
    x = input()
    if x in input_to_dir:
        dir = input_to_dir[x]
    else:
        dir = int(x)
    action = {
        "action": dir,
        "item": 0,
        "message": "empty",
        "num_cells_move": 1,
        "robot": 0,
    }

    print(f"I have turned by pressing the {x} key")
    next_observation, reward, terminated, truncated, info = env.step(action)

    # while not any(next_observation["action_status"][:2]):
    #     action["action"] = 26  # wait
    #     next_observation, reward, terminated, truncated, info = env.step(action)
    #     # print(next_observation)
    success = next_observation["action_status"][0] > 0
    print(f"There was success: {success}")
