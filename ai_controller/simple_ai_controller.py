"""
AI controller whose sole purpose is to move left 3 spaces and then move down 4 spaces.
"""
from typing import Type

import gym_collab
import time

import gymnasium as gym

from gym_collab.envs import AICollabEnv
from gym_collab.envs.ai_collab_wrapper import MARLWrapper

input_to_dir = {
    "w": 1,  # up
    "a": 4,  # left
    "s": 3,  # down
    "d": 2,  # "right"
}


def create_ai_collab_env(client_number: int) -> AICollabEnv:
    ai_collab_env = gym.make('gym_collab/AICollabWorld-v0', use_occupancy=True, view_radius=50, skip_frames=10,
                             client_number=client_number,
                             host='0.0.0.0', port=8080, address="https://localhost:5683", cert_file=None, key_file=None)
    return ai_collab_env  # type: ignore[override]


env1 = MARLWrapper(create_ai_collab_env(client_number=1))
# env2 = MARLWrapper(create_ai_collab_env(client_number=2))
observation, info = env1.reset()
# observation2, info2 = env2.reset()

while True:
    inp = input()
    xs = inp.split()
    x = xs[0]
    # env = env2 if len(xs) > 1 else env1
    # item = int(xs[1]) if len(xs) > 1 else 0
    num_cells = int(xs[2]) if len(xs) > 2 else 1
    if x in input_to_dir:
        dir = input_to_dir[x]
    else:
        dir = int(x)

    print(f"I have turned by pressing the {x} key")
    next_observation, reward, terminated, truncated, info = env1.step(dir)

    # while not any(next_observation["action_status"][:2]):
    #     action["action"] = 26  # wait
    #     next_observation, reward, terminated, truncated, info = env.step(action)
    #     # print(next_observation)
    # success = next_observation["action_status"][0] > 0
    success = False
    print(f"There was success: {success}")
