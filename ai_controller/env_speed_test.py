"""
AI controller whose sole purpose is to move left 3 spaces and then move down 4 spaces.
"""
import random
import time

import gymnasium as gym

from gym_collab.envs import AICollabEnv
from gym_collab.envs.action import Action
from gym_collab.envs.utils import wrap_action_enum

random.seed(0)


def int_to_enum(code: int):
    for act in Action:
        if act.value == code:
            return act
    raise ValueError(f"The code: {code} does not exist in enum")


def create_ai_collab_env(client_number: int) -> AICollabEnv:
    ai_collab_env = gym.make('gym_collab/AICollabWorld-v0', use_occupancy=True, view_radius=50, skip_frames=10,
                             client_number=client_number,
                             host='0.0.0.0', port=8080, address="https://localhost:5683", cert_file=None, key_file=None)
    return ai_collab_env  # type: ignore[override]


env = create_ai_collab_env(client_number=1)
observation, info = env.reset()

start_time = time.time()
completed_steps = 0
steps_limit = 50
while completed_steps < steps_limit:
    done = False
    action_issued = [False, False]
    last_action = [0, 0]

    move_act_code = random.randint(0, 3)
    action = wrap_action_enum(int_to_enum(move_act_code))
    while not done and completed_steps < steps_limit:

        # Make sure to issue concurrent actions but not of the same type. Else, wait.
        if action["action"] < Action.danger_sensing.value and not action_issued[0]:
            action_issued[0] = True
            last_action[0] = action["action"]
        elif action["action"] != Action.wait.value and action["action"] >= Action.danger_sensing.value and not \
        action_issued[1]:
            last_action_arguments = [action["item"], action["robot"], action["message"]]
            action_issued[1] = True
            last_action[1] = action["action"]
        else:
            action["action"] = Action.wait.value

        next_observation, reward, terminated, truncated, info = env.step(action)

        # When any action has completed
        if next_observation and any(next_observation['action_status']):
            completed_steps += 1
            print(f"Completed step {completed_steps}")

            if any(next_observation['action_status'][:2]):
                action_issued[0] = False
            if any(next_observation['action_status'][2:4]):
                action_issued[1] = False
            move_act_code = random.randint(0, 3)
            action = wrap_action_enum(int_to_enum(move_act_code))

        if terminated or truncated:
            done = True

print(f"The simulator took {time.time() - start_time} seconds to execute {completed_steps} move commands")

print("Closing environment")
env.close()
