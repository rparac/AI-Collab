import os

import gymnasium as gym
from gymnasium.wrappers import RecordEpisodeStatistics

from gym_collab.envs import AICollabEnv
from gym_collab.envs.ai_collab_wrapper import MARLWrapper
from gym_collab.envs.helper_wrappers import AgentNameWrapper
from rm_marl.agent import RewardMachineAgent
from rm_marl.algo import QRM
from rm_marl.envs.wrappers import SingleAgentEnvWrapper, RewardMachineWrapper
from rm_marl.envs.ai_collab import AICollabLabellingFunctionWrapper
from rm_marl.reward_machine import RewardMachine
from rm_marl.trainer import Trainer

BASE_PATH = os.path.join(os.path.dirname(__file__), "data/ai-collab")


def create_ai_collab_env(client_number: int) -> AICollabEnv:
    ai_collab_env = gym.make('gym_collab/AICollabWorld-v0', use_occupancy=True, view_radius=50, skip_frames=10,
                             client_number=client_number,
                             host='0.0.0.0', port=8080, address="https://localhost:5683", cert_file=None, key_file=None)

    return ai_collab_env  # type: ignore[override]


rm_path = os.path.join(BASE_PATH, "rm_agent_1.txt")

agent_key = "A"
trainer_run_config = {
    "training": True,
    "total_episodes": 10,
    "log_freq": 1,
    "log_dir": os.path.join(os.path.dirname(__file__), "logs"),
    "testing_freq": 1,
    "greedy": False,
    "synchronize": True,
    "counterfactual_update": True,
    "recording_freq": 100000,  # avoid triggering it because Julian doesn't support render
    "seed": 123,
    "name": "ai-collab-exp1",
    "show_q_function_diff": False,
    "q_true": None,
}

rm = RewardMachine.load_from_file(rm_path)

collab_env = create_ai_collab_env(client_number=1)
env = MARLWrapper(collab_env)

# QRM must work with Discrete action space, so we pass it through here
agent_dict = {
    agent_key: RewardMachineAgent(agent_key, rm, algo_cls=QRM, algo_kws={"action_space": env.action_space})
}

env = AgentNameWrapper(env)
env = SingleAgentEnvWrapper(env, agent_key)
env = AICollabLabellingFunctionWrapper(env)
env = RewardMachineWrapper(
    env,
    rm,
    label_mode=RewardMachineWrapper.LabelMode.ALL,
)
env = RecordEpisodeStatistics(env)
env_dict = {"E1": env}

trainer = Trainer(env_dict, env_dict, agent_dict)
trainer.run(trainer_run_config)

print("Done")
