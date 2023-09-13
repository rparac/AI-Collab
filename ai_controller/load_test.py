import os

import gymnasium
from gymnasium.wrappers import RecordEpisodeStatistics

from gym_collab.envs import AICollabEnv
from gym_collab.envs.ai_collab_wrapper import MARLWrapper
from gym_collab.envs.helper_wrappers import AgentNameWrapper
from rm_marl.agent import RewardMachineAgent
from rm_marl.algo import QRM
from rm_marl.envs.ai_collab import AICollabLabellingFunctionWrapper, OriginalAICollabLabellingFunctionWrapper
from rm_marl.envs.wrappers import SingleAgentEnvWrapper, AutomataWrapper
from rm_marl.reward_machine import RewardMachine
from rm_marl.trainer import Trainer

BASE_PATH = os.path.dirname(__file__)
save_path = os.path.join(BASE_PATH, "logs/ai-collab-exp1/sim2sim_no_pos")

rm_path = {
    "A1": os.path.join(BASE_PATH, "data/ai-collab/rm_agent_1.txt"),
    "A2": os.path.join(BASE_PATH, "data/ai-collab/rm_agent_2.txt"),
}

def _create_env_and_agent(env, aid):
    env1 = AgentNameWrapper(env)
    env1 = SingleAgentEnvWrapper(env1, aid)
    env1 = OriginalAICollabLabellingFunctionWrapper(env1)
    env1 = AutomataWrapper(
        env1,
        RewardMachine.load_from_file(rm_path[aid]),
        label_mode=AutomataWrapper.LabelMode.ALL,
        termination_mode=AutomataWrapper.TerminationMode.ENV
    )
    env1 = RecordEpisodeStatistics(env1)

    return env1

def create_ai_collab_env(client_number: int) -> AICollabEnv:
    ai_collab_env = gymnasium.make('gym_collab/AICollabWorld-v0', use_occupancy=True, view_radius=50, skip_frames=10,
                             client_number=client_number,
                             host='0.0.0.0', port=8080, address="https://localhost:5683", cert_file=None, key_file=None)
    return ai_collab_env  # type: ignore[override]

env = MARLWrapper(create_ai_collab_env(client_number=1))

env1 = _create_env_and_agent(env, "A1")
env_dict = {"E1": env1}
# env1.unwrapped.render_mode = "human"

trainer_loaded = Trainer.load(save_path)
trainer_loaded.envs = env_dict
trainer_loaded.testing_envs = env_dict

print(trainer_loaded)

trainer_loaded.run({
    "training": False,
    "log_freq": 1,
    "recording_freq": 10000000000000, # 1, Render not implemented
    "total_episodes": 1,
    "greedy": True,
    "seed": 123,
    "synchronize": True,
    "log_dir": os.path.join(BASE_PATH, "logs"),
    "name": "ai-collab-exp1-eval",
    "show_q_function_diff": False,
    "q_true": None,
})
