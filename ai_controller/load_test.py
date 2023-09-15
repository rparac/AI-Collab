import datetime as dt
import os

import gymnasium
from gymnasium.wrappers import RecordEpisodeStatistics
from torch.utils.tensorboard import SummaryWriter

from gym_collab.envs import AICollabEnv
from gym_collab.envs.ai_collab_wrapper import MARLWrapper
from gym_collab.envs.helper_wrappers import AgentNameWrapper, EnvComposition
from rm_marl.agent import RewardMachineAgent
from rm_marl.algo import QRM
from rm_marl.envs.ai_collab import AICollab, AICollabLabellingFunctionWrapper, FIX_NEW_PICKUP, FIX_ORIENTATION, \
    OriginalAICollabLabellingFunctionWrapper
from rm_marl.envs.wrappers import SingleAgentEnvWrapper, AutomataWrapper, RewardMachineWrapper, \
    RandomLabelingFunctionWrapper, RandomLabelingConfig, RMProgressReward
from rm_marl.reward_machine import RewardMachine
from rm_marl.trainer import Trainer

BASE_PATH = os.path.join(os.path.dirname(__file__), "data/ai-collab")
save_path = os.path.join(os.path.dirname(__file__), "logs/ai-collab-exp1/heavy_object_carried")

CENTRALIZED_TRAINING = False
MAX_STEPS = 5000

# suffix = ""
suffix = "_1h_1l"

ENV_PATH = os.path.join(BASE_PATH, f"small_env_2a{suffix}.txt")
RM_PATHS = {
    "A1": os.path.join(BASE_PATH, f"rm_agent_1{suffix}.txt"),
    "A2": os.path.join(BASE_PATH, f"rm_agent_2{suffix}.txt"),
    "team": os.path.join(BASE_PATH, f"rm_team{suffix}.txt")
}


def create_ai_collab_env(client_number: int) -> AICollabEnv:
    ai_collab_env = gymnasium.make('gym_collab/AICollabWorld-v0', use_occupancy=True, view_radius=50, skip_frames=10,
                                   client_number=client_number,
                                   host='0.0.0.0', port=8080, address="https://localhost:5683", cert_file=None,
                                   key_file=None)
    return ai_collab_env  # type: ignore[override]

def _create_env(env, aid):
    env1 = AgentNameWrapper(env)
    env1 = OriginalAICollabLabellingFunctionWrapper(env1)
    env1 = AutomataWrapper(
        env1,
        RewardMachine.load_from_file(RM_PATHS[aid]),
        label_mode=AutomataWrapper.LabelMode.RM,
        termination_mode=AutomataWrapper.TerminationMode.ENV
    )
    env1 = RecordEpisodeStatistics(env1)

    return env1


env1 = MARLWrapper(create_ai_collab_env(client_number=1))
env2 = MARLWrapper(create_ai_collab_env(client_number=2))


# env1 = _create_env(env1, "A1")
# env2 = _create_env(env2, "A2")

trainer_run_config = {
    "training": False,
    "total_episodes": 1,
    "log_freq": 1,
    "log_dir": os.path.join(os.path.dirname(__file__), "logs"),
    "testing_freq": 1,
    "greedy": True,
    "synchronize": True,
    "counterfactual_update": True,
    "recording_freq": 1000000000000000, # render not implemented
    "seed": 123,
    "name": "ai-collab-ma-exp1-test",

    "show_q_function_diff": False,
    "q_true": None,
    "centralized": CENTRALIZED_TRAINING,
    "fix_new_pickup": FIX_NEW_PICKUP,
    "fix_orientation": FIX_ORIENTATION,
    "max_steps": MAX_STEPS,
    "suffix": suffix,
    "env_parallel_reset": True,
}

trainer_loaded = Trainer.load(save_path)

env_dict = {
    "A1": env1,
    "A2": env2,
}

shared_env = EnvComposition(env_dict)
shared_env = OriginalAICollabLabellingFunctionWrapper(shared_env)
shared_env = AutomataWrapper(
    shared_env,
    RewardMachine.load_from_file(RM_PATHS["team"]),
    label_mode=AutomataWrapper.LabelMode.RM,
    termination_mode=AutomataWrapper.TerminationMode.ENV
)
shared_env = RecordEpisodeStatistics(shared_env)
shared_env_dict = {
    "E": shared_env
}

log_dir = os.path.join(
    trainer_run_config["log_dir"],
    trainer_run_config["name"],
    dt.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
)
logger = SummaryWriter(log_dir)
trainer_loaded._run(shared_env_dict, trainer_run_config, logger)
