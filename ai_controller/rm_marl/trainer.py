import copy
import datetime as dt
import os
import warnings
from collections import defaultdict
from typing import Dict
import gymnasium

# import joblib
import cloudpickle
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from .algo import QRM
from .algo.dqn import DQN


class Trainer:
    def __init__(self, local_envs: dict, shared_envs: dict, agents: dict):
        self.envs = local_envs or shared_envs
        self.testing_envs = shared_envs
        self.agents = agents

        self.total_steps = 0
        self.test_episodes = 0

    def run(self, run_config: dict):
        log_dir = os.path.join(
            run_config["log_dir"],
            run_config["name"],
            dt.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
        )
        logger = SummaryWriter(log_dir)

        try:
            self._run(self.envs, run_config, logger)
        except KeyboardInterrupt:
            pass

        _ = [e.close() for e in self.envs.values()]
        logger.close()
        if run_config["training"]:
            self.save(log_dir)

    def _run(self, envs: dict, run_config: dict, logger: SummaryWriter):
        base_seed = run_config["seed"]

        steps = defaultdict(list)
        losses = defaultdict(list)
        rewards = defaultdict(list)

        _ = [a.set_log_folder(os.path.join(logger.log_dir, aid)) for aid, a in self.agents.items()]

        for episode in tqdm(range(1, 1 + run_config["total_episodes"])):
            if not run_config["training"]:
                self.test_episodes += 1

            episode_losses = defaultdict(list)
            episode_frames = defaultdict(list)

            seed = base_seed + episode - 1

            # reset and initial setup
            dones = {env_id: False for env_id in envs.keys()}
            env_agents = {}

            _ = [a.reset(seed=seed) for a in self.agents.values()]

            obs, infos, env_agents, shared_events = {}, {}, {}, {}
            for env_id, env in envs.items():
                o, i = env.reset(seed=seed)
                obs[env_id] = o
                infos[env_id] = i

                env_agents[env_id] = {
                    aid: a for aid, a in self.agents.items() if aid in obs[env_id]
                }
                shared_events[env_id] = self._get_shared_events(env_agents[env_id])

            steps_count = 0
            while not all(dones.values()):
                steps_count += 1

                if run_config["training"]:
                    self.total_steps += self.env.unwrapped.extra1

                for env_id, env in envs.items():

                    if episode % run_config["recording_freq"] == 0:
                        episode_frames[env_id].append(env.render())

                    if dones[env_id]:
                        continue

                    actions = {aid: a.action(
                        self._project_obs(obs[env_id], a, aid),
                        greedy=run_config.get("greedy", True)
                        if not run_config["training"]
                        else False,
                        # training=run_config["training"],
                    )
                        for aid, a in env_agents[env_id].items()
                    }
                    next_obs, reward, terminated, truncated, info = env.step(actions)

                    done = terminated or truncated

                    labels = info["labels"]
                    agent_labels = {}
                    _ = [agent_labels.update(self._project_labels(labels, a, aid)) for aid, a in
                         env_agents[env_id].items()]

                    if run_config["synchronize"]:
                        synchronized_labels = self._synchronize(shared_events[env_id], agent_labels)
                        assert all(agent_labels[aid] == synchronized_labels[aid] for aid in env_agents[
                            env_id].keys()), f"Not synchronized!! {agent_labels}, {synchronized_labels}"
                    else:
                        synchronized_labels = agent_labels

                    # update the agent's RM and Q-functions
                    agent_loss = []
                    interrupt_episode = terminated or truncated
                    for aid, a in env_agents[env_id].items():
                        current_u = a.u
                        loss, interrupt = a.update_agent(
                            self._project_obs(obs[env_id], a, aid),
                            actions[aid],
                            reward,
                            terminated,
                            truncated,
                            self._project_obs(next_obs, a, aid),
                            synchronized_labels[aid],
                            learning=run_config["training"],
                        )

                        if run_config["training"]:
                            if interrupt:
                                interrupt_episode = True
                                info["episode"] = {
                                    "l": env.get_wrapper_attr('episode_lengths'),
                                    "r": env.get_wrapper_attr('episode_returns')
                                }
                                break

                            agent_loss.append(loss)
                            if run_config["counterfactual_update"] and not interrupt_episode:
                                self._counterfactual_update(
                                    env,
                                    a,
                                    self._project_obs(obs[env_id], a, aid),
                                    current_u,
                                    actions[aid],
                                    done,
                                    self._project_obs(next_obs, a, aid),
                                )

                    obs[env_id] = next_obs
                    infos[env_id] = info
                    dones[env_id] = interrupt_episode

                    if agent_loss:
                        with warnings.catch_warnings():
                            warnings.simplefilter("ignore", category=RuntimeWarning)
                            mean_loss = np.nanmean(agent_loss)
                        if not np.isnan(mean_loss):
                            episode_losses[env_id].append(mean_loss)

            # track metrics and log them in TB
            for env_id in envs.keys():
                prefix = "training" if run_config["training"] else "eval"

                steps[env_id].append(infos[env_id]["episode"]["l"])
                rewards[env_id].append(infos[env_id]["episode"]["r"])
                if episode_losses[env_id]:
                    losses[env_id].append(np.mean(episode_losses[env_id]))

                if episode % run_config["log_freq"] == 0:
                    if losses[env_id]:
                        logger.add_scalar(
                            f"{prefix}/loss/{env_id}", np.mean(losses[env_id][-run_config["log_freq"]:]),
                            episode if run_config["training"] else self.test_episodes
                        )
                    logger.add_scalar(
                        f"{prefix}/num_steps/{env_id}", np.mean(steps[env_id][-run_config["log_freq"]:]),
                        episode if run_config["training"] else self.test_episodes
                    )
                    logger.add_scalar(
                        f"{prefix}/reward/{env_id}", np.mean(rewards[env_id][-run_config["log_freq"]:]),
                        episode if run_config["training"] else self.test_episodes
                    )
                    if run_config["show_q_function_diff"]:
                        diff = self.compute_q_fun_diff(run_config["q_true"], env_agents[env_id])
                        logger.add_scalar(
                            f"{prefix}/q_function_diff/{env_id}", diff,
                            episode if run_config["training"] else self.test_episodes
                        )

                if episode_frames[env_id]:
                    video = np.array(episode_frames[env_id]).transpose(0, 3, 1, 2)[
                            np.newaxis, :
                            ]
                    logger.add_video(f"{prefix}/replay/{env_id}", video, self.total_steps)

            if run_config["training"] and episode % run_config["testing_freq"] == 0:
                self._run(self.testing_envs, {
                    "training": False,
                    "log_freq": 1,
                    "recording_freq": 1,
                    "total_episodes": 1,
                    "greedy": run_config.get("greedy", True),
                    "seed": run_config["seed"],
                    "synchronize": run_config["synchronize"],
                    "show_q_function_diff": run_config["show_q_function_diff"],
                    "q_true": run_config["q_true"],
                }, logger)

    # Custom saving for cloudpickle as Julian's environment cannot be saved
    def __reduce__(self):
        # The environments should be recreated and reassigned to the Trainer
        return self.__class__, (dict(), dict(), self.agents)

    @staticmethod
    def compute_q_fun_diff(q_true: np.ndarray, env_agents: Dict[str, "RewardMachineAgent"]) -> np.float64:
        diffs = []
        for ag_id, agent in env_agents.items():
            algo = agent.algo
            assert isinstance(algo, DQN)

            initial_state, initial_u = {'A1': np.array([1, 5, 5, 3, 1, 1])}, 0
            # TODO: check if the output of the network is not normalized (i.e. are we comparing apples with apples? are the 2 q values on the same scale?)
            net_q_value = algo.get_q_values(initial_state, initial_u)
            table_q_values = q_true[initial_u][QRM._to_hashable_state_(initial_state)]

            # FIXME: should this be np.mean(np.abs(...)) ?
            # diffs.append(np.abs(np.mean(net_q_value - table_q_values)))
            diffs.append(np.mean(np.abs(net_q_value - table_q_values)))
        return np.mean(diffs)

    @staticmethod
    def _project_labels(labels, a, aid):
        return {
            aid: a.project_labels(labels)
        }

    @staticmethod
    def _project_obs(obs, _a, aid):
        return {
            i: o for i, o in obs.items() if i == aid
        }

    @staticmethod
    def _get_shared_events(env_agents):
        shared_events = {}
        for aid1, agent1 in env_agents.items():
            for aid2, agent2 in env_agents.items():
                if aid1 >= aid2:
                    continue
                intersection = set(agent1.rm.get_valid_events()).intersection(agent2.rm.get_valid_events())
                shared_events[tuple(sorted([aid1, aid2]))] = intersection
        return shared_events

    @staticmethod
    def _synchronize(shared_events, agent_labels):
        agent_labels = copy.deepcopy(agent_labels)

        for (aid1, aid2), events in shared_events.items():
            for e in events:
                if not all(e in agent_labels[aid] for aid in (aid1, aid2)):
                    agent_labels[aid1] = tuple(l for l in agent_labels[aid1] if l != e)
                    agent_labels[aid2] = tuple(l for l in agent_labels[aid2] if l != e)

        return agent_labels

    @staticmethod
    def _counterfactual_update(env, agent, state, current_u, action, done, next_state):
        labels = env.get_labels(next_state, state)

        if not labels:
            return

        for u in agent.rm.states:
            if u != current_u and not agent.rm.is_state_terminal(u):
                l = env.filter_labels(labels, u)

                next_u = agent.rm.get_next_state(u, l)
                r = agent.rm.get_reward(u, next_u)

                agent.learn(state, u, action, r, done, next_state, next_u)

    def save(self, path):
        trainer_path = os.path.join(path, "trainer.pkl")
        with open(trainer_path, 'wb') as f:
            cloudpickle.dump(self, f)
        print(f"Trainer saved at: {trainer_path}")

    @classmethod
    def load(cls, path):
        if "trainer.pkl" not in path:
            path = os.path.join(path, "trainer.pkl")
        with open(path, 'rb') as f:
            return cloudpickle.load(f)
        # return joblib.load(path)
