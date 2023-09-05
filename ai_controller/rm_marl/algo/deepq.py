import baselines.common.tf_util as U
import gym
import numpy as np
import tensorflow as tf
from baselines import deepq
from baselines.common import set_global_seeds
from baselines.common.models import get_network_builder
from baselines.common.schedules import LinearSchedule
from baselines.common.tf_util import get_session
from baselines.deepq.deepq import ActWrapper, load_act
from baselines.deepq.models import build_q_func
from baselines.deepq.replay_buffer import ReplayBuffer
from baselines.deepq.utils import ObservationInput

from rm_marl.algo import Algo

network_rm_arguments = {
    "num_layers": 3,
    "num_hidden": 1024,
    "activation": tf.nn.relu,
}

path = 'path.tmp'


class DeepQ(Algo):
    """
    Assumes states are numbered from 0 to n - 1
    """

    def __init__(
            self,
            action_space: "gym.spaces.Discrete",
            observation_space: "gym.spaces.Dict",
            num_rm_states: int,
            replay_buffer_size: int = 10000,
            seed: int = 1,
            network=None,
    ):
        self.action_space = action_space
        self.num_rm_states = num_rm_states

        # The observation space is a dictionary including the env features and a one-hot representation of the state in the reward machine
        self.observation_dict = gym.spaces.Dict({'features': observation_space,
                                                 'rm-state': gym.spaces.Box(low=0, high=1, shape=(self.num_rm_states,),
                                                                            dtype=np.uint8)})
        flatdim = gym.spaces.flatdim(self.observation_dict)
        s_low = self._find_low_value(self.observation_dict)
        s_high = self._find_high_value(self.observation_dict)
        self.observation_space = gym.spaces.Box(low=s_low, high=s_high, shape=(flatdim,), dtype=np.float32)
        # self.observation_space = gym.spaces.flatten_space(self.observation_dict)gym.spaces.Box(low=s_low, high=s_high, shape=(flatdim,), dtype=np.float32)

        # TODO: paramterize
        self.replay_buffer_size = replay_buffer_size * self.num_rm_states
        self.batch_size = 32 * self.num_rm_states
        self.learning_starts = 1000
        self.target_network_update_freq = 100
        self.training_freq = self.num_rm_states

        # TODO: parametrize
        self.replay_buffer = ReplayBuffer(self.replay_buffer_size)
        self.exploration = LinearSchedule(schedule_timesteps=10000, initial_p=1.0, final_p=0.02)

        sess = get_session()
        set_global_seeds(seed)
        if network is None:
            self.network = get_network_builder('mlp')(**network_rm_arguments)
        else:
            self.network = network
        q_func = build_q_func(self.network)

        make_obs_ph = lambda name: ObservationInput(self.observation_space, name=name)
        act, train, update_target, debug = deepq.build_train(
            make_obs_ph=make_obs_ph,
            q_func=q_func,
            num_actions=self.action_space.n,
            optimizer=tf.train.AdamOptimizer(learning_rate=5e-4),
        )
        act_params = {
            'make_obs_ph': make_obs_ph,
            'q_func': q_func,
            'num_actions': self.action_space.n,
        }
        self.act = ActWrapper(act, act_params)
        self.train = train
        self.update_target = update_target

        # Initialize the parameters and copy them to the target network.
        U.initialize()
        self.update_target()

        self.step_tracker = 0

    # TODO: test network loading
    # custom method for cloudpickle serialization
    def __reduce__(self):
        return self.__class__, (self.action_space, self.observation_space, self.num_rm_states, self.network)

    @classmethod
    def __newobj__(cls, action_space, observation_space, num_rm_states, network):
        return cls(action_space, observation_space, num_rm_states, network=network)

    def _find_low_value(self, space: gym.spaces.Space):
        if isinstance(space, gym.spaces.Box):
            return space.low[0]
        if isinstance(space, gym.spaces.Discrete):
            return 0
        if isinstance(space, gym.spaces.Dict):
            return min([self._find_low_value(space[k]) for k in space])

        return -1

    def _find_high_value(self, space: gym.spaces.Space):
        if isinstance(space, gym.spaces.Box):
            return space.high[0]
        if isinstance(space, gym.spaces.Discrete):
            return space.n - 1
        if isinstance(space, gym.spaces.Dict):
            return min([self._find_high_value(space[k]) for k in space])
        return -1

    def action(self, state, u, greedy: bool = False):
        # # Execute the action wrapper with the state and u
        # obs = self._get_network_observation(state, u)
        # action = self.act(obs[None], update_eps=self.exploration.value(self.step_tracker))[0]
        # self.step_tracker += 1
        # return action
        return 0

    def learn(self, state, u, action, reward, done, next_state, next_u):
        # # Take action and update exploration to the newest value
        #
        # obs = self._get_network_observation(state, u)
        # next_obs = self._get_network_observation(next_state, next_u)
        #
        # # Store transition in the replay buffer.
        # self.replay_buffer.add(obs, action, reward, next_obs, float(done))
        # obs = next_obs
        #
        # loss = 0
        # # Minimize the error in Bellman's equation on a batch sampled from replay buffer.
        # if self.step_tracker > self.learning_starts and self.step_tracker % self.training_freq == 0:
        #     obses_t, actions, rewards, obses_tp1, dones = self.replay_buffer.sample(self.batch_size)
        #     losses = self.train(obses_t, actions, rewards, obses_tp1, dones, np.ones_like(rewards))
        #     loss = np.abs(losses.sum())
        # # Update target network periodically.
        # if self.step_tracker % self.target_network_update_freq == 0:
        #     self.update_target()
        #
        # return loss
        return 0

    # TODO: the original implementation makes the state all 0 if we reached the end
    def _get_network_observation(self, state, u):
        # 1-hot vector of the current state
        rm_feat = np.zeros(self.num_rm_states)
        rm_feat[u] = 1
        rm_obs = {'features': state, 'rm-state': rm_feat}
        return gym.spaces.flatten(self.observation_dict, rm_obs)
