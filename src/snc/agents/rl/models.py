import numpy as np
import tensorflow as tf

from tf_agents.environments.tf_py_environment import TFPyEnvironment

from bellman.environments.initial_state_distribution_model import (
    DeterministicInitialStateModel,
)
from bellman.environments.reward_model import RewardModel
from bellman.environments.termination_model import TerminationModel


class CRWRewardModel(RewardModel):
    """
    Reward function for the controlled random walk environment, based on cost_per_buffer.
    Information from the environment is neeeded.
    """
    def __init__(self, observation_spec: tf.TensorSpec, action_spec: tf.TensorSpec, env: TFPyEnvironment):
        self.cost_per_buffer = env.cost_per_buffer
        super().__init__(observation_spec, action_spec)

    def _step_reward(
        self, observation: tf.Tensor, action: tf.Tensor, next_observation: tf.Tensor
    ) -> tf.Tensor:
        cost = np.dot(self.cost_per_buffer.transpose(), observation)
        reward = - float(cost)
        return tf.cast(reward, self._reward_spec.dtype)


class CRWInitialStateModel(DeterministicInitialStateModel):
    """
    Initial state model for the the controlled random walk environment.
    Information from the environment is neeeded.
    """

    def __init__(self, env: TFPyEnvironment):
        self.initial_state = env.state_initialiser.get_initial_state()
        super().__init__(state=self.initial_state)
