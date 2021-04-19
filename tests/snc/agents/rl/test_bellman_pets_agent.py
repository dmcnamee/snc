import pytest
import numpy as np
import tensorflow as tf
from copy import deepcopy

from tf_agents.drivers.dynamic_episode_driver import DynamicEpisodeDriver
from tf_agents.replay_buffers.tf_uniform_replay_buffer import TFUniformReplayBuffer
from tf_agents.specs.tensor_spec import BoundedTensorSpec

import snc.utils.snc_tools as snc
from snc.environments.rl_environment_wrapper import rl_env_from_snc_env
from snc.agents.rl.agents import create_bellman_pets_agent
from snc.environments.scenarios import load_scenario


# import mrl env components
# env._state_initialiser
#
# from bellman.environments.initial_state_distribution_model import (
#     ProbabilisticInitialStateDistributionModel,
# )
# from bellman.environments.reward_model import RewardModel
# from bellman.environments.termination_model import TerminationModel
#
#
# class MountainCarTermination(TerminationModel):
#     """
#     The OpenAI gym termination criteria for Mountain Car are different depending on
#     whether the environment is discrete or continuous. The goal position in the
#     discrete environment is 0.5, while in the continuous environment it is 0.45.
#     """
#
#     def __init__(self, observation_spec: tf.TensorSpec, goal_position=0.5):
#         super().__init__(observation_spec)
#         self._goal_position = tf.constant(goal_position, dtype=observation_spec.dtype)
#
#     def _terminates(self, observation: tf.Tensor) -> bool:
#         return tf.greater_equal(observation[:, 0], self._goal_position)
#
#
# def _calculate_mechanical_energy(state: tf.Tensor) -> tf.Tensor:
#     potential_energy = 0.0025 * tf.sin(3 * state[:, 0])
#     kinetic_energy = 0.5 * tf.square(state[:, 1])
#
#     return potential_energy + kinetic_energy
#
#
# @gin.configurable
# class MountainCarReward(RewardModel):
#     """
#     Reward function for the mountain car environment, based on the change
#     in mechanical energy.
#
#     Mechanical energy is defined as the sum of the potential and kinetic energy of the car.
#     """
#
#     def _step_reward(
#         self, observation: tf.Tensor, action: tf.Tensor, next_observation: tf.Tensor
#     ) -> tf.Tensor:
#         reward = 100 * (
#             _calculate_mechanical_energy(next_observation)
#             - _calculate_mechanical_energy(observation)
#         )
#         return tf.cast(reward, self._reward_spec.dtype)
#
#
# @gin.configurable
# class MountainCarInitialState(ProbabilisticInitialStateDistributionModel):
#     """
#     Initial state distribution for the mountain car environment, ranges are based on
#     OpenAI Gym implementation.
#     """
#
#     def __init__(self, observation_spec: tf.TensorSpec):
#         distribution = tfp.distributions.Uniform(
#             low=tf.convert_to_tensor([-0.6, 0.0], dtype=observation_spec.dtype),
#             high=tf.convert_to_tensor([-0.4, 0], dtype=observation_spec.dtype),
#         )
#         super().__init__(distribution)
#
#
# class DeterministicCRWStateInitialiser(CRWStateInitialiser):
#     """
#     Sets the initial state deterministically according to a given value.
#     """
#
#     def __init__(self, initial_state: snc_types.StateSpace):
#         """
#         :param initial_state: Initial number of jobs per buffer.
#         """
#         self.initial_state = np.array(initial_state)
#
#     def get_initial_state(self) -> snc_types.StateSpace:
#         initial_state = np.asarray(self.initial_state)
#         return initial_state.reshape((len(self.initial_state), 1))
#




@pytest.mark.parametrize(
    'env_name,expected_action_spec_shape', [('single_server_queue', tf.TensorShape((1, 2)))]
)
def test_bellman_pets_agent_init(env_name, expected_action_spec_shape):
    """
    Tests agent set up and initialisation.
    """
    # Set up environment using default parameters.
    # Environment parameters do not affect the test result here.
    tf_env, _ = rl_env_from_snc_env(load_scenario(env_name, job_gen_seed=10)[1],
                                    discount_factor=0.99, normalise_observations=False)

    # Instantiate and initialise a PETS agent for the environment.
    bellman_pets_agent = create_bellman_pets_agent(tf_env, reward_model_class=, initial_state_distribution_model_class=)

    # Validate initialisation by checking relevant properties of the initalised agent.
    assert isinstance(bellman_pets_agent.action_spec, BoundedTensorSpec)
    assert bellman_pets_agent.action_spec.shape == expected_action_spec_shape
    assert bellman_pets_agent.name == "PETS_Agent"
    assert bellman_pets_agent.time_step_spec == tf_env.time_step_spec()


def test_bellman_pets_agent_init_with_multiple_resource_sets():
    """
    Tests agent set up and initialisation with multiple action subspaces (multiple resource sets).
    """
    # Set the environment name for this case as the asserts are difficult to make as variables.
    env_name = 'double_reentrant_line_shared_res_homogeneous_cost'
    # Set up the environment parameters.
    # Environment parameters do not affect the test result here.
    tf_env, _ = rl_env_from_snc_env(load_scenario(env_name, job_gen_seed=10)[1],
                                    discount_factor=0.99, normalise_observations=False)

    # Instantiate and initialise a PPO agent for the environment.
    bellman_pets_agent = create_bellman_pets_agent(tf_env, num_epochs=10)
    bellman_pets_agent.initialize()
    # Validate initialisation by checking some properties of the initalised agent.
    assert isinstance(bellman_pets_agent.action_spec, tuple)
    assert len(bellman_pets_agent.action_spec) == 2
    assert isinstance(bellman_pets_agent.action_spec[0], BoundedTensorSpec)
    assert isinstance(bellman_pets_agent.action_spec[1], BoundedTensorSpec)
    assert bellman_pets_agent.action_spec[0].shape == tf.TensorShape((1, 3))
    assert bellman_pets_agent.action_spec[1].shape == tf.TensorShape((1, 3))
    assert bellman_pets_agent.name == "PETS_Agent"
    assert bellman_pets_agent.time_step_spec == tf_env.time_step_spec()


# Parameterise with environments which cover the cases of a single resource set and multiple
# resource sets.
@pytest.mark.parametrize(
    'env_name',
    ['single_server_queue', 'double_reentrant_line_shared_res_homogeneous_cost']
)
def test_bellman_pets_agent_play(env_name):
    """
    Extension of the agent set up and initialisation test to include playing episodes.
    """
    # Set up environment using default parameters.
    # Environment parameters do not affect the test result here.
    tf_env, action_dims = rl_env_from_snc_env(load_scenario(env_name, job_gen_seed=10)[1],
                                              discount_factor=0.99, normalise_observations=False)

    # Instantiate and initialise a PPO agent for the environment.
    bellman_pets_agent = create_bellman_pets_agent(tf_env, num_epochs=10)
    bellman_pets_agent.initialize()

    # Reset the environment
    tf_env.reset()
    # Play 5 time steps in the environment.
    for _ in range(5):
        # Since we do not have the state stored at this point we capture it from the environment
        # fresh each time step as a TimeStep object (a named tuple).
        time_step = tf_env.current_time_step()
        # Attain our agent's action.
        action_step = bellman_pets_agent.collect_policy.action(time_step)
        # Ensure that the action is one-hot as expected
        if isinstance(action_step.action, tuple):
            action = tf.concat(action_step.action, axis=-1)
        else:
            action = action_step.action

        # Ensure that the action is binary as expected.
        assert snc.is_binary(action)
        # Play the action out in the environment.
        tf_env.step(action_step.action)


# Parameterise with environments which cover the cases of a single resource set and multiple
# resource sets.
@pytest.mark.parametrize(
    'env_name',
    ['single_server_queue', 'double_reentrant_line_shared_res_homogeneous_cost']
)
def test_bellman_pets_agent_learning(env_name):
    """
    Extension of the play test for an agent playing in the environment to include training.
    Note: This does not test that training improves the policy. It simply tests that the training
    loop runs effectively and changes the policy parameters.
    """
    # Set up environment using default parameters.
    # Environment parameters do not affect the test result here.
    tf_env, _ = rl_env_from_snc_env(
        load_scenario(env_name, job_gen_seed=10, override_env_params={'max_episode_length': 25})[1],
        discount_factor=0.99, normalise_observations=False
    )

    # Set up a training step counter.
    global_step = tf.compat.v1.train.get_or_create_global_step()
    # Instantiate a PPO agent
    bellman_pets_agent = create_bellman_pets_agent(tf_env, num_epochs=10, training_step_counter=global_step)

    # Instantiate a replay buffer.
    replay_buffer = TFUniformReplayBuffer(
        data_spec=bellman_pets_agent.collect_data_spec,
        batch_size=tf_env.batch_size,
        max_length=1000)

    # Use a driver to handle data collection for the agent. This handles a lot of the backend
    # TensorFlow set up and solves previous errors with episodes of differing lengths.
    collect_driver = DynamicEpisodeDriver(
        tf_env,
        bellman_pets_agent.collect_policy,
        observers=[replay_buffer.add_batch],
        num_episodes=2)

    # collect_driver.run = tf.function(collect_driver.run)

    # Get the initial states of the agent and environment before training.
    time_step = tf_env.reset()
    policy_state = bellman_pets_agent.collect_policy.get_initial_state(tf_env.batch_size)

    # Take a copy of the variables in order to ensure that training does lead to parameter changes.
    initial_vars = deepcopy(bellman_pets_agent.trainable_variables)
    assert len(initial_vars) > 0, "Agent has no trainable variables."

    # Set up a minimal training loop to simply test training mechanics work.
    for _ in range(5):
        # Collect experience.
        time_step, policy_state = collect_driver.run(
            time_step=time_step,
            policy_state=policy_state
        )
        # Now the replay buffer should have data in it so we can collect the data and train the
        # agent.
        experience = replay_buffer.gather_all()
        bellman_pets_agent.train(experience)
        # Clear the replay buffer and return to play.
        replay_buffer.clear()

    # Check that training has had some effect
    for v1, v2 in zip(initial_vars, bellman_pets_agent.trainable_variables):
        assert not np.allclose(v1.numpy(), v2.numpy())
