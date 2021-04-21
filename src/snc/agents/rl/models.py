@gin.configurable
class CRWRewardModel(RewardModel):
    """
    Reward function for the controlled random walk environment, based on cost_per_buffer.
    """
    def __init__(self, env: TFPyEnvironment):
        self.cost_per_buffer = env.cost_per_buffer
        super.__init__(self)

    def _step_reward(
        self, observation: tf.Tensor, action: tf.Tensor, next_observation: tf.Tensor
    ) -> tf.Tensor:
        cost = np.dot(self.cost_per_buffer.transpose(), observation)
        reward = - float(cost)
        return tf.cast(reward, self._reward_spec.dtype)
