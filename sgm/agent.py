from sgm.dependencies import *
from sgm.actor_critic import (GoalConditionedActorNetwork,
                              GoalConditionedCriticNetwork)


class UvfAgent(tf_agent.TFAgent):
    """A UVF Agent."""

    def __init__(
            self,
            time_step_spec,
            action_spec,
            ou_stddev=1.0,
            ou_damping=1.0,
            target_update_tau=0.05,
            target_update_period=5,
            max_episode_steps=None,
            ensemble_size=3,
            combine_ensemble_method='min',
            use_distributional_rl=True):
        """Creates a Uvf Agent.

        Args:
          time_step_spec: A `TimeStep` spec of the expected time_steps.
          action_spec: A nest of BoundedTensorSpec representing the actions.
          ou_stddev: Standard deviation for the Ornstein-Uhlenbeck (OU) noise added
            in the default collect policy.
          ou_damping: Damping factor for the OU noise added in the default collect
            policy.
          target_update_tau: Factor for soft update of the target networks.
          target_update_period: Period for soft update of the target networks.
          max_episode_steps: Int indicating number of steps in an episode. Used for
            determining the number of bins for distributional RL.
          ensemble_size: (int) Number of models in ensemble of critics.
          combine_ensemble_method: (str) At test time, how to combine the distances
            predicted by each member of the ensemble. Options are 'mean', 'min',
            and 'td3'. The 'td3' option is pessimistic w.r.t. the pdf, and then
            takes computes the corresponding distance. The 'min' option takes the
            minimum q values, corresponding to taking the maximum predicted
            distance. Note that we never aggregate predictions during training.
          use_distributional_rl: (bool) Whether to use distributional RL.
        """
        tf.Module.__init__(self, name='UvfAgent')

        assert max_episode_steps is not None
        self._max_episode_steps = max_episode_steps
        self._ensemble_size = ensemble_size
        self._use_distributional_rl = use_distributional_rl

        # Create the actor
        self._actor_network = GoalConditionedActorNetwork(
            time_step_spec.observation, action_spec)
        self._target_actor_network = self._actor_network.copy(
            name='TargetActorNetwork')

        # Create a prototypical critic, which we will copy to create the ensemble.
        critic_net_input_specs = (time_step_spec.observation, action_spec)
        critic_network = GoalConditionedCriticNetwork(
            critic_net_input_specs,
            output_dim=max_episode_steps if use_distributional_rl else None,
        )
        self._critic_network_list = []
        self._target_critic_network_list = []
        for ensemble_index in range(self._ensemble_size):
            self._critic_network_list.append(
                critic_network.copy(name='CriticNetwork%d' % ensemble_index))
            self._target_critic_network_list.append(
                critic_network.copy(name='TargetCriticNetwork%d' % ensemble_index))

        # Create variables for each net.
        net_list = [
                       self._actor_network, self._target_actor_network
                   ] + self._critic_network_list + self._target_critic_network_list
        for net in net_list:
            net.create_variables()

        self._actor_optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=3e-4)
        self._critic_optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=3e-4)

        self._ou_stddev = ou_stddev
        self._ou_damping = ou_damping
        self._target_update_tau = target_update_tau
        self._target_update_period = target_update_period

        self._update_target = self._get_target_updater(
            target_update_tau, target_update_period)

        policy = actor_policy.ActorPolicy(
            time_step_spec=time_step_spec, action_spec=action_spec,
            actor_network=self._actor_network, clip=True)
        collect_policy = actor_policy.ActorPolicy(
            time_step_spec=time_step_spec, action_spec=action_spec,
            actor_network=self._actor_network, clip=False)
        collect_policy = ou_noise_policy.OUNoisePolicy(
            collect_policy,
            ou_stddev=self._ou_stddev,
            ou_damping=self._ou_damping,
            clip=True)

        super(UvfAgent, self).__init__(
            time_step_spec,
            action_spec,
            policy,
            collect_policy,
            train_sequence_length=2)

    def initialize_search(self, active_set, max_search_steps=3,
                          combine_ensemble_method='min'):
        self._combine_ensemble_method = combine_ensemble_method
        self._max_search_steps = max_search_steps
        self._active_set_tensor = tf.convert_to_tensor(active_set)
        pdist = self._get_pairwise_dist(self._active_set_tensor, masked=True,
                                        aggregate=combine_ensemble_method)
        distances = scipy.sparse.csgraph.floyd_warshall(pdist, directed=True)
        self._distances_tensor = tf.convert_to_tensor(
            distances, dtype=tf.float32)

    def _get_pairwise_dist(self, obs_tensor, goal_tensor=None, masked=False,
                           aggregate='mean'):
        """Estimates the pairwise distances.

        Args:
          obs_tensor: Tensor containing observations
          goal_tensor: (optional) Tensor containing a second set of observations. If
            not specified, computes the pairwise distances between obs_tensor and
            itself.
          masked: (bool) Whether to ignore edges that are too long, as defined by
            max_search_steps.
          aggregate: (str) How to combine the predictions from the ensemble. Options
            are to take the minimum predicted q value (i.e., the maximum distance),
            the mean, or to simply return all the predictions.
        """
        if goal_tensor is None:
            goal_tensor = obs_tensor
        dist_matrix = []
        for obs_index in range(obs_tensor.shape[0]):
            obs = obs_tensor[obs_index]
            obs_repeat_tensor = tf.ones_like(
                goal_tensor, dtype="float32") * tf.expand_dims(obs, 0)
            obs_repeat_tensor = tf.ones_like(
                goal_tensor, dtype="float32") * tf.expand_dims(obs, 0)
            obs_goal_tensor = {'observation': obs_repeat_tensor,
                               'goal': goal_tensor}
            pseudo_next_time_steps = time_step.transition(obs_goal_tensor,
                                                          reward=0.0,  # Ignored
                                                          discount=1.0)
            dist = self._get_dist_to_goal(
                pseudo_next_time_steps, aggregate=aggregate)
            dist_matrix.append(dist)

        pairwise_dist = tf.stack(dist_matrix)
        if aggregate is None:
            pairwise_dist = tf.transpose(pairwise_dist, perm=[1, 0, 2])

        if masked:
            mask = (pairwise_dist > self._max_search_steps)
            return tf.where(mask, tf.fill(pairwise_dist.shape, np.inf),
                            pairwise_dist)
        else:
            return pairwise_dist

    def _get_critic_output(self, critic_net_list, next_time_steps,
                           actions=None):
        """Calls the critic net.

        Args:
          critic_net_list: (list) List of critic networks.
          next_time_steps: time_steps holding the observations and step types
          actions: (optional) actions to compute the Q values for. If None, returns
          the Q values for the best action.
        Returns:
          q_values_list: (list) List containing a tensor of q values for each member
          of the ensemble. For distributional RL, computes the expectation over the
          distribution.
        """
        q_values_list = []
        critic_net_input = (next_time_steps.observation, actions)
        for critic_index in range(self._ensemble_size):
            critic_net = critic_net_list[critic_index]
            q_values, _ = critic_net(
                critic_net_input, next_time_steps.step_type)
            q_values_list.append(q_values)
        return q_values_list

    def _get_expected_q_values(self, next_time_steps, actions=None):
        if actions is None:
            actions, _ = self._actor_network(next_time_steps.observation,
                                             next_time_steps.step_type)

        q_values_list = self._get_critic_output(self._critic_network_list,
                                                next_time_steps, actions)

        expected_q_values_list = []
        for q_values in q_values_list:
            if self._use_distributional_rl:
                q_probs = tf.nn.softmax(q_values, axis=1)
                batch_size = q_probs.shape[0]
                bin_range = tf.range(
                    1, self._max_episode_steps + 1, dtype=tf.float32)
                # NOTE: We want to compute the value of each bin, which is the
                # negative distance. Without properly negating this, the actor is
                # optimized to take the *worst* actions.
                neg_bin_range = -1.0 * bin_range
                tiled_bin_range = tf.tile(tf.expand_dims(neg_bin_range, 0),
                                          [batch_size, 1])
                assert q_probs.shape == tiled_bin_range.shape

                # Take the inner produce between these two tensors
                expected_q_values = tf.reduce_sum(
                    q_probs * tiled_bin_range, axis=1)
                expected_q_values_list.append(expected_q_values)
            else:
                expected_q_values_list.append(q_values)
        return tf.stack(expected_q_values_list)

    def _get_state_values(self, next_time_steps, actions=None, aggregate='mean'):
        """Computes the value function, averaging across bins (for distributional RL)
        and the ensemble (for bootstrap RL).

        Args:
          next_time_steps: time_steps holding the observations and step types
          actions: actions for which to compute the Q values. If None, uses the
          best actions (i.e., returns the value function).
        Returns:
          state_values: Tensor storing the observation values for each sample in the
          batch. These values should all be negative.
        """
        with tf.name_scope('state_values'):
            expected_q_values = self._get_expected_q_values(
                next_time_steps, actions)
            if aggregate is not None:
                if aggregate == 'mean':
                    expected_q_values = tf.reduce_mean(
                        expected_q_values, axis=0)
                elif aggregate == 'min':
                    expected_q_values = tf.reduce_min(
                        expected_q_values, axis=0)
                else:
                    raise ValueError('Unknown method for combining ensemble: %s' %
                                     aggregate)

            # Clip the q values if not using distributional RL. If using
            # distributional RL, the q values are implicitly clipped.
            if not self._use_distributional_rl:
                min_q_val = -1.0 * self._max_episode_steps
                max_q_val = 0.0
                expected_q_values = tf.maximum(expected_q_values, min_q_val)
                expected_q_values = tf.minimum(expected_q_values, max_q_val)
            return expected_q_values

    def _get_dist_to_goal(self, next_time_step, aggregate='mean'):
        q_values = self._get_state_values(next_time_step, aggregate=aggregate)
        return -1.0 * q_values

    def _get_waypoint(self, next_time_steps, feasibility_mask=None):
        if feasibility_mask is not None:
            active_set_tensor = tf.boolean_mask(self._active_set_tensor,
                                                feasibility_mask)
        else:
            active_set_tensor = self._active_set_tensor
        obs_tensor = next_time_steps.observation['observation']
        goal_tensor = next_time_steps.observation['goal']
        obs_to_active_set_dist = self._get_pairwise_dist(
            obs_tensor, active_set_tensor, masked=True,
            aggregate=self._combine_ensemble_method)  # B x A
        active_set_to_goal_dist = self._get_pairwise_dist(
            active_set_tensor, goal_tensor, masked=True,
            aggregate=self._combine_ensemble_method)  # A x B

        # The search_dist tensor should be (B x A x A)
        search_dist = sum([
            tf.expand_dims(obs_to_active_set_dist, 2),
            tf.expand_dims(self._distances_tensor, 0),
            tf.expand_dims(tf.transpose(active_set_to_goal_dist), axis=1)
        ])

        # We assume a batch size of 1.
        assert obs_tensor.shape[0] == 1
        min_search_dist = tf.reduce_min(search_dist, axis=[1, 2])[0]
        waypoint_index = tf.argmin(tf.reduce_min(
            search_dist, axis=[2]), axis=1)[0]
        waypoint = active_set_tensor[waypoint_index]

        # For the search policy's bookkeeping
        if feasibility_mask is not None:
            assert len(feasibility_mask.shape) == 1
            original_index = np.argwhere(feasibility_mask == True)[waypoint_index, 0]
        else:
            original_index = waypoint_index

        return waypoint, min_search_dist, original_index

    def _initialize(self):
        for ensemble_index in range(self._ensemble_size):
            common.soft_variables_update(
                self._critic_network_list[ensemble_index].variables,
                self._target_critic_network_list[ensemble_index].variables,
                tau=1.0)
        # Caution: actor should only be updated once.
        common.soft_variables_update(
            self._actor_network.variables,
            self._target_actor_network.variables,
            tau=1.0)

    def _get_target_updater(self, tau=1.0, period=1):
        """Performs a soft update of the target network parameters.

        For each weight w_s in the original network, and its corresponding
        weight w_t in the target network, a soft update is:
        w_t = (1- tau) x w_t + tau x ws

        Args:
          tau: A float scalar in [0, 1]. Default `tau=1.0` means hard update.
          period: Step interval at which the target networks are updated.
        Returns:
          An operation that performs a soft update of the target network parameters.
        """
        with tf.name_scope('get_target_updater'):
            def update():  # pylint: disable=missing-docstring
                critic_update_list = []
                for ensemble_index in range(self._ensemble_size):
                    critic_update = common.soft_variables_update(
                        self._critic_network_list[ensemble_index].variables,
                        self._target_critic_network_list[ensemble_index].variables, tau)
                    critic_update_list.append(critic_update)
                actor_update = common.soft_variables_update(
                    self._actor_network.variables,
                    self._target_actor_network.variables, tau)
                return tf.group(critic_update_list + [actor_update])

            return common.Periodically(update, period, 'periodic_update_targets')

    def _experience_to_transitions(self, experience):
        transitions = trajectory.to_transition(experience)
        transitions = tf.nest.map_structure(lambda x: tf.squeeze(x, [1]),
                                            transitions)

        time_steps, policy_steps, next_time_steps = transitions
        actions = policy_steps.action
        return time_steps, actions, next_time_steps

    def _train(self, experience, weights=None):
        del weights
        time_steps, actions, next_time_steps = self._experience_to_transitions(
            experience)

        # Update the critic
        critic_vars = []
        for ensemble_index in range(self._ensemble_size):
            critic_net = self._critic_network_list[ensemble_index]
            critic_vars.extend(critic_net.variables)

        with tf.GradientTape(watch_accessed_variables=False) as tape:
            assert critic_vars
            tape.watch(critic_vars)
            critic_loss = self.critic_loss(
                time_steps, actions, next_time_steps)
        tf.debugging.check_numerics(critic_loss, 'Critic loss is inf or nan.')
        critic_grads = tape.gradient(critic_loss, critic_vars)
        self._apply_gradients(critic_grads, critic_vars,
                              self._critic_optimizer)

        # Update the actor
        actor_vars = self._actor_network.variables
        with tf.GradientTape(watch_accessed_variables=False) as tape:
            assert actor_vars, 'No actor variables to optimize.'
            tape.watch(actor_vars)
            actor_loss = self.actor_loss(time_steps)
        tf.debugging.check_numerics(actor_loss, 'Actor loss is inf or nan.')
        actor_grads = tape.gradient(actor_loss, actor_vars)
        self._apply_gradients(actor_grads, actor_vars, self._actor_optimizer)

        self.train_step_counter.assign_add(1)
        self._update_target()
        total_loss = actor_loss + critic_loss
        return tf_agent.LossInfo(total_loss, (actor_loss, critic_loss))

    def _apply_gradients(self, gradients, variables, optimizer):
        # Tuple is used for py3, where zip is a generator producing values once.
        grads_and_vars = tuple(zip(gradients, variables))
        optimizer.apply_gradients(grads_and_vars)

    def critic_loss(self,
                    time_steps,
                    actions,
                    next_time_steps):
        """Computes the critic loss for UvfAgent training.

        Args:
          time_steps: A batch of timesteps.
          actions: A batch of actions.
          next_time_steps: A batch of next timesteps.
        Returns:
          critic_loss: A scalar critic loss.
        """
        with tf.name_scope('critic_loss'):
            # We compute the target actions once for all critics.
            target_actions, _ = self._target_actor_network(
                next_time_steps.observation, next_time_steps.step_type)

            critic_loss_list = []
            q_values_list = self._get_critic_output(self._critic_network_list,
                                                    time_steps, actions)
            target_q_values_list = self._get_critic_output(
                self._target_critic_network_list, next_time_steps, target_actions)
            assert len(target_q_values_list) == self._ensemble_size
            for ensemble_index in range(self._ensemble_size):
                # The target_q_values should be a Batch x ensemble_size tensor.
                target_q_values = target_q_values_list[ensemble_index]

                if self._use_distributional_rl:
                    target_q_probs = tf.nn.softmax(target_q_values, axis=1)
                    batch_size = target_q_probs.shape[0]
                    one_hot = tf.one_hot(tf.zeros(batch_size, dtype=tf.int32),
                                         self._max_episode_steps)
                    # Calculate the shifted probabilities
                    # Fist column: Since episode didn't terminate, probability that the
                    # distance is 1 equals 0.
                    col_1 = tf.zeros((batch_size, 1))
                    # Middle columns: Simply the shifted probabilities.
                    col_middle = target_q_probs[:, :-2]
                    # Last column: Probability of taking at least n steps is sum of
                    # last two columns in unshifted predictions:
                    col_last = tf.reduce_sum(target_q_probs[:, -2:], axis=1,
                                             keepdims=True)

                    shifted_target_q_probs = tf.concat([col_1, col_middle, col_last],
                                                       axis=1)
                    assert one_hot.shape == shifted_target_q_probs.shape
                    td_targets = tf.where(next_time_steps.is_last(),
                                          one_hot,
                                          shifted_target_q_probs)
                    td_targets = tf.stop_gradient(td_targets)
                else:
                    td_targets = tf.stop_gradient(
                        next_time_steps.reward +
                        next_time_steps.discount * target_q_values)

                q_values = q_values_list[ensemble_index]
                if self._use_distributional_rl:
                    critic_loss = tf.nn.softmax_cross_entropy_with_logits_v2(
                        labels=td_targets,
                        logits=q_values
                    )
                else:
                    critic_loss = common.element_wise_huber_loss(
                        td_targets, q_values)
                critic_loss = tf.reduce_mean(critic_loss)
                critic_loss_list.append(critic_loss)

            critic_loss = tf.reduce_mean(critic_loss_list)

            with tf.name_scope('Losses/'):
                tf.compat.v2.summary.scalar(
                    name='critic_loss', data=critic_loss, step=self.train_step_counter)

        return critic_loss

    def actor_loss(self, time_steps):
        """Computes the actor_loss for UvfAgent training.

        Args:
          time_steps: A batch of timesteps.
        Returns:
          actor_loss: A scalar actor loss.
        """
        with tf.name_scope('actor_loss'):
            actions, _ = self._actor_network(time_steps.observation,
                                             time_steps.step_type)
            with tf.GradientTape(watch_accessed_variables=False) as tape:
                tape.watch(actions)
                avg_expected_q_values = self._get_state_values(time_steps, actions,
                                                               aggregate='mean')
                actions = tf.nest.flatten(actions)
            dqdas = tape.gradient([avg_expected_q_values], actions)

            actor_losses = []
            for dqda, action in zip(dqdas, actions):
                loss = common.element_wise_squared_loss(
                    tf.stop_gradient(dqda + action), action)
                loss = tf.reduce_sum(loss, axis=1)
                loss = tf.reduce_mean(loss)
                actor_losses.append(loss)

            actor_loss = tf.add_n(actor_losses)

            with tf.name_scope('Losses/'):
                tf.compat.v2.summary.scalar(
                    name='actor_loss', data=actor_loss, step=self.train_step_counter)

        return actor_loss
