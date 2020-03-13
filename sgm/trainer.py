from sgm.dependencies import *


def get_checkpoint_and_writers(log_dir, tf_agent, experiment_name="ckpt"):
    current_time = datetime.datetime.now().strftime("%b-%d-%Y-%I-%M-%S-%p")
    log_dir = os.path.join(log_dir, f"{experiment_name}-{current_time}")

    # save checkpoint object
    checkpoint_dir = os.path.join(log_dir, "ckpt")
    tf.compat.v1.logging.info("Saving checkpoints to directory %s", checkpoint_dir)
    ckpt = tf.train.Checkpoint(step=tf.Variable(1), net=tf_agent)
    manager = tf.train.CheckpointManager(ckpt, checkpoint_dir, max_to_keep=3)
    ckpt.restore(manager.latest_checkpoint)

    # create summary writer
    tf.compat.v1.logging.info("Creating tensorboard summary writer with directory %s", log_dir)
    summary_writer = tf.contrib.summary.create_file_writer(log_dir)

    return ckpt, manager, summary_writer


def train_eval(
        tf_agent,
        tf_env,
        eval_tf_env,
        num_iterations=2000000,
        # Params for collect
        initial_collect_steps=1000,
        batch_size=64,
        # Params for eval
        num_eval_episodes=100,
        eval_interval=10000,
        eval_distances=[2, 5, 10],
        # Params for checkpoints, summaries, and logging
        log_interval=1000,
        save_model_dir=None,
        random_seed=0,
        experiment_name="ckpt"):
    """A simple train and eval for UVF."""
    tf.compat.v1.logging.info('random_seed = %d' % random_seed)
    np.random.seed(random_seed)
    random.seed(random_seed)
    tf.compat.v1.set_random_seed(random_seed)

    max_episode_steps = tf_env.pyenv.envs[0]._duration
    global_step = tf.compat.v1.train.get_or_create_global_step()
    replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
        tf_agent.collect_data_spec,
        batch_size=tf_env.batch_size)

    eval_metrics = [
        tf_metrics.AverageReturnMetric(buffer_size=num_eval_episodes),
    ]

    eval_policy = tf_agent.policy
    collect_policy = tf_agent.collect_policy
    initial_collect_driver = dynamic_step_driver.DynamicStepDriver(
        tf_env,
        collect_policy,
        observers=[replay_buffer.add_batch],
        num_steps=initial_collect_steps)

    collect_driver = dynamic_step_driver.DynamicStepDriver(
        tf_env,
        collect_policy,
        observers=[replay_buffer.add_batch],
        num_steps=1)

    initial_collect_driver.run = common.function(initial_collect_driver.run)
    collect_driver.run = common.function(collect_driver.run)
    tf_agent.train = common.function(tf_agent.train)

    initial_collect_driver.run()

    time_step = None
    policy_state = collect_policy.get_initial_state(tf_env.batch_size)

    timed_at_step = global_step.numpy()
    time_acc = 0

    # Dataset generates trajectories with shape [Bx2x...]
    dataset = replay_buffer.as_dataset(
        num_parallel_calls=3,
        sample_batch_size=batch_size,
        num_steps=2).prefetch(3)
    iterator = iter(dataset)

    ckpt, manager, summary_writer = get_checkpoint_and_writers(save_model_dir, tf_agent, experiment_name)

    with summary_writer.as_default(), tf.contrib.summary.always_record_summaries():
        for _ in tqdm.tqdm(range(num_iterations)):

            start_time = time.time()

            ckpt.step.assign_add(1)
            if int(ckpt.step) % log_interval == 0:
                save_path = manager.save()
                print("Saved checkpoint for step {}: {}".format(
                    int(ckpt.step), save_path))

            time_step, policy_state = collect_driver.run(
                time_step=time_step,
                policy_state=policy_state,
            )

            experience, _ = next(iterator)
            train_loss = tf_agent.train(experience)
            time_acc += time.time() - start_time
            tf.contrib.summary.scalar(f"Train_loss", train_loss.loss, step=global_step.numpy())

            if global_step.numpy() % log_interval == 0:
                tf.compat.v1.logging.info('step = %d, loss = %f', global_step.numpy(),
                                          train_loss.loss)
                steps_per_sec = log_interval / time_acc
                tf.compat.v1.logging.info('%.3f steps/sec', steps_per_sec)
                tf.contrib.summary.scalar("steps_per_sec", steps_per_sec, step=global_step.numpy())
                time_acc = 0

            if global_step.numpy() % eval_interval == 0:
                start = time.time()
                tf.compat.v1.logging.info('step = %d' % global_step.numpy())
                for dist in eval_distances:
                    try:
                        eval_tf_env.pyenv.envs[0].gym.set_sample_goal_args(
                            prob_constraint=1.0, min_dist=dist - 1, max_dist=dist + 1)
                        tf.compat.v1.logging.info('\t set goal dist = %d' % dist)
                    except AttributeError:
                        pass

                    results = metric_utils.eager_compute(
                        eval_metrics,
                        eval_tf_env,
                        eval_policy,
                        num_episodes=num_eval_episodes,
                        train_step=global_step,
                        summary_prefix='Metrics',
                    )
                    for (key, value) in results.items():
                        tf.compat.v1.logging.info('\t\t %s (eval distance=%f) %.2f', key, dist, value.numpy())
                        tf.contrib.summary.scalar(f"Eval_{key}_d{dist}", value.numpy(), step=global_step.numpy())
                    # For debugging, it's helpful to check the predicted distances for
                    # goals of known distance.
                    pred_dist = []
                    for _ in range(num_eval_episodes):
                        ts = eval_tf_env.reset()
                        dist_to_goal = tf_agent._get_dist_to_goal(ts)[0]
                        pred_dist.append(dist_to_goal.numpy())
                    average_dist, std_dist = np.mean(pred_dist), np.std(pred_dist)
                    tf.compat.v1.logging.info('\t\t predicted_dist = %.1f (%.1f)' %
                                              (np.mean(pred_dist), np.std(pred_dist)))
                    tf.contrib.summary.scalar(f"Eval_AverageDist_d{dist}", average_dist, step=global_step.numpy())
                    tf.contrib.summary.scalar(f"Eval_StdDist_d{dist}", std_dist, step=global_step.numpy())
                tf.compat.v1.logging.info('\t eval_time = %.2f' % (time.time() - start))

    return train_loss
