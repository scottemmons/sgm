from sgm.dependencies import *


def initialize_sorb_search(agent, sorb_search_policy, max_search_steps=6):
    """Make this call before trying to get actions from sorb_policy"""
    agent.initialize_search(sorb_search_policy.rb_vec, max_search_steps=max_search_steps)


def initialize_latent_search(agent, latent_search_policy, max_search_steps=10):
    """Make this call before trying to get actions from latent_policy"""
    agent.initialize_search(latent_search_policy.rb_vec, max_search_steps=max_search_steps)


def set_env_difficulty(tf_env, difficulty):
    """Set goal difficulty of a TF environment"""
    max_goal_dist = tf_env.pyenv.envs[0].gym.max_goal_dist
    tf_env.pyenv.envs[0].gym.set_sample_goal_args(
        prob_constraint=1.0,
        min_dist=max(0, max_goal_dist * (difficulty - 0.05)),
        max_dist=max_goal_dist * (difficulty + 0.05))


def take_cleanup_steps(search_policy, tf_env, num_steps, search_policy_type, duration=600):
    if search_policy_type == "SoRB":
        initialize_sorb_search(search_policy._agent, search_policy)
    elif search_policy_type == "SGM":
        initialize_latent_search(search_policy._agent, search_policy)
    else:
        assert False, "Supports only 'SoRB' and 'SGM' for search_policy_type"

    cleanup_start = time.process_time()

    # Configure search_policy for cleanup
    search_policy._cleanup = True

    # Environment setup
    tf_env.pyenv.envs[0]._duration = duration
    set_env_difficulty(tf_env, 0.95)

    # Step through the environment and clean
    steps_taken = 0
    while steps_taken < num_steps:
        goal = search_policy.set_own_goal()
        goal = tf.convert_to_tensor(goal[np.newaxis])
        ts = tf_env.reset()
        ts.observation['goal'] = goal
        for env_step in range(tf_env.pyenv.envs[0]._duration):
            if ts.is_last() or steps_taken >= num_steps:
                break
            try:
                action = search_policy.action(ts)
            except:
                break
            if search_policy._reached_final_waypoint:
                break
            ts = tf_env.step(action)
            steps_taken += 1
            ts.observation['goal'] = goal

    cleanup_end = time.process_time()
    cleanup_time = cleanup_end - cleanup_start

    return cleanup_time


def eval_search_policy(search_policy, tf_env, search_policy_type, trials=1000, difficulty=0.6, duration=600):
    if search_policy_type == "SoRB":
        initialize_sorb_search(search_policy._agent, search_policy)
    elif search_policy_type == "SGM":
        initialize_latent_search(search_policy._agent, search_policy)
    else:
        assert False, "Supports only 'SoRB' and 'SGM' for search_policy_type"

    eval_start = time.process_time()

    # Don't want additional cleanup to happen here outside of cleanup step budget
    search_policy._cleanup = False

    # Environment setup
    tf_env.pyenv.envs[0]._duration = duration
    set_env_difficulty(tf_env, difficulty)

    # Run evaluation trials
    successes = 0
    action_attempts = 0
    action_time = 0
    for trial in range(trials):
        ts = tf_env.reset()
        count = 0
        failed = False
        for env_step in range(tf_env.pyenv.envs[0]._duration):
            if ts.is_last():
                break
            try:
                action_attempts += 1
                action_start = time.process_time()
                action = search_policy.action(ts)
                action_end = time.process_time()
                action_time += action_end - action_start
            except:
                action_end = time.process_time()
                action_time += action_end - action_start
                failed = True
                break
            ts = tf_env.step(action)
            count += 1
        if count < tf_env.pyenv.envs[0]._duration and not failed:
            successes += 1

    eval_end = time.process_time()
    eval_time = eval_end - eval_start

    return float(successes) / float(trials), action_time, action_attempts, eval_time


def cleanup_and_eval(search_policy, search_policy_type, env_description, eval_tf_env, logdir, eval_difficulty=0.6,
                     k_nearest=5, eval_trials=100, total_cleanup_steps=50000, eval_period=2500):
    """
    Run cleanup, incrementally evaluating success rate.

    Args:
        search_policy: an instance of SoRBSearchPolicy or SGMSearchPolicy, the method to cleanup and evaluate
        search_policy_type: a string specifying either 'SoRB' or 'SGM'. should match the type of `search_policy`
        env_description: a string describing `eval_tf_env` used for logging purposes
        eval_tf_env: the tf environment used for cleanup and evaluation
        logdir: a string specifying the directory for logging
        eval_difficulty: a float between 0 and 1 where 0 is the easiest difficulty and 1 is the hardest difficulty
        k_nearest: will keep the k closest neighbors during edge filtering
        eval_trials: the number of trials used for each success rate evaluation
        total_cleanup_steps: the total number of environment steps to take during cleanup
        eval_period: evaluate the success rate every `eval_period` environment steps

    Returns:
        A string, the directory containing the logged results
    """

    from sgm.utils.logger import DFLogger

    # Validate input parameters
    assert search_policy_type == "SoRB" or search_policy_type == "SGM", "Supports only 'SoRB' and 'SGM' for search_policy_type"

    # Description for filesystem
    search_policy_description = search_policy_type.lower().replace(" ", "_")

    # Log results to csv
    logfolder = os.path.join(logdir,
                             env_description + "_" + search_policy_description + "_" + datetime.datetime.now().strftime(
                                 '%Y-%m-%d_%H-%M-%S'))
    logfile = os.path.join(logfolder, "evaluation.csv")
    columns = ["Planner", "Success Rate", "Cleanup Steps", "Evaluation Trials", "Difficulty", "Priority Cleaning",
               "K-Nearest Filter", "Time Searching Graph for Path", "Path Planning Attempts", "Path Planning Failures",
               "Localization Failures", "Time to Choose Action", "Action Attempts", "Time to Clean Graph",
               "Time to Evaluate Success Rate"]
    logger = DFLogger(logfile, columns=columns, autowrite=True)

    #################################
    # Evaluate initial success rate #
    #################################

    # Evaluate initial success rate
    search_policy.reset_graph_search_stats()
    success_rate, action_time, action_attempts, eval_time = eval_search_policy(search_policy, eval_tf_env,
                                                                               search_policy_type, trials=eval_trials,
                                                                               difficulty=eval_difficulty)
    path_planning_attempts, path_planning_fails, graph_search_time, localization_fails = search_policy.get_graph_search_stats()

    # Log initial evaluation
    print("{} has initial success rate {:.2f}".format(search_policy_type, success_rate))
    logger.add(search_policy_type, success_rate, np.inf, eval_trials, eval_difficulty, False, np.inf, graph_search_time,
               path_planning_attempts, path_planning_fails, localization_fails, action_time, action_attempts, np.inf,
               eval_time)

    # Save initial search policy
    search_policy.save(os.path.join(logfolder, search_policy_description + "_initial"))

    # Print initial evaluation time
    print("Evaluated initial {} success rate in {:.2f} seconds".format(search_policy_type, eval_time))

    ################################
    # Cleanup and evaluation loops #
    ################################

    # Filter search policy
    search_policy.keep_k_nearest(k_nearest)

    # First loop is edge case for timing data
    cleanup_time = np.inf

    # Run the cleanup
    cleanup_steps_taken = 0
    while cleanup_steps_taken < total_cleanup_steps + eval_period:
        # Time the loop
        loop_start = time.process_time()

        # Evaluate SoRB success rate
        search_policy.reset_graph_search_stats()
        success_rate, action_time, action_attempts, eval_time = eval_search_policy(search_policy, eval_tf_env,
                                                                                   search_policy_type,
                                                                                   trials=eval_trials,
                                                                                   difficulty=eval_difficulty)
        path_planning_attempts, path_planning_fails, graph_search_time, localization_fails = search_policy.get_graph_search_stats()

        # Log SoRB run
        print(
            "After filtering and {} cleanup steps, {} has success rate {:.2f}".format(
                cleanup_steps_taken, search_policy_type, success_rate))
        logger.add(search_policy_type, success_rate, cleanup_steps_taken, eval_trials, eval_difficulty, False,
                   k_nearest, graph_search_time, path_planning_attempts, path_planning_fails, localization_fails,
                   action_time, action_attempts, cleanup_time, eval_time)

        # Save cleaned search policy
        search_policy.save(os.path.join(logfolder, search_policy_description + "_filtered_{}_cleanup_steps".format(
            cleanup_steps_taken)))

        # Take cleanup steps
        if cleanup_steps_taken < total_cleanup_steps:
            cleanup_time = take_cleanup_steps(search_policy, eval_tf_env, eval_period, search_policy_type)
            cleanup_steps_taken += eval_period

        else:
            break

        # Print loop time
        loop_end = time.process_time()
        print("Took {} cleanup steps and evaluated {} success rate in {:.2f} seconds".format(eval_period,
                                                                                             search_policy_type,
                                                                                             loop_end - loop_start))

    return logfolder
