from sgm.dependencies import *
from sgm.envs import *


def get_plan_vec(search_policy, start, goal):
    """Create a numpy array of waypoints for an existing plan"""
    waypoint_vec = [start]
    for waypoint_index in search_policy._waypoint_vec:
        waypoint_vec.append(search_policy.rb_vec[waypoint_index])
    waypoint_vec.append(goal)
    waypoint_vec = np.array(waypoint_vec)
    return waypoint_vec


def set_env_difficulty(tf_env, difficulty):
    """Set goal difficulty of a TF environment"""
    max_goal_dist = tf_env.pyenv.envs[0].gym.max_goal_dist
    tf_env.pyenv.envs[0].gym.set_sample_goal_args(
        prob_constraint=1.0,
        min_dist=max(0, max_goal_dist * (difficulty - 0.05)),
        max_dist=max_goal_dist * (difficulty + 0.05))


def reset_with_difficulty(tf_env, difficulty=0.5):
    """Set goal difficulty and reset a TF environment"""
    set_env_difficulty(tf_env, difficulty)

    ts = tf_env.reset()
    start = ts.observation['observation'].numpy()[0]
    goal = ts.observation['goal'].numpy()[0]

    return ts, start, goal

def plot_points(points, tf_env):
    plt.figure(figsize=(6, 6))
    plt.scatter(*points.T)
    plot_walls(tf_env.pyenv.envs[0].env.walls)
    plt.show()

def plot_qvalue_neighbors(base_index, cutoff, rb_vec, pdist_combined, tf_env, invert=False):
    neighbor_checks = pdist_combined[base_index, :] < cutoff
    if invert:
        neighbor_checks = np.invert(neighbor_checks)
    neighbors = rb_vec[neighbor_checks]

    plot_points(neighbors, tf_env)

def plot_qconsistency_neighbors(base_index, cutoff, rb_vec, pdist_combined, tf_env, scale=False, invert=False,
                                mask=None):
    if mask is not None:
        pdist_combined = pdist_combined[:, mask]
    base_row = pdist_combined[base_index]
    if scale:
        # calculate scaling to minimize np.sum((base_row - scaling * target_row)**2)
        dot_products = np.sum(base_row * pdist_combined, axis=1)
        sum_of_squares = np.sum(pdist_combined ** 2, axis=1)
        scalings = dot_products / sum_of_squares
        qval_diffs = base_row - scalings[:, np.newaxis] * pdist_combined
    else:
        qval_diffs = base_row - pdist_combined
    qval_inconsistencies = np.linalg.norm(qval_diffs, np.inf, axis=1)
    neighbor_checks = qval_inconsistencies < cutoff
    if invert:
        neighbor_checks = np.invert(neighbor_checks)
    neighbors = rb_vec[neighbor_checks]

    plot_points(neighbors, tf_env)

def plot_plan(waypoint_vec, tf_env):
    start = waypoint_vec[0]
    goal = waypoint_vec[-1]

    plt.figure(figsize=(6, 6))
    plot_walls(tf_env.pyenv.envs[0].env.walls)

    plt.scatter([start[0]], [start[1]], marker='+',
                color='red', s=200, label='start')
    plt.scatter([goal[0]], [goal[1]], marker='*',
                color='green', s=200, label='goal')
    plt.plot(waypoint_vec[:, 0], waypoint_vec[:, 1], 'y-s', alpha=0.3, label='waypoint')
    plt.legend(loc='lower left', bbox_to_anchor=(-0.1, -0.15), ncol=4, fontsize=16)
    plt.show()


def plot_graph(g, rb_vec, tf_env):
    plt.figure(figsize=(6, 6))
    plot_walls(tf_env.pyenv.envs[0].env.walls)
    plt.scatter(rb_vec[g.nodes, 0], rb_vec[g.nodes, 1])

    edges_to_plot = g.edges
    edges_to_plot = np.array(list(edges_to_plot))
    print("Plotting {} nodes and {} edges".format(g.number_of_nodes(), len(edges_to_plot)))

    for i, j in edges_to_plot:
        s_i = rb_vec[i]
        s_j = rb_vec[j]
        plt.plot([s_i[0], s_j[0]], [s_i[1], s_j[1]], c='k', alpha=0.5)

    plt.show()


def visualize_rollout(search_policy, tf_env, difficulty, duration=600):
    """
    Visualize all waypoint plans as search_policy prunes edges and replans when rolled out in tf_env
    """
    tf_env.pyenv.envs[0]._duration = duration
    set_env_difficulty(tf_env, difficulty)
    seed = np.random.randint(0, 1000000)

    goals = []
    obs_vecs = []
    plan_vecs = []
    for i in range(2):
        search = i != 0
        desc = "search" if search else "no search"
        np.random.seed(seed)
        search_policy.replanned_last_action = False
        ts = tf_env.reset()
        start = ts.observation["observation"].numpy()[0]
        goal = ts.observation["goal"].numpy()[0]
        obs_vec = []
        plan_vec = []
        for j in tqdm.tqdm_notebook(range(duration), desc=desc):
            if ts.is_last():
                break
            obs_vec.append(ts.observation["observation"].numpy()[0])
            if search:
                plan_vec = get_plan_vec(search_policy, start, goal)
                action = search_policy.action(ts)
            else:
                action = search_policy._agent.policy.action(ts)
            ts = tf_env.step(action)
            if search_policy.replanned_last_action or ts.is_last() or j == duration - 1:
                goals.append(goal)
                obs_vecs.append(obs_vec)
                plan_vecs.append(plan_vec)
                start = ts.observation["observation"].numpy()[0]
                goal = ts.observation["goal"].numpy()[0]
                obs_vec = []
                plan_vec = []

    subfigures = len(goals)
    columns = 2
    rows = np.ceil(subfigures / columns)
    plt.figure(figsize=(6 * columns, 5 * rows))
    for i in range(subfigures):
        title = "No Search" if i == 0 else f"Plan {i}"
        plt.subplot(rows, columns, i + 1)
        plot_walls(tf_env.pyenv.envs[0].env.walls)

        goal = goals[i]
        obs_vec = np.array(obs_vecs[i])
        plan_vec = plan_vecs[i]

        plt.plot(obs_vec[:, 0], obs_vec[:, 1], 'b-o', alpha=0.3)
        plt.scatter([obs_vec[0, 0]], [obs_vec[0, 1]], marker='+',
                    color='red', s=200, label='start')
        plt.scatter([obs_vec[-1, 0]], [obs_vec[-1, 1]], marker='+',
                    color='green', s=200, label='end')
        plt.scatter([goal[0]], [goal[1]], marker='*',
                    color='green', s=200, label='goal')
        plt.title(title, fontsize=24)

        if i > 0:
            plt.plot(plan_vec[:, 0], plan_vec[:, 1], 'y-s', alpha=0.3, label='waypoint')
    if subfigures % columns == 0:
        plt.legend(loc='lower left', bbox_to_anchor=(-0.8, -0.15), ncol=4, fontsize=16)
    else:
        plt.legend(loc='lower left', bbox_to_anchor=(0.8, -0.15), ncol=4, fontsize=16)

    plt.show()
