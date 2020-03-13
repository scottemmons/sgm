from typing import Union

from numpy.core._multiarray_umath import ndarray

from sgm.dependencies import *


class SoRBSearchPolicy(tf_policy.Base):
    def __init__(self, agent, pdist, rb_vec, cleanup=False, no_waypoint_hopping=False, weighted_path_planning=False,
                 localize_to_nearest=False):
        """
        Args:
            agent: the UvfAgent that self._action will use to act in the environment
            pdist: a matrix of dimension len(rb_vec) x len(rb_vec) where pdist[i,j] gives the distance going from
                rb_vec[i] to rb_vec[j]
            rb_vec: a replay buffer vector storing the observations that will be used as nodes in the graph
            cleanup: if True, will prune edges when fail to reach waypoint after self._attempt_cutoff
            no_waypoint_hopping: if True, will not try to proceed to goal until all waypoints have been reached
            weighted_path_planning: whether or not to use edge weights when planning a shortest path from start to goal
            localize_to_nearest: if True, will incrementally add edges with incoming start and goal nodes until path
                exists from start to goal; otherwise, adds all edges with incoming start and goal nodes that have
                distance less than `max_search_steps`
        """
        self.pdist = pdist
        self.rb_vec = rb_vec
        self._agent = agent
        self._attempt_cutoff = 3 * self._agent._max_search_steps
        self._cleanup = cleanup
        self._no_waypoint_hopping = no_waypoint_hopping
        self._weighted_path_planning = weighted_path_planning
        self._localize_to_nearest = localize_to_nearest
        self._waypoint_vec = []
        self.replanned_last_action = False
        self._graph_search_time = 0
        self.reset_graph_search_stats()
        self._g = self._build_graph()

        super(SoRBSearchPolicy, self).__init__(agent.policy.time_step_spec, agent.policy.action_spec)

    def _build_graph(self):
        g = nx.DiGraph()
        pdist_combined = np.max(self.pdist, axis=0)
        for i, s_i in enumerate(self.rb_vec):
            for j, s_j in enumerate(self.rb_vec):
                length = pdist_combined[i, j]
                if length < self._agent._max_search_steps:
                    g.add_edge(i, j, weight=length)
        return g

    def save(self, folder):
        if not os.path.exists(folder):
            os.makedirs(folder)
        np.save(os.path.join(folder, "pdist.npy"), self.pdist)
        np.save(os.path.join(folder, "rb_vec.npy"), self.rb_vec)
        nx.write_gpickle(self._g, os.path.join(folder, "g.pickle"))

    def load(self, folder):
        self.pdist = np.load(os.path.join(folder, "pdist.npy"))
        self.rb_vec = np.load(os.path.join(folder, "rb_vec.npy"))
        self._g = nx.read_gpickle(os.path.join(folder, "g.pickle"))
        self.reset_graph_search_stats()

    def reset_graph_search_stats(self):
        self._path_planning_attempts = 0
        self._path_planning_fails = 0
        self._graph_search_time = 0
        self._localization_fails = 0

    def get_graph_search_stats(self):
        return self._path_planning_attempts, self._path_planning_fails, self._graph_search_time, self._localization_fails

    def keep_k_nearest(self, k):
        """
        For each node in the graph, keeps only the k outgoing edges with lowest weight.
        """
        for node in self._g.nodes():
            edges = list(self._g.edges(nbunch=node, data='weight', default=np.inf))
            edges.sort(key=lambda x: x[2])
            try:
                edges_to_remove = edges[k:]
            except IndexError:
                edges_to_remove = []
            self._g.remove_edges_from(edges_to_remove)

    def _construct_planning_graph(self, time_step):
        start_to_rb = self._agent._get_pairwise_dist(time_step.observation['observation'], self.rb_vec, aggregate='min',
                                                     masked=True).numpy().flatten()
        rb_to_goal = self._agent._get_pairwise_dist(self.rb_vec, time_step.observation['goal'], aggregate='min',
                                                    masked=True).numpy().flatten()

        planning_graph = self._g.copy()
        if self._localize_to_nearest:
            sorted_start_indices = np.argsort(start_to_rb)
            sorted_goal_indices = np.argsort(rb_to_goal)
            neighbors_added = 0
            while neighbors_added < len(start_to_rb):
                i = sorted_start_indices[neighbors_added]
                j = sorted_goal_indices[neighbors_added]
                planning_graph.add_edge('start', i, weight=start_to_rb[i])
                planning_graph.add_edge(j, 'goal', weight=rb_to_goal[j])
                try:
                    nx.shortest_path(planning_graph, source='start', target='goal')
                    break
                except nx.NetworkXNoPath:
                    neighbors_added += 1

        else:
            for i, (dist_from_start, dist_to_goal) in enumerate(zip(start_to_rb, rb_to_goal)):
                if dist_from_start < self._agent._max_search_steps:
                    planning_graph.add_edge('start', i, weight=dist_from_start)
                if dist_to_goal < self._agent._max_search_steps:
                    planning_graph.add_edge(i, 'goal', weight=dist_to_goal)

        if not np.any(start_to_rb < self._agent._max_search_steps) or not np.any(
                rb_to_goal < self._agent._max_search_steps):
            self._localization_fails += 1

        self._planning_graph = planning_graph
        return planning_graph

    def _get_path(self, time_step):
        self._construct_planning_graph(time_step)
        g2 = self._planning_graph
        try:
            self._path_planning_attempts += 1
            graph_search_start = time.process_time()
            if self._weighted_path_planning:
                path = nx.shortest_path(g2, source='start', target='goal', weight='weight')
            else:
                path = nx.shortest_path(g2, source='start', target='goal')
        except:
            self._path_planning_fails += 1
            raise RuntimeError("Failed to find path")
        finally:
            graph_search_end = time.process_time()
            self._graph_search_time += graph_search_end - graph_search_start
        edge_lengths = []
        for (i, j) in zip(path[:-1], path[1:]):
            edge_lengths.append(g2[i][j]['weight'])
        wypt_to_goal_dist = np.cumsum(edge_lengths[::-1])[::-1]  # Reverse CumSum
        waypoint_vec = list(path)[1:-1]
        return waypoint_vec, wypt_to_goal_dist[1:]

    def _initialize_path(self, time_step):
        self._waypoint_vec, self._wypt_to_goal_dist_vec = self._get_path(time_step)
        self._waypoint_counter = 0
        self._waypoint_attempts = 0
        self._reached_final_waypoint = False

    def _reached_waypoint(self, dist_to_waypoint, observation, waypoint_index):
        return dist_to_waypoint < self._agent._max_search_steps

    def _action(self, time_step, policy_state=(), seed=None):
        assert time_step.observation['observation'].shape[0] == 1
        goal = time_step.observation['goal']
        dist_to_goal = self._agent._get_dist_to_goal(time_step)[0].numpy()
        if time_step.is_first():
            self._initialize_path(time_step)
        if self._cleanup and self._waypoint_attempts >= self._attempt_cutoff:
            # prune edge and replan
            if self._waypoint_counter != 0 and not self._reached_final_waypoint:
                src_node = self._waypoint_vec[self._waypoint_counter - 1]
                dest_node = self._waypoint_vec[self._waypoint_counter]
                self._g.remove_edge(src_node, dest_node)
            self._initialize_path(time_step)
            self.replanned_last_action = True
        else:
            self.replanned_last_action = False
        waypoint = self.rb_vec[self._waypoint_vec[self._waypoint_counter]]
        time_step.observation['goal'] = waypoint[None]
        dist_to_waypoint = self._agent._get_dist_to_goal(time_step)[0].numpy()
        if self._reached_waypoint(dist_to_waypoint, time_step.observation['observation'][0],
                                  self._waypoint_vec[self._waypoint_counter]):
            if not self._reached_final_waypoint:
                self._waypoint_attempts = 0
            self._waypoint_counter += 1
            if self._waypoint_counter > len(self._waypoint_vec) - 1:
                self._reached_final_waypoint = True
                self._waypoint_counter = len(self._waypoint_vec) - 1
            waypoint = self.rb_vec[self._waypoint_vec[self._waypoint_counter]]
            time_step.observation['goal'] = waypoint[None]
            dist_to_waypoint = self._agent._get_dist_to_goal(time_step._replace())[0].numpy()
        dist_to_goal_via_wypt = dist_to_waypoint + self._wypt_to_goal_dist_vec[self._waypoint_counter]

        if (self._no_waypoint_hopping and not self._reached_final_waypoint) or (
                dist_to_goal_via_wypt < dist_to_goal) or (
                dist_to_goal > self._agent._max_search_steps):
            time_step.observation['goal'] = tf.convert_to_tensor(waypoint[None])
            self._waypoint_attempts += 1
        else:
            time_step.observation['goal'] = goal
            if self._reached_final_waypoint:
                self._waypoint_attempts += 1
        return self._agent.policy.action(time_step, policy_state, seed)

    def set_own_goal(self):
        goal_index = np.random.randint(low=0, high=self.rb_vec.shape[0])
        return self.rb_vec[goal_index].copy()


class SGMSearchPolicy(SoRBSearchPolicy):
    def __init__(self, agent, pdist, rb_vec, embedding_vec, embedding_cutoff=0.05, consistency_cutoff=5,
                 query_embeddings=False, cache_pdist=True, cleanup=True, no_waypoint_hopping=True,
                 weighted_path_planning=False, localize_to_nearest=True):
        """
        Args:
            agent: the UvfAgent that self._action will use to act in the environment
            pdist: a matrix of dimension len(rb_vec) x len(rb_vec) where pdist[i,j] gives the distance going from
                rb_vec[i] to rb_vec[j]
            rb_vec: a replay buffer vector storing the observations that will be used as nodes in the graph
            embedding_vec: an array of shape replay_buffer_size x z-dim which has the latent embedding of each
                observation in the replay buffer `rb_vec`
            cleanup: if True, will prune edges when fail to reach waypoint after self._attempt_cutoff
            no_waypoint_hopping: if True, will not try to proceed to goal until all waypoints have been reached
            weighted_path_planning: whether or not to use edge weights when planning a shortest path from start to goal
            localize_to_nearest: if True, will incrementally add edges with incoming start and goal nodes until path
                exists from start to goal; otherwise, adds all edges with incoming start and goal nodes that have
                distance less than `max_search_steps`
        """
        self.embedding_vec = embedding_vec
        self.embedding_cutoff = embedding_cutoff
        self.consistency_cutoff = consistency_cutoff
        self.query_embeddings = query_embeddings
        self.cache_pdist = cache_pdist
        if cache_pdist:
            self.cached_pdist = pdist
        super(SGMSearchPolicy, self).__init__(agent, pdist, rb_vec,
                                              cleanup=cleanup,
                                              no_waypoint_hopping=no_waypoint_hopping,
                                              weighted_path_planning=weighted_path_planning,
                                              localize_to_nearest=localize_to_nearest)

    def save(self, folder):
        super(SGMSearchPolicy, self).save(folder)
        np.save(os.path.join(folder, "embedding_vec.npy"), self.embedding_vec)
        np.save(os.path.join(folder, "embedding_vars.npy"), self.embedding_vars)

    def load(self, folder):
        super(SGMSearchPolicy, self).load(folder)
        self.embedding_vec = np.load(os.path.join(folder, "embedding_vec.npy"))
        self.embedding_vars = np.load(os.path.join(folder, "embedding_vars.npy"))

    def embed(self, observation):
        return observation

    def _reached_waypoint(self, dist_to_waypoint, observation, waypoint_index, waypoint_q_cutoff=1, verbose=False):
        waypoint_qvals_combined = np.max(self.pdist, axis=0)[waypoint_index, :]
        obs_qvals = self._agent._get_pairwise_dist(observation[np.newaxis], self.rb_vec, aggregate=None).numpy()
        obs_qvals_combined = np.max(obs_qvals, axis=0).flatten()

        qval_diffs = waypoint_qvals_combined - obs_qvals_combined
        qval_inconsistency = np.linalg.norm(qval_diffs, np.inf)
        return qval_inconsistency < waypoint_q_cutoff

    def _build_graph(self):
        """If self.cache_pdist, assumes indexing of self.rb_vec matches indexing of self.cached_pdist"""
        self._g = nx.DiGraph()
        observations_to_add = self.rb_vec
        embeddings_to_add = self.embedding_vec
        for index, (observation, embedding) in enumerate(zip(observations_to_add, embeddings_to_add)):
            self._update_graph(observation, embedding, cache_index=index)
        return self._g

    def _update_graph(self, observation, embedding, cache_index=None, edge_cutoff=10, beta=0.05,
                      embedding_buffer=False):
        """Performs dynamic graph building.

        If self.cache_pdist and not query_embeddings, calculates qvalues by
        looking them up in self.cached_pdist. Otherwise, calculates by qvalues
        with a call to self._agent._get_pairwise_dist().

        Args:
            observation: the observation to consider adding to the graph. observation should have
                   dimensions matching those in self.rb_vec
            embedding: a 1D vector, the latent space embedding of obs
            cache_index: the index of obs in self.cached_pdist, used to look
                         up qvalues if self.cache_pdist
            edge_cutoff: draw directed edges between nodes when their qvalue
                         distance is less than edge_cutoff
            beta: percentage assigned to newest embedding space observation
                  in exponential moving average / variance calculations
            embedding_buffer: if True, fill the replay buffer with embedding
                              vectors instead of observations

        Result:
            Stores a directed graph of merged nodes in self._g
            Populates the replay buffer self.rb_vec either with a
                representative observation for each node or, if
                embedding_buffer, with a moving average of the node's latent
                embedding
            Stores a moving average of nodes' latent embeddings in
                self.embedding_vec
            Stores a moving variance of nodes' latent embeddings in
                self.embedding_vars
            Calculates pairwise qvalues between nodes, stored in self.pdist
        """
        assert self._g.number_of_nodes() == 0 or \
               self._g.number_of_nodes() == len(self.rb_vec) == len(self.embedding_vec)
        assert len(embedding.shape) == 1, "we assume the latents are vectors"
        if self.cache_pdist: assert cache_index is not None

        # Initial construction of objects
        if self._g.number_of_nodes() == 0:
            self._g.add_node(0)
            self.rb_vec = observation[np.newaxis]
            self.embedding_vec = embedding[np.newaxis]
            self.embedding_vars = np.zeros((1))
            if self.query_embeddings:
                self.pdist = self._agent._get_pairwise_dist(self.embedding_vec, aggregate=None).numpy()
            elif self.cache_pdist:
                self.pdist = self.get_cached_pairwise_dist(np.array([cache_index]), np.array([cache_index]))
            else:
                self.pdist = self._agent._get_pairwise_dist(self.rb_vec, aggregate=None).numpy()
            if self.cache_pdist:
                self.cache_indices = np.array([cache_index])

        # Merge with existing node or create new node
        else:
            # Localize to nearest neighbors in embedding space
            embedding_neighbors = np.arange(len(self.embedding_vec))[
                self.embedding_consistency(embedding, self.embedding_vec)]

            # Get maximum distances (i.e., minimum qvalues)
            pdist_combined, observation_to_rb_combined, rb_to_observation_combined, observation_to_rb, rb_to_observation = \
                self.get_distances_to_and_from(self.query_embeddings, self.cache_pdist, observation=observation,
                                               embedding=embedding, cache_index=cache_index)

            # Try to merge with a neighbor based on qvalue consistency
            merged = False
            for neighbor in embedding_neighbors:
                # Merge if qvalues are consistent
                if self.qvalue_consistency(neighbor, pdist_combined, observation_to_rb_combined,
                                           rb_to_observation_combined):
                    difference_from_avg = embedding - self.embedding_vec[neighbor]
                    self.embedding_vec[neighbor] = self.embedding_vec[neighbor] + beta * difference_from_avg
                    self.embedding_vars[neighbor] = (1 - beta) * (
                            self.embedding_vars[neighbor] + beta * np.sum(difference_from_avg ** 2))
                    merged = True
                    break

            # Add node if cannot merge
            if not merged:
                # Add node to graph
                new_index = self._g.number_of_nodes()
                in_indices = np.arange(new_index)[rb_to_observation_combined < edge_cutoff]
                in_weights = rb_to_observation_combined[in_indices]
                out_indices = np.arange(new_index)[observation_to_rb_combined < edge_cutoff]
                out_weights = observation_to_rb_combined[out_indices]
                self._g.add_node(new_index)
                self._g.add_weighted_edges_from(zip(in_indices, [new_index] * len(in_indices), in_weights))
                self._g.add_weighted_edges_from(zip([new_index] * len(out_indices), out_indices, out_weights))

                # The only qvalue distance we don't yet have is the new observation to itself.
                # Can concatenate qvalues we already have to save |V|^2 qvalue query.
                # Used to update self.pdist
                if self.cache_pdist:
                    observation_to_observation = self.get_cached_pairwise_dist(np.array([cache_index]),
                                                                               np.array([cache_index]))
                else:
                    observation_to_observation = self._agent._get_pairwise_dist(observation[np.newaxis],
                                                                                observation[np.newaxis],
                                                                                aggregate=None).numpy().reshape(-1, 1,
                                                                                                                1)

                # Add node to other attributes
                self.rb_vec = np.concatenate((self.rb_vec, observation[np.newaxis]), axis=0)
                self.embedding_vec = np.concatenate((self.embedding_vec, embedding[np.newaxis]), axis=0)
                self.embedding_vars = np.append(self.embedding_vars, [0])
                self.pdist = np.concatenate((self.pdist, observation_to_rb), axis=1)
                self.pdist = np.concatenate(
                    (self.pdist, np.concatenate((rb_to_observation, observation_to_observation), axis=1)), axis=2)
                if self.cache_pdist:
                    self.cache_indices = np.append(self.cache_indices, cache_index)

        # Let embeddings occupy replay buffer
        if embedding_buffer:
            self.rb_vec = self.embedding_vec

    def get_cached_pairwise_dist(self, row_indices, col_indices):
        assert self.cache_pdist
        assert len(row_indices.shape) == len(col_indices.shape) == 1
        row_entries = row_indices.shape[0]
        col_entries = col_indices.shape[0]
        row_advanced_index = np.tile(row_indices, (col_entries, 1)).T
        col_advanced_index = np.tile(col_indices, (row_entries, 1))
        if len(self.cached_pdist.shape) == 2:
            return self.cached_pdist[row_advanced_index, col_advanced_index]
        elif len(self.cached_pdist.shape) == 3:
            return self.cached_pdist[:, row_advanced_index, col_advanced_index]
        else:
            raise RuntimeError("Cached pdist has unrecognized shape")

    def get_distances_to_and_from(self, query_embeddings, cache_pdist,
                                  observation=None, embedding=None,
                                  cache_index=None):
        if query_embeddings:
            assert embedding is not None
            observation_to_rb = self._agent._get_pairwise_dist(embedding[np.newaxis], self.embedding_vec,
                                                               aggregate=None).numpy()
            rb_to_observation = self._agent._get_pairwise_dist(self.embedding_vec, embedding[np.newaxis],
                                                               aggregate=None).numpy()
        elif cache_pdist:
            assert cache_index is not None
            observation_to_rb = self.get_cached_pairwise_dist(np.array([cache_index]), self.cache_indices)
            rb_to_observation = self.get_cached_pairwise_dist(self.cache_indices, np.array([cache_index]))
        else:
            assert observation is not None
            observation_to_rb = self._agent._get_pairwise_dist(observation[np.newaxis], self.rb_vec,
                                                               aggregate=None).numpy()
            rb_to_observation = self._agent._get_pairwise_dist(self.rb_vec, observation[np.newaxis],
                                                               aggregate=None).numpy()
        pdist_combined = np.max(self.pdist, axis=0)
        observation_to_rb_combined = np.max(observation_to_rb, axis=0).flatten()
        rb_to_observation_combined = np.max(rb_to_observation, axis=0).flatten()

        return pdist_combined, observation_to_rb_combined, rb_to_observation_combined, observation_to_rb, rb_to_observation

    def k_nearest_embeddings(self, embedding, embeddings, k):
        differences = embeddings - embedding
        norms = np.linalg.norm(differences, axis=1)
        return np.argpartition(norms, k)[:k]

    def embedding_consistency(self, embedding, embeddings, verbose=False):
        differences = embeddings - embedding
        inconsistency = np.linalg.norm(differences, axis=1)
        if verbose:
            print(f"econsistency = {inconsistency}")
        return inconsistency < self.embedding_cutoff

    def qvalue_consistency(self, neighbor_index, pdist_combined, observation_to_rb_combined, rb_to_observation_combined,
                           verbose=False):
        # Find adjacent nodes
        in_indices = np.array(list(self._g.predecessors(neighbor_index)))
        out_indices = np.array(list(self._g.successors(neighbor_index)))

        # Be conservative about merging in this edge case
        if len(in_indices) == 0 and len(out_indices) == 0:
            return False

        # Calculate qvalues with adjacent nodes
        if len(in_indices) != 0:
            existing_in_qvals = pdist_combined[in_indices, neighbor_index]
            new_in_qvals = rb_to_observation_combined[in_indices]
        else:
            existing_in_qvals = np.array([])
            new_in_qvals = np.array([])
        if len(out_indices) != 0:
            existing_out_qvals = pdist_combined[neighbor_index, out_indices]
            new_out_qvals = observation_to_rb_combined[out_indices]
        else:
            existing_out_qvals = np.array([])
            new_out_qvals = np.array([])
        existing_qvals = np.append(existing_in_qvals, existing_out_qvals)
        new_qvals = np.append(new_in_qvals, new_out_qvals)

        # Measure qvalue consistency
        qval_diffs = new_qvals - existing_qvals
        qval_inconsistency = np.linalg.norm(qval_diffs, np.inf)

        # Report qvalue consistency
        if verbose:
            print(f"qval_inconsistency = {qval_inconsistency}")
            print(f"qval_cutoff = {self.consistency_cutoff}\n")

        # Determine consistency with self.consistency_cutoff
        return qval_inconsistency < self.consistency_cutoff
