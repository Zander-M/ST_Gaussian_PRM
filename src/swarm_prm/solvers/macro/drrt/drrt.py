"""
    DRRT for continuous space motion planning
    https://arxiv.org/pdf/1903.00994
    This version does not have rewiring behavior and does not use a heuristic 
"""
from collections import defaultdict, Counter

import numpy as np
from scipy.spatial import KDTree
import time

from swarm_prm.utils.gaussian_prm import GaussianPRM

class DRRT:
    def __init__(self, gaussian_prm:GaussianPRM, agent_radius,
                 starts_agent_count, goals_agent_count, num_agents,
                 num_random_sample = 5000,
                 time_limit=6000, iterations=10):
        """
            We use the same roadmap for multiple agents. If a Gaussian node
            does not exceed its capacity, we do not consider it a collision.

            State is represented as a list of agent node indices.
        """
        self.gaussian_prm = gaussian_prm
        self.nodes = np.array(self.gaussian_prm.samples)
        self.num_agents = num_agents
        self.roadmap = self.gaussian_prm.raw_map
        self.roadmap_neighbors = self.build_neighbors()
        self.time_limit = time_limit
        self.iterations = iterations
        self.agent_radius = agent_radius

        # Initialize problem instance
        self.start_agent_count = starts_agent_count
        self.goal_agent_count = goals_agent_count
        self.goal_state = {}
        for node_idx, node_count in zip(self.gaussian_prm.goals_idx, self.goal_agent_count):
            if node_count > 0:
                self.goal_state[node_idx] = node_count
        # Goal signature
        self.goal_state = tuple(sorted(self.goal_state.items()))
        print(self.goal_state)

        # Finding target assignments
        self.node_capacity = np.array([node.get_capacity(self.agent_radius) for node in self.gaussian_prm.gaussian_nodes])
        self.current_node_capacity = [0 for _ in range(len(self.gaussian_prm.samples))]
        for i, start_idx in enumerate(self.gaussian_prm.starts_idx):
            self.current_node_capacity[start_idx] = self.start_agent_count[i]
        
        # initialize agent location
        self.current_agent_node_idx = []
        for i, start_idx in enumerate(self.gaussian_prm.starts_idx):
            self.current_agent_node_idx += [start_idx] * self.start_agent_count[i]
        self.current_agent_node_idx = tuple(self.current_agent_node_idx)

        # tree structure
        self.visited_states = [self.current_agent_node_idx]
        self.visited_states_idx = {self.current_agent_node_idx:0}
        self.visited_states_signatures = {self.get_state_signature(self.current_agent_node_idx):0}
        self.visited_states_location = np.array([[self.nodes[idx] for idx in self.current_agent_node_idx]])

        # DRRT structure 
        self.cost = {self.current_agent_node_idx:0} # cost
        self.tree = {self.current_agent_node_idx: None} # parent

        # KDTree buffer
        self.kd_tree_buffer_size = 100
        self.kd_tree_buffer = []

    def build_neighbors(self):
        """
            Get neighbor states
        """
        graph = defaultdict(list)

        for i, edge in enumerate(self.gaussian_prm.roadmap):
            u, v = edge
            graph[u].append((v, self.gaussian_prm.roadmap_cost[i]))
            graph[v].append((u, self.gaussian_prm.roadmap_cost[i]))
        
        for v in range(len(self.nodes)):
            graph[v].append((v, 0)) # add wait edges

        return graph

    def connect_to_target(self, goal_state):
        """
            Connect currect tree to target
        """
        path = []
        curr_state = goal_state
        while curr_state is not None:
            path.append(curr_state)
            curr_state = self.tree[curr_state]
        return path[::-1]
    
    def compute_pairwise_distance(self):
        """
            Compute pairwise distance betwe
        """
        
    def expand(self):
        """
            Expand DRRT 
        """
        xs = np.random.uniform(0, self.roadmap.width, self.num_agents)
        ys = np.random.uniform(0, self.roadmap.height, self.num_agents)
        q_rand = np.column_stack((xs, ys)) 
        nn_time = time.time()
        v_near = self.nearest_neighbor(q_rand)
        nn_time = time.time() - nn_time
        Od_time = time.time()
        v_new = self.Od(v_near, q_rand)
        Od_time = time.time() - Od_time
        verify_time = time.time()
        if v_new not in self.visited_states\
            and self.verify_node(v_new)\
            and self.verify_connect(v_near, v_new):
            self.visited_states.append(v_new)
            self.visited_states_idx[v_new] = len(self.visited_states) - 1 # add vertex
            self.visited_states_signatures[self.get_state_signature(v_new)] = len(self.visited_states) - 1
            new_state_location = np.array([[self.nodes[idx] for idx in v_new]])
            self.visited_states_location = np.concat((self.visited_states_location, new_state_location))
            self.tree[v_new] = v_near # type:ignore # add edge
        verify_time = time.time() - verify_time
        return nn_time, Od_time, verify_time
        
    def nearest_neighbor(self, q_rand):
        """
            Find Nearest Neighbor in the tree
            We first check buffer and then check the KDTree
        """
        min_state = np.argmin(np.sum(np.linalg.norm(self.visited_states_location - q_rand, axis=2), axis=1))
        return self.visited_states[min_state]

    def Od(self, v_near, q_rand):
        """
            Oracle steering function
        """
        next_state = []
        for agent in range(self.num_agents):
            if v_near[agent] in self.gaussian_prm.goals_idx: 
                # Agent wait at goal
                next_state.append(v_near[agent])
                continue
            current_pos = self.nodes[v_near[agent]]
            diff = q_rand[agent] - current_pos  
            norm_diff = np.linalg.norm(diff)
            random_dir_vec = np.divide(diff, norm_diff, out=np.zeros_like(diff), where=(norm_diff != 0))
            neighbors = self.roadmap_neighbors[v_near[agent]]
            neighbor_ids = [neighbor[0] for neighbor in neighbors]
            neighbor_vecs = self.nodes[neighbor_ids] - current_pos
            norms = np.linalg.norm(neighbor_vecs, axis=1, keepdims=True)
            neighbor_unit_vecs = neighbor_vecs / np.where(norms == 0, 1, norms)
            cos_sim = neighbor_unit_vecs @ random_dir_vec.T
            next_idx = neighbor_ids[np.argmax(cos_sim)]
            next_state.append(next_idx)
        return tuple(next_state)
    
    def get_state_signature(self, state):
        """
            Get state signature for goal state check
        """
        signature = Counter(state)
        hashable_signature = tuple(sorted(signature.items()))
        return hashable_signature

    def get_solution(self):
        """
            Get solution per agent
        """
        start_time = time.time()
        iteration = 0
        nn_time = 0
        Od_time = 0
        verify_time = 0

        while time.time() - start_time < self.time_limit:
            iteration += 1
            for _ in range(self.iterations): # expansion before goal state check
                nn_t, Od_t, v_t = self.expand()
                nn_time += nn_t
                Od_time += Od_t
                verify_time += v_t
            if self.goal_state in self.visited_states_signatures:
                path = self.connect_to_target(self.visited_states[self.visited_states_signatures[self.goal_state]])
                print("Found solution")
                return path, self.cost
            if iteration % 1000 == 0:
                print("Iteration:", iteration)
                print("nearest neighbor time: ", nn_time)
                print("Od time: ", Od_time)
                print("Verify time: ", verify_time)
        print("exceeded run time")
        print(self.visited_states)
        return None, None

    #   Connection verification
    def verify_node(self, node):
        """
            Verify if new state is valid
            Return false if node capacity exceeded
        """
        unique, counts = np.unique(node, return_counts=True)
        return np.all(self.node_capacity[unique] >= counts)

    def verify_connect(self, node1, node2):
        """
            Verify if two states can be connected
            Return false if agents moving in different directions 
        """
        edge = set()
        for i in range(self.num_agents):
            if (node2[i], node1[i]) not in edge:
                edge.add((node1[i], node2[i]))
            else:
                return False
        return True