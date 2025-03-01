"""
    DRRT Star for continuous space motion planning
    https://arxiv.org/pdf/1903.00994
    This version does not have rewiring behavior and does not use a heuristic
"""
from collections import defaultdict

import numpy as np
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment
import time

from swarm_prm.utils.gaussian_prm import GaussianPRM

class DRRT:
    def __init__(self, gaussian_prm:GaussianPRM, num_agents, agent_radius,
                 goal_state_prob=0.1, max_time=6000, iterations=10):
        """
            We use the same roadmap for multiple agents. If a Gaussian node
            does not exceed its capacity, we do not consider it a collision.

            State is represented as a list of agent node indices.
        """
        self.gaussian_prm = gaussian_prm
        self.nodes = np.array(self.gaussian_prm.samples)
        self.num_agents = num_agents
        self.roadmap = self.gaussian_prm.map
        self.roadmap_neighbors = self.build_neighbors()
        self.max_time = max_time
        self.iterations = iterations
        self.agent_radius = agent_radius
        self.goal_state_prob = goal_state_prob

        # Initialize problem instance
        self.start_agent_count = [int(w*self.num_agents) for w in self.gaussian_prm.starts_weight]
        self.goal_agent_count = [int(w*self.num_agents) for w in self.gaussian_prm.goals_weight]

        self.goal_state = self.get_assignment()

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

        self.visited_states = {self.current_agent_node_idx}
        
        # DRRT structure 

        self.cost = {self.current_agent_node_idx:0} # cost
        self.tree = {self.current_agent_node_idx: None} # parent

    def build_neighbors(self):
        """
            Get neighbor states
        """
        graph = defaultdict(list)

        for edge in self.gaussian_prm.roadmap:
            u, v = edge
            graph[u].append(v)
            graph[v].append(u)
        
        for v in range(len(self.nodes)):
            graph[v].append(v) # add wait edges

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
        
    def expand(self):
        """
            Expand DRRT 
        """
        q_rand = np.random.randint(0, len(self.nodes), size=(self.num_agents))
        v_near = self.nearest_neighbor(q_rand)
        v_new = self.Od(v_near, q_rand)
        if v_new not in self.visited_states\
            and self.verify_node(v_new)\
            and self.verify_connect(v_near, v_new):

            self.visited_states.add(v_new) # add vertex
            self.tree[v_new] = v_near # type:ignore # add edge
        
    def nearest_neighbor(self, q_rand):
        """
            Find Nearest Neighbor in the tree
        """
        min_dist = float("inf")
        min_state = None 
        for state in self.visited_states:
            positions = np.array([self.nodes[node_idx] for node_idx in state])
            random_positions = np.array([self.nodes[node_idx] for node_idx in q_rand])
            dist = np.sum(np.linalg.norm(random_positions-positions, axis=1))
            if dist < min_dist:
                min_dist = dist
                min_state = state
        return min_state

    def get_cost(self, node):
        """
            Compute cost of node
            TODO: update cost
        """
    
    def Od(self, v_near, q_rand):
        """
            Oracle steering function
        """
        next_state = []
        for agent in range(self.num_agents):
            if v_near[agent] == q_rand[agent]:
                next_state.append(v_near[agent]) # if samples the same state, just wait.
                continue
            current_pos = self.nodes[v_near[agent]]
            diff = self.nodes[q_rand[agent]] - current_pos  
            norm_diff = np.linalg.norm(diff)
            random_dir_vec = np.divide(diff, norm_diff, out=np.zeros_like(diff), where=(norm_diff != 0))
            neighbors = self.roadmap_neighbors[v_near[agent]]
            cos_sim = []
            neighbor_ids = neighbors
            neighbor_vecs = self.nodes[neighbor_ids] - current_pos
            norms = np.linalg.norm(neighbor_vecs, axis=1, keepdims=True)
            neighbor_unit_vecs = neighbor_vecs / np.where(norms == 0, 1, norms)
            cos_sim = neighbor_unit_vecs @ random_dir_vec
            next_idx = neighbor_ids[np.argmax(cos_sim)]
            next_state.append(next_idx)
        return tuple(next_state)
    
    def get_parent(self, nodes):
        """
            Get node parent
        """
        return self.tree[nodes]
    
    def get_distance(self, node1, node2):
        """
            Get node distance
        """

    def get_assignment(self):
        """
            Get goal assignment
        """
        starts = []
        for i, g_node in enumerate(self.gaussian_prm.starts):
            starts += [g_node.get_mean()] * self.start_agent_count[i] 

        goals = []
        goals_idx = []
        for i, g_node in enumerate(self.gaussian_prm.goals):
            goals += [g_node.get_mean()] * self.goal_agent_count[i]
            goals_idx += [self.gaussian_prm.goals_idx[i]] * self.goal_agent_count[i]

        distance_matrix = cdist(starts, goals)
        _, col_ind = linear_sum_assignment(distance_matrix)
        goal_state = tuple([goals_idx[idx] for idx in col_ind])
        return goal_state

    def get_solution(self):
        """
            Get solution per agent
        """
        start_time = time.time()
        while time.time() - start_time < self.max_time:
            for _ in range(self.iterations):
                self.expand()
            if self.goal_state in self.visited_states:
                path = self.connect_to_target(self.goal_state)
                print("Found solution")
                return path, self.cost
        print("exceeded run time")
        print(self.visited_states)
        return None, None

    def verify_node(self, node):
        """
            Verify if new state is valid
            Return false if node capacity exceeded
        """
        count = np.zeros(len(self.nodes))
        for i in node:
            count[i] += 1
        return all(self.node_capacity - count) # guarantee if all node capacity > agent count
        

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