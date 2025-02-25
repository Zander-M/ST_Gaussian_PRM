"""
    DRRT Star for continuous space motion planning
    https://arxiv.org/pdf/1903.00994
    This version does not have rewiring behavior but has a heuristic
"""
from collections import defaultdict, deque

import numpy as np
from scipy.spatial import KDTree
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment
import time

np.random.seed(0)

from swarm_prm.solvers.utils.gaussian_prm import GaussianPRM

class DRRTStar:
    def __init__(self, gaussian_prm:GaussianPRM, agent_radius, num_agents,
                max_time=6000, iteration=100):
        """
            We use the same roadmap for multiple agents. If a Gaussian node
            does not exceed its capacity, we do not consider it a collision.

            State is represented as a list of agent node indices.
        """
        self.gaussian_prm = gaussian_prm
        self.nodes = np.array(self.gaussian_prm.samples)
        self.num_agents = num_agents
        self.kd_tree = KDTree(self.nodes)
        self.roadmap = self.gaussian_prm.map
        self.roadmap_neighbors = self.build_neighbors()
        self.max_time = max_time
        self.agent_radius = agent_radius
        self.iteration = iteration

        # Initialize problem instance
        self.start_agent_count = [int(w*self.num_agents) for w in self.gaussian_prm.starts_weight]
        self.goal_agent_count = [int(w*self.num_agents) for w in self.gaussian_prm.goals_weight]
        self.start_state, self.goal_state = self.get_assignment()

        # Compute Heuristic
        self.heuritics = self.build_heuristic()

        # Finding target assignments

        self.node_capacity = [node.get_capacity(self.agent_radius) for node in self.gaussian_prm.gaussian_nodes]
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
        return graph
    
    def build_heuristic(self):
        """
            Compute heuristics 
        """
        heuristics = {}
        for goal_idx in enumerate(self.gaussian_prm.goals_idx):
            heuristics[goal_idx] = {goal_idx: {"parent": None, "val": 0}}
            open_list = deque([goal_idx])
            while len(open_list) > 0:
                new_node = open_list.popleft()
                for neighbor in self.roadmap_neighbors[new_node]:
                    if neighbor not in heuristics[goal_idx]:
                        heuristics[goal_idx][neighbor]["parent"] = new_node 
                        heuristics[goal_idx][neighbor]["val"] = heuristics[goal_idx][new_node]["val"] + 1
                        open_list.append(neighbor)
        return heuristics

    def random_sample(self):
        """
            Return a random composite state
        """
        return np.random.randint(0, len(self.nodes), size=(self.num_agents))

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
        
    def expand_drrt_star(self, v_last=None):
        """
            Expand DRRT 
        """
        if v_last is None:
            q_rand = self.random_sample()
            v_near = self.nearest_neighbor(q_rand)
        else:
            # Keep current frontier
            q_rand = np.array([self.nodes[idx] for idx in self.goal_state])
            v_near = v_last

        v_new = self.Id(v_near, q_rand)

        if v_new not in self.visited_states:
            self.visited_states.add(v_new) # add vertex
            self.tree[v_new] = v_near # type:ignore # add edge
        
    def nearest_neighbor(self, random_state):
        """
            Find Nearest Neighbor in the tree
        """
        min_dist = float("inf")
        min_state = None 
        for state in self.visited_states:
            positions = np.array([self.nodes[node_idx] for node_idx in state])
            dist = np.sum(np.linalg.norm(random_state-positions, axis=1))
            if dist < min_dist:
                min_dist = dist
                min_state = state
        return min_state

    def get_cost(self, node):
        """
            Compute cost of node
            TODO: update cost
        """
    
    def Id(self, v_near, q_rand):
        """
            Oracle steering function
        """
        next_state = []
        for agent in range(self.num_agents):
            if q_rand[agent] == self.goal_state[agent]:

                # Follow shortest path to goal
                neighbors = self.roadmap_neighbors[v_near[agent]]
                h_values = [self.heuritics[self.goal_state[agent]][neighbor]["val"] for neighbor in neighbors]
                next_state.append(neighbors[np.argmin(h_values)])
            else:
                # Select random next state
                current_pos = self.gaussian_prm.samples[v_near[agent]]
                diff = q_rand[agent] - current_pos  
                norm_diff = np.linalg.norm(diff)
                random_dir_vec = np.divide(diff, norm_diff, out=np.zeros_like(diff), where=(norm_diff != 0))
    
                neighbors = self.roadmap_neighbors[v_near[agent]]
                cos_sim = []
                neighbor_ids = neighbors[1:]
                neighbor_vecs = self.nodes[neighbor_ids] - current_pos
                norms = np.linalg.norm(neighbor_vecs, axis=1, keepdims=True)
                neighbor_unit_vecs = neighbor_vecs / np.where(norms == 0, 1, norms)
                cos_sim = neighbor_unit_vecs @ random_dir_vec
                next_idx = neighbor_ids[np.argmax(cos_sim)]
                next_state.append(next_idx)
        return tuple(next_state)
    
    def get_neighbor_states(self, state):
        """
            Get Neighbor states from the 
        """
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

        starts= []
        starts_idx = []
        
        goals = []
        goals_idx = []

        for i, g_node in enumerate(self.gaussian_prm.starts):
            starts += [g_node.get_mean()] * self.start_agent_count[i]
            starts_idx += [self.gaussian_prm.starts_idx[i]] * self.start_agent_count[i]

        for i, g_node in enumerate(self.gaussian_prm.goals):
            goals += [g_node.get_mean()] * self.goal_agent_count[i]
            goals_idx += [self.gaussian_prm.goals_idx[i]] * self.goal_agent_count[i]

        distance_matrix = cdist(starts, goals)
        row_ind, col_ind = linear_sum_assignment(distance_matrix)

        start_state = tuple([starts_idx[idx] for idx in row_ind])
        goal_state = tuple([goals_idx[idx] for idx in col_ind])
        return start_state, goal_state

    def get_solution(self):
        """
            Get solution per agent
        """

        v_last = self.start_state
        start_time = time.time()
        while time.time() - start_time < self.max_time:
            for _ in range(self.iteration):
                v_last = self.expand_drrt_star(v_last)

            if self.goal_state in self.visited_states:
                path = self.connect_to_target(self.goal_state)
                print("Found solution")
                return path, self.cost

        print("exceeded run time")
        print(self.visited_states)
        return None, None

    def verify_node(self, state):
        """
            Verify if the new state is valid
        """
        pass

    def verify_connect(self, state_1, state_2):
        """
            Verify if two states can be connected
        """
        pass