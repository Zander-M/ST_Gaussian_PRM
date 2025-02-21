"""
    DRRT for continuous space motion planning
    https://arxiv.org/pdf/1903.00994
    This version does not have rewiring behavior
"""

import numpy as np
from scipy.spatial import KDTree
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment

from swarm_prm.solvers.utils.gaussian_prm import GaussianPRM

class DRRT:
    def __init__(self, gaussian_prm:GaussianPRM, num_agents, agent_radius, 
                 connect_radius=10, max_iter=30000):
        """
            We use the same roadmap for multiple agents. If a Gaussian node
            does not exceed its capacity, we do not consider it a collision.

            State is represented as a list of agent node indices.
        """
        self.gaussian_prm = gaussian_prm
        self.nodes = self.gaussian_prm.samples
        self.num_agents = num_agents
        self.kd_tree = KDTree(self.nodes)
        self.roadmap = self.gaussian_prm.map
        self.max_iter = max_iter
        self.agent_radius = agent_radius
        self.connect_radius = connect_radius

        # Initialize problem instance
        self.start_agent_count = [int(w*self.num_agents) for w in self.gaussian_prm.starts_weight]
        self.goal_agent_count = [int(w*self.num_agents) for w in self.gaussian_prm.goals_weight]

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

        self.visited_states = [self.current_agent_node_idx]
        
        # DRRT structure 

        self.cost = {self.current_agent_node_idx:0} # cost
        self.tree = {self.current_agent_node_idx: None} # parent

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
        q_rand = np.random.rand(self.num_agents, 2) \
            * np.array([self.roadmap.width, self.roadmap.height])
        v_near = self.nearest_neighbor(q_rand)
        v_new = self.Id(v_near, q_rand)
        if v_new not in self.visited_states:
            self.visited_states.append(v_new) # add vertex
            self.tree[v_new] = v_near # type:ignore # add edge
        
    def nearest_neighbor(self, random_state):
        """
            Find Nearest Neighbor in the tree
        """
        min_dist = float("inf")
        min_idx = -1
        for i, state in enumerate(self.visited_states):
            positions = np.array([self.nodes[node_idx] for node_idx in state])
            dist = np.sum(np.linalg.norm(random_state-positions, axis=1))
            if dist < min_dist:
                min_dist = dist
                min_idx = i
        return self.visited_states[min_idx]

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
            current_pos = self.gaussian_prm.samples[v_near[agent]]
            random_dir_vec = (q_rand[agent] - current_pos)  
            random_dir_vec = random_dir_vec/np.linalg.norm(random_dir_vec)
            neighbors = self.kd_tree.query_ball_point(current_pos, self.connect_radius)
            cos_sim = []
            for neighbor in neighbors[1:]:
                dir_vec = self.nodes[neighbor] - current_pos
                dir_vec = dir_vec / np.linalg.norm(dir_vec)
                cos_sim.append(dir_vec @ random_dir_vec)
            next_idx = neighbors[np.argmax(cos_sim)]
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
            starts.append(g_node.get_mean() * self.start_agent_count[i]) 

        goals = []
        for i, g_node in enumerate(self.gaussian_prm.goals):
            goals.append(g_node.get_mean() * self.goal_agent_count[i])

        distance_matrix = cdist(starts, goals)
        _, col_ind = linear_sum_assignment(distance_matrix)
        goal_state = tuple([self.gaussian_prm.goals_idx[idx] for idx in col_ind])
        return goal_state

    def get_solution(self):
        """
            Get solution per agent
        """
        current_state = {
            "parent": None,
            "state": self.current_agent_node_idx,
            "cost": 0
        }

        goal_state = self.get_assignment()
        print("Goal", goal_state)

        for _ in range(self.max_iter):
            self.expand()
            if goal_state in self.visited_states:
                path = self.connect_to_target(goal_state)
                return path, self.cost
        print("exceeded max iter")
        print(self.visited_states)
        return None, None

    def verify_node(self, node):
        """
            Verify if the new state is valid
        """
        pass

    def verify_connect(self, node1, node2):
        """
            Verify if two states can be connected
        """
        pass