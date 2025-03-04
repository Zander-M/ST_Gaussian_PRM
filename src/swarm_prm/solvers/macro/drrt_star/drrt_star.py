"""
    DRRT Star for continuous space motion planning
    https://arxiv.org/pdf/1903.00994
"""
from collections import defaultdict

import numpy as np
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment
import time

from swarm_prm.utils.gaussian_prm import GaussianPRM
from swarm_prm.solvers.macro.drrt_star import johnsons_algorithm

class DRRT_Star:
    def __init__(self, gaussian_prm:GaussianPRM, num_agents, agent_radius,
                 max_time=30, iterations=1):
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
        self.shortest_distance = johnsons_algorithm(self.roadmap_neighbors)
        self.max_time = max_time
        self.iterations = iterations
        self.agent_radius = agent_radius

        # Initialize problem instance
        self.start_agent_count = [int(w*self.num_agents) for w in self.gaussian_prm.starts_weight]
        self.goal_agent_count = [int(w*self.num_agents) for w in self.gaussian_prm.goals_weight]

        # Finding target assignments
        self.start_state, self.goal_state = self.get_assignment()

        self.node_capacity = np.array([node.get_capacity(self.agent_radius) for node in self.gaussian_prm.gaussian_nodes])
        self.current_node_capacity = [0 for _ in range(len(self.gaussian_prm.samples))]
        for i, start_idx in enumerate(self.gaussian_prm.starts_idx):
            self.current_node_capacity[start_idx] = self.start_agent_count[i]
        
        # initialize agent location
        self.start_agent_node_idx = []
        for i, start_idx in enumerate(self.gaussian_prm.starts_idx):
            self.start_agent_node_idx += [start_idx] * self.start_agent_count[i]
        self.start_agent_node_idx = tuple(self.start_agent_node_idx)
        self.visited_states = {self.start_agent_node_idx}
        
        # DRRT structure 
        self.parent = {self.start_agent_node_idx: None} # parent
        self.best_path = None
        self.best_path_cost = float("inf")

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
    
    def get_neighbors(self, v_new):
        """
            Get composite state neighbors.
            Adj(v_new, G_hat) cap T
        """
        neighbors = []
        neighbor_candidates = []
        for v_i in v_new:
            neighbor_candidates.append(set([v[0] for v in self.roadmap_neighbors[v_i]]))
        
        for state in self.visited_states:
            if all(v in candidate for v, candidate in zip(state, neighbor_candidates)):
                neighbors.append(state)
            
        return neighbors

    def choose_best_neighbor(self, neighbors, v_new):
        """
            argmin cost(V)  cost(L(V, Vnew))
        """
        v_best = None
        cost = float("inf")
        for neighbor in neighbors:

            # reject invalid states
            if self.verify_node(v_new) \
            and self.verify_connect(neighbor, v_new):
                v_new_cost = self.get_cost(neighbor) + self.get_distance(neighbor, v_new)
                if cost > v_new_cost:
                    cost = v_new_cost 
                    v_best = neighbor
        return v_best, cost

    def get_cost(self, state):
        """
            Get cost from current state to start state
        """
        if state == self.start_state:
            return 0
        if state not in self.visited_states:
            return float("inf")
        else: 
            return self.get_cost(self.parent[state]) + self.get_distance(self.parent[state], state)

    def get_heuristic(self, state):
        """
            Get Heuristic for macro states. We use actual path cost
        """
        return np.sum([self.shortest_distance[(v1, v2)] for v1, v2 in zip(state, self.goal_state)])

    def connect_to_target(self, goal_state):
        """
            Connect currect tree to target
        """
        if goal_state not in self.visited_states:
            return None, float("inf")
        path = []
        curr_state = goal_state
        while curr_state is not None:
            path.append(curr_state)
            curr_state = self.parent[curr_state]
        return path[::-1], self.get_cost(goal_state)
        
    def expand_drrt_star(self, v_last):
        """
            Expand DRRT Star
        """
        if v_last is None:
            q_rand = np.random.randint(0, len(self.nodes), size=self.num_agents)
            v_near = self.nearest_neighbor(q_rand)
        else:
            q_rand = self.goal_state
            v_near = v_last
        v_new = self.Id(v_near, q_rand)
        neighbors = self.get_neighbors(v_new)
        v_best, v_new_cost = self.choose_best_neighbor(neighbors, v_new)

        if v_best is None:
            return None

        if self.get_cost(v_new) > self.best_path_cost:
            return None

        if v_new not in self.visited_states:
            # v_new verified in get_neighbors
            self.visited_states.add(v_new) # add vertex
            self.parent[v_new] = v_best # type:ignore # add edge
        else: 
            # Rewire v_new if cost is lower
            if v_new_cost < self.get_cost(v_new):
                self.parent[v_new] = v_best
        
        # Rewire neighbors
        for v in neighbors:
            transition_cost = self.get_cost(v_new) + self.get_distance(v_new, v)
            if  transition_cost < self.get_cost(v):
                self.parent[v] = v_new # type: ignore
        if self.get_heuristic(v_new) < self.get_heuristic(v_best):
            return v_new
        else: 
            return None 

    def nearest_neighbor(self, q_rand):
        """
            Find Nearest Neighbor in the tree
        """
        min_dist = float("inf")
        min_state = None 
        for state in self.visited_states:
            dist = np.sum([self.shortest_distance[(v1, v2)] for v1, v2 in zip(q_rand, state)])
            if dist < min_dist:
                min_dist = dist
                min_state = state
        return min_state

    def Id(self, v_near, q_rand):
        """
            Oracle steering function
        """
        next_state = [None]*self.num_agents
        for agent in range(self.num_agents):
            neighbors = [neighbor[0] for neighbor in self.roadmap_neighbors[v_near[agent]]]
            if q_rand[agent] == self.goal_state[agent]:
                heuristics = np.array([self.shortest_distance[(neighbor, self.goal_state[agent])] for neighbor in neighbors])
                next_state[agent] = neighbors[np.argmin(heuristics)]
            else: 
                next_state[agent] = neighbors[np.random.randint(len(neighbors))]
        return tuple(next_state)
    
    def get_distance(self, node1, node2):
        """
            Get node distance
        """
        return np.sum([self.shortest_distance[(v1, v2)] for v1, v2 in zip(node1, node2)])

    def get_assignment(self):
        """
            Get goal assignment
        """
        starts = []
        starts_idx = []
        for i, g_node in enumerate(self.gaussian_prm.starts):
            starts += [g_node.get_mean()] * self.start_agent_count[i] 
            starts_idx += [self.gaussian_prm.starts_idx[i]] * self.start_agent_count[i]
        goals = []
        goals_idx = []
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
        start_time = time.time()
        v_last = self.start_state
        while time.time() - start_time < self.max_time:
            for _ in range(self.iterations):
                v_last = self.expand_drrt_star(v_last)
            
            path, cost = self.connect_to_target(self.goal_state)
            if path is not None and cost < self.best_path_cost:
                print("Cost: ", cost)
                self.best_path = path
                self.best_path_cost = cost
        return self.best_path, self.best_path_cost

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
            if node1[i] == node2[i]:
                continue
            transition = (node1[i], node2[i])
            reverse_transition = (node2[i], node1[i])
            if reverse_transition in edge:
                return False
            edge.add(transition)
        return True