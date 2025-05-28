"""
    DRRT Star for continuous space motion planning
    https://arxiv.org/pdf/1903.00994
"""
from collections import defaultdict, Counter

import hnswlib # For fast NN check
import numpy as np
import time


from swarm_prm.solvers.macro import register_solver, MacroSolverBase
from swarm_prm.utils.gaussian_prm import GaussianPRM
from swarm_prm.utils.johnson import johnsons_algorithm

@register_solver("DRRTStarSolver")
class DRRTStarSolver(MacroSolverBase):
    def init_solver(self, **kwargs):
        self.iterations = kwargs.get("iterations", 10)
        self.nodes = np.array(self.gaussian_prm.samples)
        self.roadmap_neighbors = self.build_neighbors()
        self.shortest_distance = johnsons_algorithm(self.roadmap_neighbors)

        # Initialize problem instance
        self.start_state = []
        for i, start_idx in enumerate(self.starts_idx):
            self.start_state += [start_idx] * self.starts_agent_count[i]
        self.start_state = tuple(self.start_state)
        self.goal_state = {}
        for node_idx, node_count in zip(self.goals_idx, self.goals_agent_count):
            if node_count > 0:
                self.goal_state[node_idx] = node_count

        # Goal signature
        self.goal_state = tuple(sorted(self.goal_state.items()))

        # Initialize node capacities
        self.node_capacity = np.array([node.get_capacity(self.agent_radius) for node in self.gaussian_prm.gaussian_nodes])

        # Verify if instance is feasible
        for i, start in enumerate(self.starts_idx):
            assert self.node_capacity[start] >= self.starts_agent_count[i], \
                "Start capacity smaller than required."

        for i, goal in enumerate(self.goals_idx):
            assert self.node_capacity[goal] >= self.goals_agent_count[i], \
                "Goal capacity smaller than required."

        # DRRT star structure 
        self.visited_states = [self.start_state]
        self.visited_states_idx = {self.start_state: 0}
        self.visited_states_signatures = {self.get_state_signature(self.start_state):0}
        self.tree = {self.start_state: None} # parent

        self.best_path = None
        self.best_path_cost = float("inf")

        # Fast NN check
        self.nn = hnswlib.Index(space="l2", dim=self.num_agents*2) # 2D agents
        self.nn.init_index(max_elements=1000000)
        start_location = np.array([[self.nodes[idx] for idx in self.start_state]]).flatten()
        self.nn.add_items(start_location, [0])

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
            return self.get_cost(self.tree[state]) + self.get_distance(self.tree[state], state)

    def get_heuristic(self, state):
        """
            Get Heuristic for macro states. We use actual path cost
        """
        return np.sum([self.shortest_distance[(v1, v2)] for v1, v2 in zip(state, self.goal_state)])
    
    def get_state_signature(self, state):
        """
            Get state signature for goal state check
        """
        signature = Counter(state)
        hashable_signature = tuple(sorted(signature.items()))
        return hashable_signature

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
            curr_state = self.tree[curr_state]
        return path[::-1], self.get_cost(goal_state)
        
    def expand_drrt_star(self, v_last):
        """
            Expand DRRT Star
        """
        if v_last is None:
            xs = np.random.uniform(0, self.obstacle_map.width, self.num_agents)
            ys = np.random.uniform(0, self.obstacle_map.height, self.num_agents)
            q_rand = np.column_stack((xs, ys)) 
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
            self.visited_states.append(v_new) # add vertex
            self.visited_states_idx[v_new] = len(self.visited_states) - 1
            self.visited_states_signatures[self.get_state_signature(v_new)] = len(self.visited_states) - 1
            self.parent[v_new] = v_best # type:ignore # add edge
        else: 
            # Rewire v_new if cost is lower
            if v_new_cost < self.get_cost(v_new):
                self.tree[v_new] = v_best
        
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
            We first check both the buffer and the KDTree
        """
        # For efficiency reasons, we compute sum of squared distance instead
        # of sum of distance using KDTree. The KD Tree is updated periodically.
        # This does not affect the probabilistic completeness

        flat_q_rand = q_rand.flatten()
        labels, distances = self.nn.knn_query(flat_q_rand, k=1)
        nearest_idx = labels[0][0]
        return self.visited_states[nearest_idx]

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

    def solve(self):
        """
            Get solution per agent
        """
        start_time = time.time()
        v_last = self.start_state
        while time.time() - start_time < self.time_limit:
            for _ in range(self.iterations):
                v_last = self.expand_drrt_star(v_last)
            
            path, cost = self.connect_to_target(self.goal_state)
            if path is not None and cost < self.best_path_cost:
                print("Cost: ", cost)
                self.best_path = path
                self.best_path_cost = cost
        solution = {
            "success": True,
            "path" : self.best_path,
            "cost": self.best_path_cost
        }
        return solution

    def verify_node(self, node):
        """
            Verify if new state is valid
            Return false if multiple agents at same node
        """
        states = set()
        for state in node:
            if state in states:
                return False
            states.add(state)
        return True

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