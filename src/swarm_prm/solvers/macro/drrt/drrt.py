"""
    DRRT for continuous space motion planning
    https://arxiv.org/pdf/1903.00994
    This version does not have rewiring behavior and does not use a heuristic 
"""
import time
from collections import Counter

from scipy.spatial.distance import cdist 
from scipy.optimize import linear_sum_assignment
import hnswlib # For fast NN check
import numpy as np

from swarm_prm.solvers.macro import MacroSolverBase, register_solver

@register_solver("DRRTSolver")
class DRRTSolver(MacroSolverBase):
    def init_solver(self, **kwargs):
        """
            We use the same roadmap for multiple agents. If a Gaussian node
            does not exceed its capacity, we do not consider it a collision.

            State is represented as a list of agent node indices.
        """

        self.iterations = kwargs.get("iterations", 1) # iterations per goal state check

        # Define goal state
        starts_loc = [self.nodes[idx] for idx in self.starts_idx]
        goals_loc = [self.nodes[idx] for idx in self.goals_idx]
        dists = cdist(starts_loc, goals_loc)
        _, self.goal_state= linear_sum_assignment(dists)
        self.goal_state = tuple(self.goal_state)

        # initialize agent location
        self.start_state = []
        for i, start_idx in enumerate(self.starts_idx):
            self.start_state += [start_idx] * self.starts_agent_count[i]
        self.start_state = tuple(self.start_state)

        # tree structure
        self.visited_states = [self.start_state]
        self.visited_states_idx = {self.start_state:0}

        # DRRT structure 
        self.cost = {self.start_state:0} # cost
        self.tree = {self.start_state: None} # parent

        # Fast NN check
        self.nn = hnswlib.Index(space="l2", dim=self.num_agents*2) # 2D agents
        self.nn.init_index(max_elements=1000000)
        start_location = np.array([[self.nodes[idx] for idx in self.start_state]]).flatten()
        self.nn.add_items(start_location, [0])

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

        # accroding to the paper, random state only depends of the dimension of
        # each agents' configuration space.

        xs = np.random.uniform(0, self.obstacle_map.width, self.num_agents)
        ys = np.random.uniform(0, self.obstacle_map.height, self.num_agents)
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
            new_state_location = np.array([[self.nodes[idx] for idx in v_new]]).flatten()
            self.nn.add_items(new_state_location, [len(self.visited_states)-1])
            self.tree[v_new] = v_near # type:ignore # add edge
        verify_time = time.time() - verify_time
        return nn_time, Od_time, verify_time
        
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

    def Od(self, v_near, q_rand):
        """
            Oracle steering function
        """
        next_state = []
        for agent in range(self.num_agents):
            if v_near[agent] in self.goals_idx: 
                # Agent wait at goal
                next_state.append(v_near[agent])
                continue
            current_pos = self.nodes[v_near[agent]]
            diff = q_rand[agent] - current_pos  
            norm_diff = np.linalg.norm(diff)
            random_dir_vec = np.divide(diff, norm_diff, out=np.zeros_like(diff), where=(norm_diff != 0))
            neighbors = self.roadmap[v_near[agent]]
            neighbor_ids = [neighbor for neighbor in neighbors]
            neighbor_vecs = self.nodes[neighbor_ids] - current_pos
            norms = np.linalg.norm(neighbor_vecs, axis=1, keepdims=True)
            neighbor_unit_vecs = neighbor_vecs / np.where(norms == 0, 1, norms)
            cos_sim = neighbor_unit_vecs @ random_dir_vec.T
            next_idx = neighbor_ids[np.argmax(cos_sim)]
            next_state.append(next_idx)
        return tuple(next_state)
    
    def solve(self):
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
            if self.goal_state in self.visited_states:
                path = self.connect_to_target(self.goal_state)
                print("Found solution")
                return {
                    "success": True,
                    "path": path,
                    "timestep": len(path),
                    "cost": self.cost
                }
            if iteration % 1000 == 0:
                print("Iteration:", iteration)
                print("nearest neighbor time: ", nn_time)
                print("Od time: ", Od_time)
                print("Verify time: ", verify_time)
        print("exceeded run time")
        print(self.visited_states)
        return {
            "success": False
        }

    #   Connection verification
    def verify_node(self, node):
        """
            Verify if new state is valid
            Return false if node capacity exceeded
        """
        if len(node) != self.num_agents:
            return False  # quick sanity check

        state_counts = Counter(node)
        for state, count in state_counts.items():
            if count > self.node_capacity[state]:
                return False
        return True

    def verify_connect(self, node1, node2):
        """
            Verify if two states can be connected
            Return false if agents moving in different directions 
        """
        return True
        edge = set()
        for i in range(self.num_agents):
            if (node2[i], node1[i]) not in edge:
                edge.add((node1[i], node2[i]))
            else:
                return False
        return True