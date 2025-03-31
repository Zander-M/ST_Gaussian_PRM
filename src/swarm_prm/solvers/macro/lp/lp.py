"""

    Linear Programming solver for finding shortest paths
"""
from collections import defaultdict
import heapq

import cvxpy as cp
import numpy as np

class LP:
    def __init__(self, gaussian_prm, agent_radius,
                 starts_agent_count, goals_agent_count, num_agents,
                 time_limit=6000):
        self.gaussian_prm = gaussian_prm
        self.nodes = np.array(self.gaussian_prm.samples)
        self.starts = self.gaussian_prm.starts_idx
        self.goals = self.gaussian_prm.goals_idx
        self.num_agents = num_agents
        self.starts_agent_count = starts_agent_count
        self.goals_agent_count = goals_agent_count
        self.agent_radius = agent_radius
        self.roadmap, self.cost_dict = self.build_roadmap_graph()
        self.node_capacity = [node.get_capacity(agent_radius) for node in self.gaussian_prm.gaussian_nodes]

    def build_roadmap_graph(self):
        """
            Build graph with edge cost
        """
        graph = defaultdict(list)
        cost = defaultdict(defaultdict)
        for i, edge in enumerate(self.gaussian_prm.roadmap):
            u, v = edge
            graph[u].append(v)
            graph[v].append(u)
            cost[u][v] = self.gaussian_prm.roadmap_cost[i]
            cost[v][u] = self.gaussian_prm.roadmap_cost[i]
        return graph, cost

    def get_shortest_paths(self):
        """
            get shortest paths from starts to goals. 
            return paths and path lengths.
        """
        path_idx = []
        path_cost = []
        paths = []
        max_path_len = 0
        for start in self.starts:
            for goal in self.goals:
                path, cost= self.dijkstra(start, goal)
                path_idx.append((start, goal))
                path_cost.append(cost)
                paths.append(path)
                max_path_len = max(max_path_len, len(path))

        # pad path with goal state
        padded_paths = []
        for path in paths:
            padded_paths.append(path + [path[-1]] * (max_path_len-len(path)))
        
        return path_idx, np.array(path_cost), padded_paths
            
    def dijkstra(self, start, goal):
        """
            Dijkstra for finding shortest path
        """
        prev = {start: None}
        open_list = [(0, start)]
        distances = {start: 0}
        while open_list:
            cost, state = heapq.heappop(open_list)
            if state == goal:
                path = []
                while state is not None:
                    path.append(state)
                    state = prev[state]
                return path[::-1], distances[goal]
            for neighbor in self.roadmap[state]:
                curr_cost = cost + self.cost_dict[state][neighbor]
                if neighbor not in distances or curr_cost < distances[neighbor]:
                    distances[neighbor] = curr_cost
                    prev[neighbor] = state
                    heapq.heappush(open_list, (curr_cost, neighbor))
        return [], 0
    
    def get_solution(self):
        """
            Get solution paths
        """
        path_idx, path_cost, paths = self.get_shortest_paths()

        # Suppose T = [0, 1, ..., num_trajectories - 1]
        num_trajectories = len(path_idx)
        x = cp.Variable(num_trajectories, integer=True)  # agents per trajectory

        # Objective: minimize total cost
        cost = cp.sum(cp.multiply(path_cost, x))  # c: vector of trajectory costs

        # Build node time to trajectory
        node_time_to_traj = defaultdict(list)
        for i, path in enumerate(paths):
            for t, node in enumerate(path):
                node_time_to_traj[(node, t)].append(i)

        constraints = []

        # 1. Start location constraints
        for i, start in enumerate(self.starts):
            start_indices = [i for i in paths if paths[0] == start]
            constraints.append(cp.sum(x[start_indices]) == self.starts_agent_count[i])

        # 2. Goal location constraints
        for i, goal in enumerate(self.goals):
            goal_indices = [i for i in paths if paths[-1] == goal]
            constraints.append(cp.sum(x[goal_indices]) == self.goals_agent_count[i])

        # 3. Capacity constraints at intermediate nodes
        for node, cap in enumerate(self.node_capacity):
            for t in range(len(paths[0])):
                trajs = node_time_to_traj.get((node, t), [])
                if trajs:
                    constraints.append(cp.sum(x[trajs]) <= cap)

        # 4. Optional: x >= 0
        constraints.append(x >= 0)

        # Solve
        prob = cp.Problem(cp.Minimize(cost), constraints)
        prob.solve(solver=cp.SCIP)  # or another MILP solver
