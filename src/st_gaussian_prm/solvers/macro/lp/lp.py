"""
    Linear Programming solver for finding shortest paths
"""
from collections import defaultdict
import heapq

import cvxpy as cp
import numpy as np

from st_gaussian_prm.solvers.macro import MacroSolverBase, register_solver

@register_solver("LPSolver")
class LPSolver(MacroSolverBase):
    def init_solver(self, **kwargs):
        """
            Solver Init
        """
        self.capacity_constraint = kwargs.get("capacity_constraint", True)

    def get_shortest_paths(self):
        """
            get shortest paths from starts to goals. 
            return paths and path lengths.
        """
        path_idx = []
        path_cost = []
        paths = []
        max_path_len = 0
        for start in self.starts_idx:
            for goal in self.goals_idx:
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

    def get_cost(self, paths):
        """
            Get average cost per agent. We use Wasserstein distance between states
            as an estimator.
        """
        cost = 0
        for path in paths:
            for u, v in zip(path[:-1], path[1:]):
                cost += self.cost_dict[u][v]
        return cost / len(paths)
    
    def solve(self):
        """
            Get solution paths
        """
        path_idx, path_cost, shortest_paths = self.get_shortest_paths()

        # Suppose T = [0, 1, ..., num_trajectories - 1]
        num_trajectories = len(path_idx)
        x = cp.Variable(num_trajectories, integer=True)  # agents per trajectory

        # Objective: minimize total cost
        cost = cp.sum(cp.multiply(path_cost, x))  # c: vector of trajectory costs

        # Build node time to trajectory
        node_time_to_traj = defaultdict(list)
        for i, path in enumerate(shortest_paths):
            for t, node in enumerate(path):
                node_time_to_traj[(node, t)].append(i)

        constraints = []

        # 1. Start location constraints
        for i, start in enumerate(self.starts_idx):
            start_indices = [j for j, path in enumerate(shortest_paths) if path[0] == start]
            constraints.append(cp.sum(x[start_indices]) == self.starts_agent_count[i])

        # 2. Goal location constraints
        for i, goal in enumerate(self.goals_idx):
            goal_indices = [j for j, path in enumerate(shortest_paths) if path[-1] == goal]
            constraints.append(cp.sum(x[goal_indices]) == self.goals_agent_count[i])

        # 3. Capacity constraints at intermediate nodes
        if self.capacity_constraint:
            for node, cap in enumerate(self.node_capacity):
                for t in range(len(shortest_paths[0])):
                    trajs = node_time_to_traj.get((node, t), [])
                    if trajs:
                        constraints.append(cp.sum(x[trajs]) <= cap)

        # 4. Optional: x >= 0
        constraints.append(x >= 0)

        # Solve
        prob = cp.Problem(cp.Minimize(cost), constraints)
        try:
            prob.solve(solver=cp.SCIP, verbose=False)
        except cp.SolverError as e:
            print("Solver failed:", e)

        # Access best available solution
        if prob.status in [
            cp.OPTIMAL, cp.OPTIMAL_INACCURATE,
            cp.USER_LIMIT, cp.SOLVER_ERROR
        ]:
            # Convert solution to paths
            paths = []
            for i, count in enumerate(x.value): # type: ignore
                for _ in range(int(count)):
                    paths.append(shortest_paths[i])

            # pad paths to the same length
            max_path_length = max([len(path) for path in paths])
            padded_paths = []
            for path in paths:
                padded_path = path + [path[-1]] * (max_path_length - len(path))
                padded_paths.append(padded_path)

            cost = self.get_cost(paths)

            return {
                "success": True,
                "timestep": max_path_length,
                "g_nodes": self.gaussian_prm.gaussian_nodes,
                "starts_idx": self.starts_idx,
                "goals_idx": self.goals_idx,
                "paths": padded_paths,  
                "cost": cost,
                "prob_value": prob.value
                }

        else:
            print("No feasible solution found.")
            return {"success": False}