"""
    Base Class for macro solver
"""
from abc import ABC, abstractmethod
from collections import defaultdict, Counter
from typing import Dict, Any
import heapq

import numpy as np

class MacroSolverBase(ABC):
    def __init__(self, gaussian_prm, agent_radius, 
                 num_agents, starts_agent_count, goals_agent_count, 
                 starts_idx, goals_idx, 
                 **experiment_config) -> None:
        self.gaussian_prm = gaussian_prm
        self.obstacle_map = self.gaussian_prm.obstacle_map # Map geometry
        self.agent_radius = agent_radius
        self.num_agents = num_agents

        self.starts_agent_count = starts_agent_count
        self.goals_agent_count = goals_agent_count

        # starts and goals are now required for each planning instance
        self.starts_idx = starts_idx
        self.goals_idx = goals_idx
        self.time_limit = experiment_config.get("time_limit", 180) # by default 3 mins

        self.roadmap, self.cost_dict = self.build_roadmap_with_cost()
        self.nodes = np.array(self.gaussian_prm.samples)
        self.node_capacity = [node.get_capacity(agent_radius) for node in self.gaussian_prm.gaussian_nodes]

        # Verify if instance is feasible
        for i, start in enumerate(self.starts_idx):
            assert self.node_capacity[start] >= self.starts_agent_count[i],\
                "Start capacity smaller than required."

        for i, goal in enumerate(self.goals_idx):
            assert self.node_capacity[goal] >= self.goals_agent_count[i], \
                "Goal capacity smaller than required."
        self.init_solver(**experiment_config)
    
    def build_roadmap_with_cost(self):
        """
            Build roadmap that includes self loops for wait actions.
            Return roadmap and the cost map
        """
        graph = defaultdict(list)
        cost = defaultdict(defaultdict)

        for i, edge in enumerate(self.gaussian_prm.roadmap):
            u, v = edge
            graph[u].append(v)
            graph[v].append(u)
            cost[u][v] = self.gaussian_prm.roadmap_cost[i]
            cost[v][u] = self.gaussian_prm.roadmap_cost[i]

        # adding wait edges
        for i in range(len(self.gaussian_prm.samples)):
            graph[i].append(i) # waiting at node has 0 transport cost
            cost[i][i] = 0
        return graph, cost
    
    def eval_capacity(self, paths):
        """
            Evaluate if solution violates node capacity.
        """
        num_violation = 0
        max_violation_percentage = 0

        for t in range(len(paths[0])):
            current_pos = [path[t] for path in paths]
            state_count = Counter(current_pos)
            for state, count in state_count.items():
                if count > self.node_capacity[state]:
                    num_violation += 1
                    violation_percentage = (count-self.node_capacity[state])/self.node_capacity[state]
                    max_violation_percentage = max (violation_percentage, max_violation_percentage)
        return num_violation, max_violation_percentage
    
    def get_shortest_path_cost(self, node1, node2):
        """
            Get shortest path cost between node 1 and node 2
        """
        prev = {node1: None}
        open_list = [(0, node1)]
        distances = {node1: 0}
        while open_list:
            cost, state = heapq.heappop(open_list)
            if state == node2:
                path = []
                while state is not None:
                    path.append(state)
                    state = prev[state]
                return path[::-1], distances[node2]
            for neighbor in self.roadmap[state]:
                curr_cost = cost + self.cost_dict[state][neighbor]
                if neighbor not in distances or curr_cost < distances[neighbor]:
                    distances[neighbor] = curr_cost
                    prev[neighbor] = state
                    heapq.heappush(open_list, (curr_cost, neighbor))
        return [], 0

    @abstractmethod
    def init_solver(self, **kwargs):
        """
            Initialize solver using solver specific kwargs
        """
        pass
        
    @abstractmethod
    def solve(self)->Dict[str, Any]:
        """
            Solving planning problem
        """
        return {
            "success": False,
            "timestep": 0, 
            "cost": 0
            }