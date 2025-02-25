"""
    Greedy based solver
"""

from collections import defaultdict, deque
import numpy as np
from scipy.optimize import linear_sum_assignment

from swarm_prm.solvers.utils.gaussian_prm import GaussianPRM

class GreedySolver:
    def __init__(self, gaussian_prm:GaussianPRM, agent_radius, num_agents,
                 goal_state_prob=0.1, max_time=6000):
        self.gaussian_prm = gaussian_prm
        self.agent_radius = agent_radius
        self.num_agents = num_agents
        self.roadmap_graph = self.build_roadmap_graph()
        self.nodes = [i for i in range(len(self.gaussian_prm.samples))]
        self.node_capacity = [node.get_capacity(self.agent_radius) for node in self.gaussian_prm.gaussian_nodes]

    def build_roadmap_graph(self):
        """
            Find the earliest timestep that reaches the max flow
        """
        graph = defaultdict(list)

        for edge in self.gaussian_prm.roadmap:
            u, v = edge
            graph[u].append(v)
            graph[v].append(u)
        return graph

    def get_solution(self):
        pass

    def bidirectional_bfs(self, start, goal):
        """
            Bidirectional BFS
        """
        forward_queue = deque([self.start])
        backward_queue= deque([self.goal])
        forward_parent= {self.start: None}
        backward_parent= {self.goal: None}

        while forward_queue and backward_queue:
            if forward_queue:
                current = forward_queue.popleft()
                for neighbor, capacity in self.residual_graph[current].items():
                    if neighbor not in forward_parent and capacity > 0:
                        forward_parent[neighbor] = current
                        if neighbor in backward_parent:
                            return self._construct_path(forward_parent, backward_parent, neighbor)
                        forward_queue.append(neighbor)

            if backward_queue:
                current = backward_queue.popleft()
                for neighbor, _ in self.residual_graph[current].items():
                    if neighbor not in backward_parent and self.residual_graph[neighbor][current] > 0:
                        backward_parent[neighbor] = current 
                        if neighbor in forward_parent:
                            return self._construct_path(forward_parent, backward_parent, neighbor)
                        backward_queue.append(neighbor)
        return None, 0

    def _construct_path(self, forward_parent, backward_parent, meeting_point):
        """
        Construct the full path from source to sink using the meeting point.
        """
        path = []
        flow = float("inf")
        # Build forward path
        current = meeting_point
        while current is not None:
            path.append(current)
            if forward_parent[current] is not None:
                flow = min(flow, self.residual_graph[forward_parent[current]][current])
            current = forward_parent[current]

        path = path[::-1]  # Reverse to get source to meeting point

        # Build backward path
        current = meeting_point
        while current is not None:
            if backward_parent[current] is not None:
                flow = min(flow, self.residual_graph[current][backward_parent[current]])
            current = backward_parent[current]
            if current is not None:
                path.append(current)
        return path, flow

