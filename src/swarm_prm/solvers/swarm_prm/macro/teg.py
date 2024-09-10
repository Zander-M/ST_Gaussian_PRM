"""
    Find the shortest time to travel through the graph by incrementally updating
    Time Expanded Graph (TEG) and checking for the max flow of the graph
"""

from collections import deque, defaultdict
import numpy as np

from swarm_prm.solvers.swarm_prm.macro.gaussian_prm import GaussianPRM

INF = 1e9 # used as infinite flow edge capacity 

class TEGGraph():
    """
        Time expanded graph
    """
    def __init__(self, gaussian_prm, agent_radius) -> None:
        self.gaussian_prm = gaussian_prm
        self.agent_radius = agent_radius
        self.roadmap_graph = self.build_roadmap_graph()
        self.residual_graph = self.build_residual_graph()



    def build_roadmap_graph(self, method="MIN_CAPACITY"):
        """
            Find the earliest timestep that reaches the max flow
        """
        graph = defaultdict(list)

        if method == "MIN_CAPACITY":
            for edge in self.gaussian_prm.roadmap:
                u, v = edge
                capacity = min(self.gaussian_prm.gaussian_nodes[u].get_capacity(self.agent_radius),
                               self.gaussian_prm.gaussian_nodes[v].get_capacity(self.agent_radius))
                graph[u].append((v, capacity))
                graph[v].append((u, capacity))

        elif method == "VERTEX_CAPACITY":
            assert False, "Unimplemented roadmap graph construction method."
        return graph

    def build_residual_graph(self):
        """
            Build Residual Graph for finding max flow
        """
        residual_graph = defaultdict(list)
        pass

    
    def build_teg(self, timestep):
        """
            Build TEG based on current timestep.
        """
        teg = {}
        return teg

    def bfs(self, start, goal, parent):
        """
            Breath First Search for finding augmenting path
        """
        visited = set()
        open = deque([start])
        visited.add(start)

        while open:
            u = open.popleft()
            for v, capacity in self.roadmap_graph[u]:
                if v not in visited and self.residual_capacities[(u, v)] > 0:
                    open.append(v)
                    visited.add(v)
                    parent[v] = u
                    if v == goal:
                        return True
        return False

        
    def edmond_karp(self):
        """
            Edmond Karp for max flow update
        """

    def find_max_flow(self):
        """
            Find max flow of the current graph using Ford-Fulkerson Algorithm
        """

    def build_teg(self):
        """
            Build teg graph from roadmap
        """




