"""
    Max Flow Solver
"""

from collections import defaultdict

class MaxFlowSolver:
    """
        Max Flow Solver that can reuse partial solutions.
    """
    def __init__(self, graph, flow=None, search_method="bfs") -> None:
        """
            Max flow on Abstract Graph 
        """
        self.graph = graph
        self.flow = flow
        self._flow_graph = self._build_augment_graph()
        self._augment_graph = self._build_augment_graph()

    
    def _build_augment_graph(self):
        """
            Build Augment Graph
        """
        augment_graph = defaultdict(lambda:dict())
        for u in self.graph:
            for v, capacity in self.graph[u]:
                augment_graph[u][v] = capacity
                augment_graph[v][u] = 0
        return augment_graph
        
    def _bfs(self, start, goal):
        """
            BFS for finding augmenting path
        """
        open_list = []
        curr_node = {
            "prev": None, 
            "node": start
        }

        open_list.append(curr_node)
        while open_list:
            curr_node = open_list.pop(0)
            if curr_node == goal:
                break
            neighbors = [node for node in self.graph[curr_node]]
            for neighbor in neighbors:
                if self._augment_graph[curr_node][neighbor] == 0:
                    continue
                node = {
                    "prev": curr_node,
                    "node": neighbor,
                }
                open_list.append(node)

        path = []
        flow = float("inf")
        while curr_node:
            path.append(curr_node["node"][0])
            flow = min(flow, curr_node["node"][1])
            curr_node = curr_node["prev"]
        return path[::-1], flow

    def _update_flow(self):
        """
            Update flow graph
        """
    
    def solve(self, method="FF"):
        """
            Finding Max Flow
        """
        if method == "FF":
            """
                Ford Fulkerson
            """
            pass
            
        
        elif method == "BS":
            """
                Bulk Search
            """
            pass
