"""
    Max Flow Solver
"""

from collections import defaultdict, deque

class MaxFlowSolver:
    """
        Max Flow Solver that can reuse partial solutions.
    """
    def __init__(self, graph, start, goal, flow=None, 
                 search_method="EK") -> None:
        """
            Max flow
        """
        self.graph = graph
        self.flow = flow
        self.search_method = search_method
        self.residual_graph = self.build_residual_graph()
        self.start = start
        self.goal = goal
    
    def build_residual_graph(self):
        """
            Build Augment Graph
        """
        augment_graph = defaultdict(lambda:dict())
        for u in self.graph:
            for v in self.graph[u]:
                augment_graph[u][v] = self.graph[u][v]
                augment_graph[v][u] = 0
        return augment_graph
        
    def bfs(self, start, goal):
        """
            BFS for finding augmenting path
        """
        prev = {start: None}
        flow = {start: float('inf')}
        open_list = deque([start])

        while open_list:
            curr_node = open_list.popleft()
            if curr_node == goal:
                return prev, flow[goal]
            for neighbor, capacity in self.residual_graph[curr_node].items():
                if neighbor not in prev and capacity > 0:
                    prev[neighbor] = curr_node
                    flow[neighbor] = min(flow[curr_node], capacity)
                    open_list.append(neighbor)
        return None, 0

    def update_flow(self, path, flow):
        """
            Update flow graph
        """
        for u, v in zip(path[:-1], path[1:]):
            self.residual_graph[u][v] -= flow
            self.residual_graph[v][u] += flow
    
    def solve(self):
        """
            Finding Max Flow
        """
        if self.search_method == "EK":
            """
                Edmond Karp 
            """
            total_flow = 0

            while True:
                prev, flow = self.bfs(self.start, self.goal)
                if not prev:
                    break
                node = self.goal
                while node != self.start:
                    prev_node = prev[node]
                    self.residual_graph[prev_node][node] -= flow
                    self.residual_graph[node][prev_node] += flow
                    node = prev[node]
                total_flow += flow

            return total_flow, self.residual_graph

        elif self.search_method== "BS":
            """
                Bulk Search
            """
            assert False, "Not Implemented"
            return 0, self.residual_graph
        
        else:
            assert False, "Not Implemented"
