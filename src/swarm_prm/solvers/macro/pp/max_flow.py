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
        self._augment_graph = self._build_augment_graph()
        self.start = start
        self.goal = goal
    
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
        prev = dict()
        prev[start] = None
        open_list = deque([start])
        visited = {}
        reached_goal = False
        while open_list:
            curr_node = open_list.popleft()
            if curr_node == goal:
                reached_goal = True
                break
            visited[curr_node] = 0
            neighbors = [node for node in self._augment_graph[curr_node] \
                if node not in visited and self._augment_graph[curr_node][node] > 0]
            for neighbor in neighbors:
                prev[neighbor] = curr_node
                open_list.append(neighbor)

        if reached_goal:
            path = []
            flow = float("inf")
            curr_node = goal
            while curr_node:
                path.append(curr_node)
                flow = min(flow, self.graph[curr_node][1])
                curr_node = curr_node["prev"]
            return path[::-1], flow
        else:
            return None, 0

    def _update_flow(self, path, flow):
        """
            Update flow graph
        """
        for u, v in zip(path[:-1], path[1:]):
            self._augment_graph[u][v] -= flow
            self._augment_graph[v][u] += flow
    
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
                path, flow = self._bfs(self.start, self.goal)
                if flow == 0:
                    break
                self._update_flow(path, flow)
                total_flow += flow

            return total_flow, self._augment_graph

        elif self.search_method== "BS":
            """
                Bulk Search
            """
            assert False, "Not Implemented"
            return 0, self._augment_graph
        
        else:
            assert False, "Not Implemented"
