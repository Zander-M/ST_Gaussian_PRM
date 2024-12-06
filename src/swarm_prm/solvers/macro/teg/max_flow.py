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
        
    def bfs(self):
        """
            BFS for finding augmenting path
        """
        prev = {self.start: None}
        flow = {self.start: float('inf')}
        open_list = deque([self.start])

        while open_list:
            curr_node = open_list.popleft()

            for neighbor, capacity in self.residual_graph[curr_node].items():
                if neighbor not in prev and capacity > 0:
                    prev[neighbor] = curr_node
                    flow[neighbor] = min(flow[curr_node], capacity)

                    if neighbor == self.goal:
                        path = []
                        curr_node = neighbor
                        while curr_node:
                            path.append(curr_node)
                            curr_node = prev[curr_node]
                        return path[::-1], flow[self.goal]
                    open_list.append(neighbor)

        return None, 0

    def bidirectional_bfs(self):
        """
            Bidirectional BFS
        """
        forward_queue = deque([self.start])
        backward_queue= deque([self.goal])
        forward_parent= {self.start: None}
        backward_parent= {self.goal: None}
        forward_visited = set([self.start])
        backward_visited= set([self.goal ])

        while forward_queue and backward_queue:
            if forward_queue:
                current = forward_queue.popleft()
                for neighbor, capacity in self.residual_graph[current].items():
                    if neighbor not in forward_parent and capacity > 0:
                        forward_parent[neighbor] = current
                        forward_visited.add(neighbor)
                        forward_queue.append(neighbor)
                        if neighbor in backward_visited:
                            return self._construct_path(forward_parent, backward_parent, neighbor)

            if backward_queue:
                current = backward_queue.popleft()
                for neighbor, capacity in self.residual_graph[current].items():
                    if current not in backward_parent and capacity > 0:
                        backward_parent[current] = neighbor
                        backward_visited.add(neighbor)
                        backward_queue.append(neighbor)
                        if neighbor in forward_visited:
                            return self._construct_path(forward_parent, backward_parent, neighbor)
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
                path, flow = self.bfs()
                if not path:
                    break
                self.update_flow(path, flow)

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
