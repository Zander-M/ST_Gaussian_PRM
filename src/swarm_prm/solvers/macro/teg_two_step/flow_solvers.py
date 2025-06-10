"""
    Flow Solvers
"""

from collections import deque, defaultdict
import heapq

### Max Flow Solver 

class MaxFlow:
    """
        Max Flow Solver that can reuse residual graphs.
    """
    def __init__(self, start, goal, 
                 residual_graph, initial_flow=0.,
                 search_method="EK") -> None:
        """
            Max flow
        """
        self.search_method = search_method

        # reuse previous search result if possible
        self.residual_graph = residual_graph 

        # if provided residual graph, update initial flow
        self.initial_flow = initial_flow
        self.start = start
        self.goal = goal
    
    def build_forward_bound(self):
        """
            Compute the shortest distance to any possible goal. If remaining time
            is less than the shortest distance, stop searching
            TODO: implement this
        """
        forward_bound = dict()
        return forward_bound

    def build_backward_bound(self):
        """
            Compute the shortest distance to any possible goal. If remaining time
            is less than the shortest distance, stop searching
            TODO: implement this
        """
        backward_bound = dict()
        return backward_bound
    
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

    def update_flow(self, path, flow):
        """
            Update flow graph
        """
        for u, v in zip(path[:-1], path[1:]):
            assert flow > 0
            self.residual_graph[u][v] -= flow
            self.residual_graph[v][u] += flow
    
    def solve(self):
        """
            Finding Max Flow using Edmonds Karp
        """
        total_flow = self.initial_flow

        while True:
            path, flow = self.bidirectional_bfs()
            if not path:
                break
            self.update_flow(path, flow)

            total_flow += flow

        return total_flow, self.residual_graph 

## Min Cost Flow Solver

import networkx as nx

class MinCostFlow:
    def __init__(self, teg, cost_graph, start, goal, num_agents):
        self.graph = nx.DiGraph()
        self.start = start
        self.goal = goal

        nodes = list(teg.keys())
        edges = []

        for u in nodes:
            for v in teg[u]:
                edges.append((u, v, 
                                   {"capacity": min(teg[u][v], num_agents), 
                                    "weight": max(int(cost_graph[u][v]*1000), 0)}))
        self.graph.add_node(start, demand=-num_agents)                            
        self.graph.add_node(goal, demand=num_agents)                            
        self.graph.add_edges_from(edges)
        
    def solve(self):
        raw_flow_dict = nx.min_cost_flow(self.graph)

        # convert to nodes only
        flow_dict = defaultdict(lambda:dict())
        for u, nodes in raw_flow_dict.items():
            if u[-1] == 1:
                for v, flow in nodes.items():
                    flow_dict[(u[0], u[1])][(v[0], v[1])] = flow

        return flow_dict