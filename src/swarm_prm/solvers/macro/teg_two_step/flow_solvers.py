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

        return flow_dict, None

# class MinCostFlow:
#     """
#     Min Cost Flow Solver for fixed-flow improvement.
#     Reduce cost using negative-cost cycle cancelling
#     """
#     def __init__(self, residual_graph, cost_graph):
#         self.residual_graph = residual_graph  # dict[u][v] = capacity
#         self.cost_graph = cost_graph          # dict[u][v] = cost
# 
#     def find_negative_cycle_bellman_ford(self):
#         """
#             Bellman-Ford to detect and extract negative-cost cycles.
#             Returns list of nodes in cycle if found
#         """
#         dist = defaultdict(lambda: 0)
#         pred = {}
#         nodes = list(self.residual_graph.keys())
# 
#         for _ in range(len(nodes)):
#             updated = False
#             for u in nodes:
#                 for v in self.residual_graph[u]:
#                     if self.residual_graph[u][v] > 0:
#                         if dist[v] > dist[u] + self.cost_graph[u][v]:
#                             dist[v] = dist[u] + self.cost_graph[u][v]
#                             pred[v] = u
#                             updated = True
#             if not updated:
#                 break
#         
#         # Find negative cycle
#         for u in nodes:
#             for v in self.residual_graph[u]:
#                 if self.residual_graph[u][v] > 0 and dist[v] > dist[u] + self.cost_graph[u][v]:
#                     cycle = []
#                     visited = set()
#                     curr = v
#                     while curr not in visited:
#                         visited.add(curr)
#                         curr = pred[curr]
#                     start = curr
#                     cycle.append(start)
#                     curr = pred[start]
#                     while curr != start:
#                         cycle.append(curr)
#                         curr = pred[curr]
#                     cycle.reverse()
#                     return cycle
#         return None
#     
#     def cancel_cycle(self, cycle):
#         """
#             Push flow along negative cycle to reduce flow cost
#         """
#         bottleneck = float('inf')
#         for u, v in zip(cycle, cycle[1:]+[cycle[0]]):
#             bottleneck = min(bottleneck, self.residual_graph[u][v])
#         
#         for u, v in zip(cycle, cycle[1:]+[cycle[0]]):
#             self.residual_graph[u][v] -= bottleneck
#             self.residual_graph[v][u] += bottleneck
#         
#         cost_reduction = 0
#         for u, v in zip(cycle, cycle[1:]+[cycle[0]]):
#             cost_reduction += self.cost_graph[u][v] * bottleneck
#         return bottleneck, cost_reduction
# 
#     def solve(self):
#         """
#             Improve cost of a feasible flow solution
#         """
#         total_cost_reduction = 0
#         while True:
#             cycle = self.find_negative_cycle_bellman_ford()
#             if not cycle:
#                 break
#             _, delta_cost = self.cancel_cycle(cycle)
#             total_cost_reduction += delta_cost
#         return self.residual_graph, -total_cost_reduction

# class MinCostFlow:
    # """
    # Min Cost Flow Optimizer for fixed-flow improvement.
    # Assumes an initial feasible flow has been found, and attempts to reduce cost
    # by cancelling negative-cost cycles in the residual graph.
    # """
    # def __init__(self, residual_graph, cost_graph, max_iterations=1000):
        # self.residual_graph = residual_graph  # dict[u][v] = capacity (can be negative for back edges)
        # self.cost_graph = cost_graph          # dict[u][v] = cost (can be negative)
        # self.max_iterations = max_iterations
# 
    # def find_negative_cycle_spfa(self):
        # """
        # SPFA (Shortest Path Faster Algorithm) to detect negative cost cycles
        # Returns list of nodes in cycle if found, else None
        # """
        # dist = defaultdict(lambda: 0)
        # pred = {}
        # in_queue = defaultdict(bool)
        # count = defaultdict(int)
        # queue = deque(self.residual_graph.keys())
# 
        # for node in queue:
            # in_queue[node] = True
# 
        # while queue:
            # u = queue.popleft()
            # in_queue[u] = False
            # for v, cap in self.residual_graph[u].items():
                # if cap <= 0:
                    # continue
                # cost = self.cost_graph[u][v]
                # if dist[v] > dist[u] + cost:
                    # dist[v] = dist[u] + cost
                    # pred[v] = u
                    # count[v] += 1
                    # if count[v] >= len(self.residual_graph):
                        # Negative cycle detected, recover it
                        # cycle = []
                        # visited = set()
                        # curr = v
                        # while curr not in visited:
                            # visited.add(curr)
                            # curr = pred[curr]
                        # start = curr
                        # cycle.append(start)
                        # curr = pred[start]
                        # while curr != start:
                            # cycle.append(curr)
                            # curr = pred[curr]
                        # cycle.reverse()
                        # return cycle
                    # if not in_queue[v]:
                        # queue.append(v)
                        # in_queue[v] = True
        # return None
# 
    # def cancel_cycle(self, cycle):
        # """
        # Push flow along a negative cost cycle to reduce total cost
        # """
        # bottleneck = float('inf')
        # for u, v in zip(cycle, cycle[1:] + [cycle[0]]):
            # bottleneck = min(bottleneck, self.residual_graph[u][v])
# 
        # for u, v in zip(cycle, cycle[1:] + [cycle[0]]):
            # self.residual_graph[u][v] -= bottleneck
            # self.residual_graph[v][u] += bottleneck
# 
        # cost_reduction = sum(
            # self.cost_graph[u][v] * bottleneck for u, v in zip(cycle, cycle[1:] + [cycle[0]])
        # )
# 
        # return bottleneck, cost_reduction
# 
    # def solve(self):
        # """
        # Improve cost of a feasible flow by cancelling negative-cost cycles
        # """
        # total_cost_reduction = 0
        # iteration = 0
        # while iteration < self.max_iterations:
            # cycle = self.find_negative_cycle_spfa()
            # if not cycle:
                # break
            # _, delta_cost = self.cancel_cycle(cycle)
            # total_cost_reduction += delta_cost
            # iteration += 1
        # if iteration == self.max_iterations:
            # print("Max iterations reached.")
        # return self.residual_graph, -total_cost_reduction