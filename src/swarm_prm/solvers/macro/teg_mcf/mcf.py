"""
    Min Cost Flow Solver
"""

from collections import defaultdict
import heapq

class MinCostFlow:
    """
        Min Cost Flow Solver that can reuse residual graphs.
    """
    def __init__(self, start, goal, residual_graph, 
                 cost_graph, 
                 flow_constraints,
                 initial_flow=0., initial_cost=0.,
                 search_method="EK") -> None:
        """
            Min Cost Flow
        """
        self.search_method = search_method

        # reuse previous search result if possible
        self.residual_graph = residual_graph 
        self.cost_graph= cost_graph 

        # if provided residual graph, update initial flow
        self.initial_flow = initial_flow
        self.initial_cost = initial_cost
        self.start = start
        self.goal = goal
        self.heuristic = self.build_heuristic()

        # flow_constraint
        self.flow_constraints = flow_constraints
    
    def build_heuristic(self):
        """
            Build heuristic based on 
            TODO: implement this
        """
        return defaultdict(lambda:0)
        
    def dijkstra(self):
        """
            Dijkstra for finding min cost augmenting path
        """
        prev = {self.start: None}
        flow = {self.start: float('inf')}
        cost = 0
        open_list = [(cost, self.start)]
        heapq.heapify(open_list)
        while len(open_list) > 0:
            cost, curr_node = heapq.heappop(open_list)
            if curr_node == self.goal:
                path = []
                while curr_node:
                    path.append(curr_node)
                    curr_node = prev[curr_node]
                return path[::-1], flow[self.goal], cost
            for neighbor, capacity  in self.residual_graph[curr_node].items():
                if neighbor not in prev and capacity > 0:
                    prev[neighbor] = curr_node
                    flow[neighbor] = min(flow[curr_node], capacity)
                    heapq.heappush(open_list, (cost+self.cost_graph[curr_node][neighbor], neighbor))
        return None, 0, 0

    def update_flow(self, path, flow):
        """
            Update flow graph
        """
        for u, v in zip(path[:-1], path[1:]):
            assert flow > 0
            self.residual_graph[u][v] -= flow
            self.residual_graph[v][u] += flow
    
    def verify_node(self, next_node):
        """
            Verify if next node is valid
            TODO: Implement this
        """
        return True
    
    def solve(self):
        """
            Finding Max Flow
        """
        if self.search_method == "EK":
            """
                Edmond Karp 
            """
            total_flow = self.initial_flow
            total_cost = self.initial_cost

            while True:
                path, flow, cost = self.dijkstra()
                if not path:
                    break
                self.update_flow(path, flow)

                total_flow += flow
                total_cost += cost*flow

            return total_flow, self.residual_graph, total_cost

        elif self.search_method== "BS":
            """
                Bulk Search
            """
            assert False, "Not Implemented"
        else:
            assert False, "Not Implemented"
