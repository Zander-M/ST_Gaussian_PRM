from collections import deque, defaultdict

class MaxFlowWithVertexCapacity:
    def __init__(self, graph, vertex_capacity):
        self.original_graph = graph  # adjacency list of the original graph
        self.vertex_capacity = vertex_capacity  # vertex capacities
        self.modified_graph = self.build_modified_graph()

    def build_modified_graph(self):
        """Splits each vertex with a capacity into two vertices connected by an edge with the vertex capacity."""
        modified_graph = defaultdict(list)
        self.new_vertex_index = max(self.original_graph) + 1

        for u in self.original_graph:
            for v, capacity in self.original_graph[u]:
                if u in self.vertex_capacity:
                    # Split the vertex u into u_in and u_out
                    u_in = u
                    u_out = self.new_vertex_index
                    self.new_vertex_index += 1
                    # Add the internal edge with the vertex capacity
                    modified_graph[u_in].append((u_out, self.vertex_capacity[u]))
                    modified_graph[u_out].append((u_in, 0))  # Reverse edge for residual graph
                    # Add the original edge to the modified graph
                    modified_graph[u_out].append((v, capacity))
                    modified_graph[v].append((u_out, 0))  # Reverse edge for residual graph
                else:
                    # If there's no vertex capacity constraint, add the edge normally
                    modified_graph[u].append((v, capacity))
                    modified_graph[v].append((u, 0))  # Reverse edge for residual graph

        return modified_graph

    def bfs(self, s, t, parent):
        visited = [False] * self.new_vertex_index
        queue = deque([s])
        visited[s] = True

        while queue:
            u = queue.popleft()

            for v, capacity in self.modified_graph[u]:
                if visited[v] is False and capacity > 0:
                    queue.append(v)
                    visited[v] = True
                    parent[v] = u

                    if v == t:
                        return True
        return False

    def edmonds_karp(self, source, sink):
        parent = [-1] * self.new_vertex_index
        max_flow = 0

        while self.bfs(source, sink, parent):
            path_flow = float('Inf')
            s = sink

            while s != source:
                path_flow = min(path_flow, self.modified_graph[parent[s]][[v for v, cap in self.modified_graph[parent[s]]].index(s)][1])
                s = parent[s]

            max_flow += path_flow

            v = sink
            while v != source:
                u = parent[v]
                self.modified_graph[u][[n for n, cap in self.modified_graph[u]].index(v)] = (v, self.modified_graph[u][[n for n, cap in self.modified_graph[u]].index(v)][1] - path_flow)
                self.modified_graph[v][[n for n, cap in self.modified_graph[v]].index(u)] = (u, self.modified_graph[v][[n for n, cap in self.modified_graph[v]].index(u)][1] + path_flow)
                v = parent[v]

        return max_flow

# Example usage:
graph = {
    0: [(1, 16), (2, 13)],
    1: [(3, 12)],
    2: [(1, 4), (4, 14)],
    3: [(2, 9), (5, 20)],
    4: [(3, 7), (5, 4)],
    5: []
}
vertex_capacity = {
    1: 20,  # Capacity of vertex 1
    3: 30   # Capacity of vertex 3
}

max_flow_solver = MaxFlowWithVertexCapacity(graph, vertex_capacity)
source = 0  # source node
sink = 5    # sink node

print(f"The maximum possible flow is {max_flow_solver.edmonds_karp(source, sink)}")
