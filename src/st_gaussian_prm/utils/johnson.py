"""
    Johnson's algorithm for finding all pair shortest distance
"""

import heapq

def bellman_ford(graph, source):
    """Runs Bellman-Ford algorithm to find shortest path from source."""
    dist = {node: float('inf') for node in graph}
    dist[source] = 0

    for _ in range(len(graph) - 1):
        for u in graph:
            for v, weight in graph[u]:
                if dist[u] + weight < dist[v]:
                    dist[v] = dist[u] + weight

    # Check for negative weight cycles
    for u in graph:
        for v, weight in graph[u]:
            if dist[u] + weight < dist[v]:
                return None  # Negative weight cycle detected
    return dist

def dijkstra(graph, source):
    """Runs Dijkstra’s algorithm from a given source node."""
    pq = [(0, source)]
    dist = {node: float('inf') for node in graph}
    dist[source] = 0

    while pq:
        d, u = heapq.heappop(pq)
        if d > dist[u]:
            continue
        for v, weight in graph[u]:
            if dist[u] + weight < dist[v]:
                dist[v] = dist[u] + weight
                heapq.heappush(pq, (int(dist[v]), v))
    return dist

def johnsons_algorithm(graph):
    """Computes shortest paths between all pairs using Johnson’s algorithm."""
    # Step 1: Add new vertex 'q' and connect to all other nodes with zero weight
    new_graph = {node: edges.copy() for node, edges in graph.items()}
    new_graph['q'] = [(node, 0) for node in graph]

    # Step 2: Run Bellman-Ford from 'q'
    h = bellman_ford(new_graph, 'q')
    if h is None:
        raise ValueError("Graph contains a negative weight cycle")

    # Step 3: Re-weight edges
    reweighted_graph = {}
    for u in graph:
        reweighted_graph[u] = []
        for v, weight in graph[u]:
            new_weight = weight + h[u] - h[v]
            reweighted_graph[u].append((v, new_weight))

    # Step 4: Run Dijkstra from each node
    shortest_paths = {}
    for u in graph:
        dijkstra_dist = dijkstra(reweighted_graph, u)

        # Step 5: Convert distances back to original weights
        for v in graph:
            if dijkstra_dist[v] < float('inf'):
                shortest_paths[(u, v)] = dijkstra_dist[v] + h[v] - h[u]
            else:
                shortest_paths[(u, v)] = float('inf')

    return shortest_paths

# Example usage
graph = {
    'A': [('B', 3), ('C', 8)],
    'B': [('C', 2), ('D', 5)],
    'C': [('D', 1)],
    'D': [],
}

if __name__ == "__main__":
    shortest_distances = johnsons_algorithm(graph)
    for (u, v), dist in shortest_distances.items():
        print(f"Shortest distance from {u} to {v}: {dist}")
