"""
    code tests
"""
from scipy.stats import gaussian_kde
from matplotlib import pyplot as plt
import numpy as np

def test_gaussian():
    """
        Test Gaussian KDE
    """
    # Generate some random data
    data = np.random.normal(0, 1, size=1000)

    # Fit a Gaussian KDE to the data
    kde = gaussian_kde(data)

    # Create a grid of points where we want to evaluate the KDE
    x_grid = np.linspace(-5, 5, 1000)

    # Evaluate the KDE on the grid
    kde_values = kde(x_grid)

    # Plot the results
    plt.figure(figsize=(8, 6))
    plt.plot(x_grid, kde_values, label='Gaussian KDE')
    plt.hist(data, bins=30, density=True, alpha=0.5, label='Histogram of data')
    plt.title('Gaussian Kernel Density Estimate')
    plt.xlabel('Value')
    plt.ylabel('Density')
    plt.legend()
    plt.savefig("test.png")

def test_prm():
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.spatial import KDTree

    class Node:
        def __init__(self, x, y):
            self.x = x
            self.y = y

    def distance(node1, node2):
        return np.sqrt((node1.x - node2.x)**2 + (node1.y - node2.y)**2)

    def is_collision_free(node, obstacles, radius=1):
        for ox, oy in obstacles:
            if distance(node, Node(ox, oy)) <= radius:
                return False
        return True

    def sample_free_space(bounds, obstacles, num_samples):
        samples = []
        min_x, max_x, min_y, max_y = bounds
        while len(samples) < num_samples:
            x = np.random.uniform(min_x, max_x)
            y = np.random.uniform(min_y, max_y)
            node = Node(x, y)
            if is_collision_free(node, obstacles):
                samples.append(node)
        return samples

    def build_roadmap(samples, k=10):
        roadmap = []
        kd_tree = KDTree([(node.x, node.y) for node in samples])
        for i, node in enumerate(samples):
            distances, indices = kd_tree.query((node.x, node.y), k=k+1)
            edges = [(i, idx) for idx, dist in zip(indices[1:], distances[1:]) if dist > 0]
            roadmap.extend(edges)
        return roadmap

    def find_path(start, goal, roadmap, samples):
        start_idx = len(samples)
        goal_idx = len(samples) + 1
        samples.extend([start, goal])

        kd_tree = KDTree([(node.x, node.y) for node in samples])
        roadmap.extend(build_roadmap([start, goal], k=10))

        graph = {i: [] for i in range(len(samples))}
        for (i, j) in roadmap:
            graph[i].append(j)
            graph[j].append(i)

        def dijkstra(graph, start, goal):
            import heapq
            queue = [(0, start)]
            distances = {node: float('inf') for node in graph}
            distances[start] = 0
            predecessors = {node: None for node in graph}

            while queue:
                current_distance, current_node = heapq.heappop(queue)
                if current_node == goal:
                    break
                for neighbor in graph[current_node]:
                    distance = current_distance + distance(samples[current_node], samples[neighbor])
                    if distance < distances[neighbor]:
                        distances[neighbor] = distance
                        predecessors[neighbor] = current_node
                        heapq.heappush(queue, (distance, neighbor))

            path = []
            while goal is not None:
                path.append(goal)
                goal = predecessors[goal]
            return path[::-1]

        path_indices = dijkstra(graph, start_idx, goal_idx)
        path = [samples[idx] for idx in path_indices]
        return path

    def plot_roadmap(roadmap, samples, obstacles, path=None):
        fig, ax = plt.subplots()
        for (i, j) in roadmap:
            ax.plot([samples[i].x, samples[j].x], [samples[i].y, samples[j].y], 'gray', linestyle='-', linewidth=0.5)

        for node in samples:
            ax.plot(node.x, node.y, 'bo', markersize=2)

        for ox, oy in obstacles:
            ax.plot(ox, oy, 'ro', markersize=3)
            ax.add_patch(plt.Circle((ox, oy), radius=1))


        if path:
            path_x = [node.x for node in path]
            path_y = [node.y for node in path]
            ax.plot(path_x, path_y, 'g-', linewidth=2)

        ax.set_aspect('equal')
        plt.show()

    # Define problem
    bounds = (0, 10, 0, 10)
    obstacles = [(2, 2), (3, 5), (7, 8), (6, 4)]
    num_samples = 100

    # PRM algorithm
    samples = sample_free_space(bounds, obstacles, num_samples)
    roadmap = build_roadmap(samples)
    start = Node(0, 0)
    goal = Node(9, 9)
    path = find_path(start, goal, roadmap, samples)

    # Plot results
    plot_roadmap(roadmap, samples, obstacles, path)

if __name__ == "__main__":
    # test_gaussian()
    test_prm()