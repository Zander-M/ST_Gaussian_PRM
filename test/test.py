"""
    code tests
"""

import ortools
from scipy.stats import gaussian_kde, multivariate_normal
from scipy.spatial import KDTree
from matplotlib import pyplot as plt
import numpy as np

from swarm_prm.envs.loader import MapLoader
from swarm_prm.solvers.swarm_prm.macro.gaussian_prm import GaussianPRM


def test_gaussian():
    """
        Test Gaussian KDE with visualization
    """
    # Create a grid of points where we want to evaluate the KDE
    x = np.linspace(0, 5, 100, endpoint=False)
    y = np.linspace(0, 5, 100, endpoint=False)
    xx, yy = np.meshgrid(x, y)

    z = multivariate_normal.pdf(np.dstack([xx, yy]), mean=[2.5, 2.5], cov=[2, 0.5])
    print(z.shape)
    
    # Plot the results
    ax = plt.figure(figsize=[8, 6]).add_subplot(projection='3d')
    ax.plot_surface(xx, yy, z, edgecolor='royalblue', lw=0.5, rstride=8, cstride=8,
                alpha=0.3)
    plt.savefig("test.png")


def test_prm():

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
            ax.add_patch(plt.Circle((ox, oy), radius=1, color="blue"))


        if path:
            path_x = [node.x for node in path]
            path_y = [node.y for node in path]
            ax.plot(path_x, path_y, 'g-', linewidth=2)

        ax.set_aspect('equal')
        plt.savefig("test.png")

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

def test_gaussian_prm():
    fname = "../data/envs/map_6.yaml"
    loader = MapLoader(fname)
    loader.visualize("test_map")
    map_instance = loader.get_map()
    gaussian_prm = GaussianPRM(map_instance, None, 300, sampling_strategy="UNIFORM_WITH_RADIUS")
    gaussian_prm.roadmap_construction()
    gaussian_prm.visualize_map("test_gprm")


def test_g_prm():
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.spatial import KDTree
    from scipy.stats import multivariate_normal

    class Node:
        def __init__(self, mean, cov):
            self.mean = np.array(mean)
            self.cov = np.array(cov)
        
        def distance_to_node(self, other):
            return np.linalg.norm(self.mean - other.mean)

    def is_collision_free(node, obstacles, radius=0.5):
        for ox, oy in obstacles:
            if multivariate_normal.pdf([ox, oy], mean=node.mean, cov=node.cov) > 1e-3:
                return False
        return True

    def distance_to_segment(px, py, ax, ay, bx, by):
        """Calculate the distance from a point (px, py) to a line segment (ax, ay) - (bx, by)"""
        if (ax == bx) and (ay == by):
            return np.sqrt((px - ax)**2 + (py - ay)**2)
    
        t = ((px - ax) * (bx - ax) + (py - ay) * (by - ay)) / ((bx - ax)**2 + (by - ay)**2)
        t = max(0, min(1, t))
        closest_x = ax + t * (bx - ax)
        closest_y = ay + t * (by - ay)
        return np.sqrt((px - closest_x)**2 + (py - closest_y)**2)

    def is_edge_collision_free(node1, node2, obstacles, radius=0.5):
        for ox, oy in obstacles:
            if distance_to_segment(ox, oy, node1.mean[0], node1.mean[1], node2.mean[0], node2.mean[1]) <= radius:
                return False
        return True

    def sample_free_space(bounds, obstacles, num_samples, cov_scale=0.1):
        samples = []
        min_x, max_x, min_y, max_y = bounds
        while len(samples) < num_samples:
            x = np.random.uniform(min_x, max_x)
            y = np.random.uniform(min_y, max_y)
            mean = [x, y]
            cov = cov_scale * np.identity(2)
            node = Node(mean, cov)
            if is_collision_free(node, obstacles):
                samples.append(node)
        return samples

    def build_roadmap(samples, obstacles, k=10, radius=0.5):
        roadmap = []
        kd_tree = KDTree([node.mean for node in samples])
        for i, node in enumerate(samples):
            distances, indices = kd_tree.query(node.mean, k=k+1)
            for idx, dist in zip(indices[1:], distances[1:]):
                if dist > 0 and is_edge_collision_free(node, samples[idx], obstacles, radius):
                    roadmap.append((i, idx))
        return roadmap

    def find_path(start, goal, roadmap, samples, obstacles, k=10):
        start_idx = len(samples)
        goal_idx = len(samples) + 1
        samples.extend([start, goal])
    
        # Build new roadmap connections for start and goal
        kd_tree = KDTree([node.mean for node in samples])
        new_roadmap = []
        for idx, node in enumerate([start, goal]):
            distances, indices = kd_tree.query(node.mean, k=k+1)
            for i, dist in zip(indices[1:], distances[1:]):
                if dist > 0 and is_edge_collision_free(node, samples[i], obstacles):
                    new_roadmap.append((start_idx + idx, i))
                    new_roadmap.append((i, start_idx + idx))
    
        roadmap.extend(new_roadmap)
    
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
                    distance = current_distance + samples[current_node].distance_to_node(samples[neighbor])
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
            ax.plot([samples[i].mean[0], samples[j].mean[0]], [samples[i].mean[1], samples[j].mean[1]], 'gray', linestyle='-', linewidth=0.5)
    
        for node in samples:
            ax.plot(node.mean[0], node.mean[1], 'bo', markersize=2)
    
        for ox, oy in obstacles:
            ax.plot(ox, oy, 'ro', markersize=3)
            ax.add_patch(plt.Circle((ox, oy), radius=1, color="blue"))
    
        if path:
            path_x = [node.mean[0] for node in path]
            path_y = [node.mean[1] for node in path]
            ax.plot(path_x, path_y, 'g-', linewidth=2)
    
        plt.savefig("test.png")

    # Define problem
    bounds = (0, 10, 0, 10)
    obstacles = [(2, 2), (3, 5), (7, 8), (6, 4)]
    num_samples = 100

    # PRM algorithm
    samples = sample_free_space(bounds, obstacles, num_samples)
    roadmap = build_roadmap(samples, obstacles)
    start = Node([0, 0], 0.1 * np.identity(2))
    goal = Node([9, 9], 0.1 * np.identity(2))
    path = find_path(start, goal, roadmap, samples, obstacles)

    # Plot results
    plot_roadmap(roadmap, samples, obstacles, path)

def test_fsolve():
    import numpy as np
    from scipy.optimize import fsolve

    # Given values
    t = 2  # distance
    p = 0.1  # probability density function value at distance t

    # Function to solve for sigma^2
    def equation(sigma2):
        return p * 2 * np.pi * sigma2 - np.exp(-t**2 / (2 * sigma2))

    # Initial guess for sigma^2
    initial_guess = t**2 / 2

    # Solve for sigma^2
    sigma2_solution, = fsolve(equation, initial_guess)

    # Covariance matrix
    covariance_matrix = np.diag([sigma2_solution, sigma2_solution])

    print("Sigma^2:", sigma2_solution)
    print("Covariance Matrix:\n", covariance_matrix)

from ortools.linear_solver import pywraplp


def test_ortools():
    """Linear programming sample."""
    # Instantiate a Glop solver, naming it LinearExample.
    solver = pywraplp.Solver.CreateSolver("GLOP")
    if not solver:
        return

    # Create the two variables and let them take on any non-negative value.
    x = solver.NumVar(0, solver.infinity(), "x")
    y = solver.NumVar(0, solver.infinity(), "y")

    print("Number of variables =", solver.NumVariables())

    # Constraint 0: x + 2y <= 14.
    # solver.Add(x + 2 * y <= 14.0)

    # Constraint 1: 3x - y >= 0.
    # solver.Add(3 * x - y >= 0.0)

    # Constraint 2: x - y <= 2.
    # solver.Add(x - y <= 2.0)
    solver.Add( x + y == 1)

    print("Number of constraints =", solver.NumConstraints())

    # Objective function: 3x + 4y.
    solver.Maximize(3 * x + 4 * y)

    # Solve the system.
    print(f"Solving with {solver.SolverVersion()}")
    status = solver.Solve()

    if status == pywraplp.Solver.OPTIMAL:
        print("Solution:")
        print(f"Objective value = {solver.Objective().Value():0.1f}")
        print(f"x = {x.solution_value():0.1f}")
        print(f"y = {y.solution_value():0.1f}")
    else:
        print("The problem does not have an optimal solution.")

    print("\nAdvanced usage:")
    print(f"Problem solved in {solver.wall_time():d} milliseconds")
    print(f"Problem solved in {solver.iterations():d} iterations")

def test_gmm_distance():
    # Test with python optimal trasport lib
    import numpy as np
    import ot

    # Mean and covariance matrix of the first 2D Gaussian distribution
    mean1 = np.array([0, 0])
    cov1 = np.array([[1, 0], [0, 1]])

    # Mean and covariance matrix of the second 2D Gaussian distribution
    mean2 = np.array([2, 2])
    cov2 = np.array([[1, 0], [0, 1]])

    # Compute the squared Wasserstein-2 distance
    wasserstein_2_distance_squared = ot.gaussian_wasserstein_distance(mean1, cov1, mean2, cov2)

    # Compute the Wasserstein-2 distance by taking the square root
    wasserstein_2_distance = np.sqrt(wasserstein_2_distance_squared)
    print("Wasserstein 2 distance", wasserstein_2_distance)
    
def test_wasserstein_distance():
    import numpy as np
    from scipy.linalg import sqrtm

    def wasserstein_2_distance(mean1, cov1, mean2, cov2):
        # Calculate the squared difference between the means
        mean_diff = np.linalg.norm(mean1 - mean2)**2

        # Calculate the square root of the covariance matrices
        cov1_sqrt = sqrtm(cov1)

        # Calculate the trace term
        cov_prod_sqrt = sqrtm(cov1_sqrt @ cov2 @ cov1_sqrt)
        trace_term = np.trace(cov1 + cov2 - 2 * cov_prod_sqrt)

        # Wasserstein-2 distance squared
        wass_dist_squared = mean_diff + trace_term


        return np.sqrt(wass_dist_squared)

    # Mean and covariance matrix of the first 2D Gaussian distribution
    mean1 = np.array([0, 0])
    cov1 = np.array([[1, 0], [0, 1]])

    # Mean and covariance matrix of the second 2D Gaussian distribution
    mean2 = np.array([2, 2])
    cov2 = np.array([[1, 0], [0, 1]])

    # Compute the Wasserstein-2 distance
    wass_dist = wasserstein_2_distance(mean1, cov1, mean2, cov2)
    print("wasserstein distance:", wass_dist)

if __name__ == "__main__":
    test_gaussian_prm()
    # test_ortools()
    # test_gmm_distance(r)
    # test_wasserstein_distance()
