"""
    CVT Utils
"""
import cvxpy as cp
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import Voronoi
from scipy.stats import chi2
from shapely.affinity import affine_transform
from shapely.geometry import Polygon
from shapely.geometry.polygon import orient

from swarm_prm.utils.gaussian_utils import GaussianGraphNode

class CVT:
    """
        Centroidal Voronoi Tessellation
    """
    def __init__(self, roadmap, num_samples=200, iteration=100, 
                 confidence_interval=0.95,
                 segment_length=3):

        self.roadmap = roadmap
        self.num_samples = num_samples
        self.iteration = iteration
        self.confidence_interval = confidence_interval
        self.segment_length = segment_length
    
    def compute_centroids(self, vor, points):
        """
        Compute centroids of the Voronoi cells that do not intersect with obstacles.
        Keep original points for cells that do intersect or go outside the bounding polygon.
        """

        bounding_polygon = self.roadmap.get_bounding_polygon()
        new_points = []

        for point, region_index in zip(points, vor.point_region):
            region = vor.regions[region_index]
            if not region or -1 in region:
                new_points.append(point)  # Keep original point for infinite regions
                continue
            
            # Create a polygon for the Voronoi cell
            region_vertices = [vor.vertices[i] for i in region]
            cell_polygon = Polygon(region_vertices)

            # If the region intersects with any obstacle, keep the original point
            if self.roadmap.is_sampling_geometry_collision(cell_polygon):
                new_points.append(point)
            else:
                # Calculate the centroid of the cell
                if cell_polygon.is_valid and not cell_polygon.is_empty:
                    centroid = cell_polygon.centroid
                    # Ensure the centroid is inside the bounding polygon
                    if bounding_polygon.contains(centroid):
                        new_points.append(centroid.coords[0])
                    else:
                        # If centroid is outside, project it to the boundary point 
                        closest_point = bounding_polygon.exterior.interpolate(bounding_polygon.exterior.project(centroid))
                        new_points.append(closest_point.coords[0])
                else:
                    new_points.append(point)  # Fallback to original point if invalid
        return np.array(new_points)
    
    def get_CVT(self):
        """
            Return the centroids and the ellipsoids
        """
        np.random.seed(0)

        samples = []
        g_nodes = []

        width = self.roadmap.width
        height = self.roadmap.height

        points = []
        while len(points) < self.num_samples:
            point = (np.random.rand()*width, np.random.rand()*height)
            while self.roadmap.is_sampling_point_collision(point):
                point = (np.random.rand()*width, np.random.rand()*height)
            points.append(point)
        points = np.array(points)

        obstacles = self.roadmap.obstacles + self.roadmap.sampling_obstacles

        boundary_points = self.roadmap.get_boundary_points(obstacles, self.segment_length)
        points = np.concat([points, boundary_points])
        for _ in range(self.iteration):
            voronoi = Voronoi(points)
            points = self.compute_centroids(voronoi, points)

        voronoi = Voronoi(points)
        for region_idx in voronoi.regions:
            if not region_idx or -1 in region_idx:
                continue
            
            region = [voronoi.vertices[i] for i in region_idx]
            cell_polygon = Polygon(region)
            if self.roadmap.is_sampling_geometry_collision(cell_polygon):
                continue

            if cell_polygon.is_valid and not cell_polygon.is_empty:
                B, d = johns_ellipsoid_edge_constraints(cell_polygon)

                # convert ellipsoid into Gaussian Node

                chi2_val = chi2.ppf(self.confidence_interval, df=2)
                mean = d
                assert B[0] is not None, "Invalid polygon."
                cov = B.T @ B / chi2_val
                samples.append(np.array(mean))
                g_nodes.append(GaussianGraphNode(mean, cov))
        
        return samples, g_nodes

# Polygon -> Gaussian Functions

def get_polygon_inequalities(polygon: Polygon):
        """
        Computes the inequality constraints (Ax ≤ b) for a convex polygon.
        Returns:
            A: (m, d) array of normal vectors for each edge.
            b: (m, ) array of constraint bounds.
        """
        vertices = np.array(polygon.exterior.coords)  # Remove duplicate last point
        edges = np.diff(vertices, axis=0)  # Edge vectors
        normals = np.column_stack([-edges[:, 1], edges[:, 0]])  # Normal vectors
        normals = -normals / np.linalg.norm(normals, axis=1, keepdims=True)  # Normalize

        b = np.sum(normals * vertices[:-1], axis=1)  # Compute b = a_i^T v_i
        return normals, b

def get_normalized_polygon(polygon: Polygon, scale=10):
    """
    Normalize a polygon to the unit circle.
    """
    polygon = orient(polygon, sign=1.0)  # Ensure counter-clockwise orientation
    centroid = np.array(polygon.centroid.xy)
    bounds = np.array(polygon.bounds)
    x_scale = bounds[2] - bounds[0] 
    y_scale = bounds[3] - bounds[1] 
    transform_matrix = np.array((scale / x_scale, scale / y_scale))
    offset = np.array([-centroid[0]* scale/x_scale , -centroid[1] * scale/y_scale])
    return affine_transform(polygon, [scale /x_scale, 0, 0, scale/y_scale, offset[0], offset[1]]), transform_matrix, offset, scale

def johns_ellipsoid_edge_constraints(polygon: Polygon):
    """
    Computes John's Ellipsoid (largest inscribed ellipsoid) for a convex polygon using CVXPY
    with polygon edge constraints instead of vertex constraints.
    """
    
    normalized_polygon, transform_matrix, offset, scale = get_normalized_polygon(polygon)
    A, b = get_polygon_inequalities(normalized_polygon)  # Compute polygon inequalities
    dim = A.shape[1]  # Dimension (should be 2 for 2D)

    # Define optimization variables
    B = cp.Variable((dim, dim), symmetric=True)  
    
    d = cp.Variable((dim,))  # Center of the ellipsoid

    # Ensure P is positive definite
    constraints = [ B >> 1e-3 * np.eye(dim)]

    # Edge constraints: max_{x ∈ E} a_i^T x ≤ b_i
    for i in range(A.shape[0]):
        constraints.append(cp.norm(B @ A[i].T, 2) <= b[i] - A[i].T @ d) 

    # log det (B^-1) = -log det (B)
    obj = cp.Maximize(cp.log_det(B))

    # Solve the convex optimization problem
    prob = cp.Problem(obj, constraints) # type: ignore
    prob.solve()

    if prob.status in ["optimal", "optimal_inaccurate"]:
        # Transform the ellipsoid back to the original space
        t = np.diag(1/transform_matrix)
        return t@B.value, -t@(offset.squeeze() - d.value) # type: ignore

    else:
        print(prob.status)
        return None, None

# Visualization Functions

def plot_ellipsoid(ax, center, A_matrix, color='r'):
    """
        Plot ellipsoid
    """
    theta = np.linspace(0, 2 * np.pi, 100)
    ellipse = np.array([np.cos(theta), np.sin(theta)])
    
    # Transform unit circle to ellipse
    L = np.linalg.cholesky(A_matrix)
    ellipse = L @ ellipse + center[:, None]

    ax.plot(ellipse[0, :], ellipse[1, :], color)

def plot_voronoi(voronoi, bounding_polygon, obstacles):
    """
        Plot Voronoi and inscribed ellipsoids
    """
    plt.figure(figsize=(10, 10))

    ax = plt.gca()
    # Plot Voronoi cells
    for region_index in voronoi.regions:
        if not region_index or -1 in region_index:
            continue  # Skip infinite regions
        
        region = [voronoi.vertices[i] for i in region_index]
        cell_polygon = Polygon(region)
        
        # Skip regions that intersect with any obstacle
        if any(cell_polygon.intersects(obs) for obs in obstacles):
            continue
        
        # Plot the Voronoi cell
        if cell_polygon.is_valid and not cell_polygon.is_empty:
            x, y = cell_polygon.exterior.xy
            plt.fill(x, y, alpha=0.4, edgecolor='k')
            center, A_matrix = johns_ellipsoid_edge_constraints(orient(cell_polygon)) # FIXIT
            plot_ellipsoid(ax, center, A_matrix)

    # Plot Voronoi sites
    # plt.plot(voronoi.points[:, 0], voronoi.points[:, 1], 'ro')

    # Plot bounding polygon
    x, y = bounding_polygon.exterior.xy
    plt.plot(x, y, 'b-', lw=2)

    # Plot obstacles
    for obs in obstacles:
        x, y = obs.exterior.xy
        plt.fill(x, y, color='gray', alpha=0.7)

    plt.xlim(0, 100)
    plt.ylim(0, 100)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.title("Voronoi Diagram Using CVT with Obstacles")
    plt.show()
