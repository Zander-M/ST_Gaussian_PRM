"""
    Voronoi Utils
"""
import cvxpy as cp
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import Voronoi
from shapely.geometry import Polygon
from shapely.geometry.polygon import orient

class CVT:
    """
        Centroidal Voronoi Tessellation
    """
    def __init__(self, roadmap, num_samples=200, iteration=10, cvar_threshold=0.95):

        self.roadmap = roadmap
        self.num_samples = num_samples
        self.iteration = iteration

        # determine the CVaR threshold as the Gaussian Nodes equiprobable lines wrt the ellipsoids
        self.cvar_threshold = cvar_threshold 
    
    def _compute_centroids(self, vor, points):
        """
        Compute centroids of the Voronoi cells that do not intersect with obstacles.
        Keep original points for cells that do intersect or go outside the bounding polygon.
        """

        bounding_polygon = self.roadmap.get_bounding_polygon
        obstacles = self.roadmap.get_obstacles

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
            if any(cell_polygon.intersects(obs) for obs in obstacles):
                new_points.append(point)
            else:
                # Calculate the centroid of the cell
                if cell_polygon.is_valid and not cell_polygon.is_empty:
                    centroid = cell_polygon.centroid
                    # Ensure the centroid is inside the bounding polygon
                    if bounding_polygon.contains(centroid):
                        new_points.append(centroid.coords[0])
                    else:
                        # If centroid is outside, keep the original point
                        # Alternatively, you could project it back onto the bounding polygon boundary
                        closest_point = bounding_polygon.exterior.interpolate(bounding_polygon.exterior.project(centroid))
                        new_points.append(closest_point.coords[0])
                else:
                    new_points.append(point)  # Fallback to original point if invalid

        return np.array(new_points)
    
    def get_gaussian_nodes(self, A_matrix, center):
        """
            Get the Gaussian nodes based on Ellipsoid A_matrix and center
        """
        
    def get_CVT(self):
        """
            Return the centroids and the ellipsoids
        """
        np.random.seed(0)
        width = self.roadmap.size[0]
        height = self.roadmap.size[1]

        x = np.random.uniform(0, width, self.num_samples)
        y = np.random.uniform(0, height, self.num_samples)
        points = np.column_stack((x, y))
        for _ in range(self.iteration):
            voronoi = Voronoi(points)
            points = self._compute_centroids(voronoi, points)
        
        # Compute the ellipsoids

    def get_polygon_inequalities(self, polygon: Polygon):
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


    def johns_ellipsoid_edge_constraints(self, polygon: Polygon):
        """
            Computes John's Ellipsoid (largest inscribed ellipsoid) for a convex polygon using CVXPY
            with polygon edge constraints instead of vertex constraints.
        """

        A, b = self.get_polygon_inequalities(polygon)  # Compute polygon inequalities
        d = A.shape[1]  # Dimension (should be 2 for 2D)

        # Define optimization variables
        P = cp.Variable((d, d), symmetric=True)  # Instead of A^-1, use P as auxiliary matrix
        c = cp.Variable((d, ))  # Center of the ellipsoid
        t = cp.Variable()  # Scaling factor to ensure the ellipsoid is inside the polygon

        # Ensure P is positive semi-definite
        constraints = [cp.PSD(P)]

        # Edge constraints: max_{x ∈ E} a_i^T x ≤ b_i
        for i in range(A.shape[0]):
            constraints.append(cp.norm(P @ A[i]) <= b[i] - A[i] @ c)  # Equivalent to a_i^T x ≤ b_i

        # Objective: Maximize log(det(P)), equivalent to maximizing log(det(A))
        obj = cp.Maximize(cp.log_det(P))

        # Solve the convex optimization problem
        prob = cp.Problem(obj, constraints)
        prob.solve()

        if prob.status in ["optimal", "optimal_inaccurate"]:
            P_value = P.value
            # A_value = np.linalg.inv(P_value) if P_value is not None else None  # Recover A from P^-1
            A_value = P.value

            return c.value, A_value
        else:
            print(prob.status)
            raise ValueError("Optimization failed.")
    
    def get_gaussian_roadmap(self):
        """
            Get the Gaussian roadmap
        """

        return 
        
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

# Visualization Functions
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