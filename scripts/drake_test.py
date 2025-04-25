import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt
from shapely.geometry import Polygon, box
from scipy.spatial import HalfspaceIntersection, ConvexHull

from pydrake.all import (
    SceneGraph,
    DiagramBuilder,
    RigidTransform,
    Box as DrakeBox,
    GeometryInstance,
    GeometrySet,
    HPolyhedron,
    IrisOptions,
)
from pydrake.geometry.optimization import Iris


def add_polygon_as_box(polygon, buffer=0.01):
    """
    Convert a shapely polygon into a padded AABB box as a ConvexSet (HPolyhedron).
    """
    minx, miny, maxx, maxy = polygon.bounds
    lb = np.array([[minx - buffer], [miny - buffer]])
    ub = np.array([[maxx + buffer], [maxy + buffer]])
    return HPolyhedron.MakeBox(lb, ub)

def chebyshev_center(A, b):
    """
    Compute Chebyshev center of polytope Ax <= b.
    Returns a point strictly inside the region.
    """
    dim = A.shape[1]
    x = cp.Variable(dim)
    r = cp.Variable(1)
    constraints = [A @ x + cp.norm(A[i], 2) * r <= b[i] for i in range(A.shape[0])]
    prob = cp.Problem(cp.Maximize(r), constraints)
    prob.solve()
    return x.value

def iris_hpoly_to_polygon(hpoly: HPolyhedron) -> Polygon:
    """
    Converts a 2D Drake HPolyhedron (Ax <= b) into a shapely Polygon using scipy.
    """
    A = hpoly.A()
    b = hpoly.b()
    assert A.shape[1] == 2, "Only supports 2D HPolyhedra for now"

    center = hpoly.ChebyshevCenter()

    if center is None:
        raise RuntimeError("Failed to compute Chebyshev center.")

    # Flatten interior point to shape (2,) for scipy
    interior_point = np.array(center[0]).flatten()

    # Convert Ax <= b to scipy format: [a1, a2, -b]
    halfspaces = np.hstack([A, -b.reshape(-1, 1)])

    # Compute intersections
    hs = HalfspaceIntersection(halfspaces, interior_point=center)

    # Return a convex shapely polygon
    return Polygon(hs.intersections).convex_hull


def run_iris_with_shapely(seed_point, obstacles, bounds=[[-100, 100], [-100, 100]]):
    """
    Run IRIS using shapely obstacles and return the resulting region as a shapely Polygon.
    """

    # Add each polygon as a box-shaped obstacle
    geom_ids = []
    for poly in obstacles:
        geom_id = add_polygon_as_box(poly)
        geom_ids.append(geom_id)

    # Define domain bounds as HPolyhedron
    bounds_np = np.array(bounds)
    lb = bounds_np[:, 0:1]
    ub = bounds_np[:, 1:2]
    domain = HPolyhedron.MakeBox(lb, ub)

    # Set up IRIS options
    options = IrisOptions()
    options.require_sample_point_is_contained = False

    # Compute IRIS region
    region = Iris(
        domain=domain,
        obstacles=geom_ids,
        sample=np.array(seed_point),
        options=options
    )

    return iris_hpoly_to_polygon(region)


def plot_region_and_obstacles(region_poly, obstacles, seed=None):
    """
    Plot the shapely obstacles and IRIS region.
    """
    fig, ax = plt.subplots()
    for poly in obstacles:
        x, y = poly.exterior.xy
        ax.fill(x, y, color='gray', alpha=0.5, label="Obstacle")
    x, y = region_poly.exterior.xy
    ax.fill(x, y, color='green', alpha=0.4, label="IRIS Region")
    if seed is not None:
        ax.plot(seed[0], seed[1], 'ro', label="Seed Point")
    ax.set_aspect('equal')
    ax.grid(True)
    plt.legend()
    plt.title("2D IRIS Region with Shapely Obstacles")
    plt.show()

# === Example usage ===
if __name__ == "__main__":
    obstacles = [
        box(24, 24, 50, 50),  # A box-shaped obstacle
        Polygon([(-10, -10), (-4, -20), (-30, -40)]),  # A triangle
    ]
    seed = [2, 2]
    iris_poly = run_iris_with_shapely(seed, obstacles)
    plot_region_and_obstacles(iris_poly, obstacles, seed)
