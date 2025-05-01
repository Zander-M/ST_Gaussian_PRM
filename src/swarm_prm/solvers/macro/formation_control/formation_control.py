"""
    Formation Control solver.
    We assume the formation of the agents follows a 2D Gaussian distribution.
    The intermediate formations are the inscribed ellipse of the intersecting
    polytopes.
    Works only for 2D at the moment
"""
import time

import numpy as np
import shapely
from scipy.stats import chi2


from swarm_prm.solvers.macro import MacroSolverBase, register_solver
from swarm_prm.solvers.macro.formation_control import IrisSampler
from swarm_prm.utils import johns_ellipsoid_edge_constraints, GaussianGraphNode

class ConvexNode:
    """
        Convex Node for planning 
    """
    def __init__(self, poly, parent):
        self.poly = poly 
        self.parent = parent
    
    def get_centroid(self):
        """
            Return centroid of the polygon
        """
        return shapely.centroid(self.poly)
    
    def is_intersect(self, poly):
        """
            Return true if a polygon is intersecting with
            current convex set
        """
        return shapely.intersects(poly, self.poly)

@register_solver("FormationControlSolver")
class FormationControlSovler(MacroSolverBase):
    def init_solver(self, **kwargs):
        """
            Solver specific initialization
        """
        self.iris_sampler = IrisSampler(self.gaussian_prm)
        self.confidence_interval = kwargs.get("confidence_interval", 0.95)
        self.overlap_thresh = kwargs.get("overlap_thresh", 0.95)

    def solve(self):
        """
            Find solution paths
        """

        # initialize start formation and goal formation
        # We take the intersetion of the start nodes and goal nodes as
        # the start/goal formation
        convex_starts = [self.iris_sampler.sample(self.nodes[start]) for start in self.starts]
        start_poly = shapely.intersection_all(convex_starts)
        convex_goals= [self.iris_sampler.sample(self.nodes[goal]) for goal in self.goals]
        goal_poly = shapely.intersection_all(convex_goals)

        convex_nodes = [ConvexNode(start_poly, None)]
        start_time = time.time()
        found_solution = False
        while time.time() - start_time < self.time_limit:
            sample = self.sample_free_point()
            sample_poly = self.iris_sampler.sample(sample)
            # check if intersect with existing polygons
            for convex_node in convex_nodes:
                # check if the region intersects with the current polygon, and the polygon should not
                # "largely" overlap with the existing polygon
                if self.check_connection(sample_poly, convex_node.poly):
                    convex_nodes.append(ConvexNode(sample_poly, convex_node))
                    break

            # check goal state
            if self.check_connection(sample_poly, goal_poly):
                paths = self.get_solution_path(convex_nodes[-1])
        
        return {"success": False}

    def sample_free_point(self):
        """
            Return collision free random point on the map
        """
        x = np.random.randint(0, self.gaussian_prm.raw_map.width)
        y = np.random.randint(0, self.gaussian_prm.raw_map.height)
        if self.gaussian_prm.raw_map.is_point_collision((x, y)):
            x = np.random.randint(0, self.gaussian_prm.raw_map.width)
            y = np.random.randint(0, self.gaussian_prm.raw_map.height)
        return np.array([x, y])
    
    def check_connection(self, poly1, poly2):
        """
            Check if two polygons should be connected based on:
            1. Connectivity
            2. Overlap ratio
            3. Capacity constraints
        """
        
        if shapely.intersects(poly1, poly2):
            intersection_poly = shapely.intersection(poly1, poly2)
            if intersection_poly.area/poly1.area < self.overlap_thresh:
                g_node = self.polygon_to_gaussian_node(intersection_poly)
                if g_node.get_capacity(agent_radius=self.agent_radius) > self.num_agents:
                    return True
        return False

    def get_solution_path(self, goal_convex_set):
        """
            get solution path in the form of time-indexed Gaussian nodes
        """
        pass

    def polygon_to_gaussian_node(self, poly):
        """
            Convert Polygon to Gaussian node
        """

        B, d = johns_ellipsoid_edge_constraints(poly)

        # convert ellipsoid into Gaussian Node
        chi2_val = chi2.ppf(self.confidence_interval, df=2)
        mean = d
        assert B[0] is not None, "Invalid polygon." # type: ignore
        cov = B.T @ B / chi2_val # type: ignore
        return GaussianGraphNode(mean, cov)
        
