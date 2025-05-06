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
    
def covex_set_intersection(cn1, cn2):
    """
        return polygon of the intersection of convex sets
    """
    return shapely.intersection(cn1.poly, cn2.poly)

@register_solver("FormationControlSolver")
class FormationControlSovler(MacroSolverBase):
    def init_solver(self, **kwargs):
        """
            Solver specific initialization
        """
        self.iris_sampler = IrisSampler(self.gaussian_prm)
        self.confidence_interval = kwargs.get("confidence_interval", 0.95)
        self.overlap_thresh = kwargs.get("overlap_thresh", 0.7)

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
        while time.time() - start_time < self.time_limit:
            sample = self.sample_free_point()
            sample_poly = self.iris_sampler.sample(sample)
            # check if intersect with existing polygons
            for convex_node in convex_nodes:
                # check if the region intersects with the current polygon, and the polygon should not
                # "largely" overlap with the existing polygon
                if self.check_connection(sample_poly, convex_node.poly):
                    convex_nodes.append(ConvexNode(sample_poly, convex_node))

                    # check goal state
                    if self.check_connection(sample_poly, goal_poly):
                        # Adding goal polygon
                        convex_nodes.append(ConvexNode(goal_poly, convex_nodes[-1]))
                        g_nodes, paths, starts_idx, goals_idx, convex_nodes = self.get_solution(convex_nodes[-1])
                        return {
                            "success": True,
                            "g_nodes": g_nodes,
                            "paths" : paths,
                            "starts_idx": starts_idx,
                            "goals_idx": goals_idx,
                            "start_poly": start_poly,
                            "goal_poly": goal_poly,
                            "convex_nodes": convex_nodes
                        }
                    else:
                        break
        
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

    def get_solution(self, goal_convex_set):
        """
            Return Gaussian nodes used by each agent and the Gaussian nodes
        """
        convex_nodes = []
        curr_convex_set = goal_convex_set
        while curr_convex_set.parent is not None:
            convex_nodes.append(curr_convex_set)
            curr_convex_set = curr_convex_set.parent
        convex_nodes.append(curr_convex_set)
        convex_nodes = convex_nodes[::-1]
        g_nodes = []

        # add start nodes and goal nodes
        node_idx = 0
        starts_idx = []
        goals_idx = []

        for start in self.starts:
            g_nodes.append(self.gaussian_prm.gaussian_nodes[start])
            starts_idx.append(node_idx)
            node_idx += 1 
        
        for goal in self.goals:
            g_nodes.append(self.gaussian_prm.gaussian_nodes[goal])
            goals_idx.append(node_idx)
            node_idx += 1

        for (prev_cs, curr_cs) in zip(convex_nodes[:-1], convex_nodes[1:]):
            poly = covex_set_intersection(prev_cs, curr_cs)
            g_nodes.append(self.polygon_to_gaussian_node(poly))

        # construct individual agent paths
        starts = []
        goals = []
        for i, start_idx in enumerate(starts_idx):
            starts += [start_idx] * self.starts_agent_count[i]

        for i, goal_idx in enumerate(goals_idx):
            goals += [goal_idx] * self.goals_agent_count[i]

        paths = [[starts[j]]+[i for i in range(node_idx, len(g_nodes))]+[goals[j]] for j in range(self.num_agents)]

        return g_nodes, paths, starts_idx, goals_idx, convex_nodes

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
        
