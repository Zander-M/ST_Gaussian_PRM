"""
    Gaussian PRM based on map info.
"""
import random
from collections import namedtuple

import numpy as np
from scipy.stats import gaussian_kde

Point = namedtuple("Point", ["x", "y"])

    
class GaussianPRM:
    """
        Gaussian PRM
    """

    class GaussianNode:
        """
            Gaussian Node
        """
        def __init__(self, pos, cvar_thresh) -> None:
            self.pos = pos
            self.cvar_thresh = cvar_thresh

        def compute_distribution(self):
            """
                Compute the current distribution based on distance to closest obstacle
            """

        def get_CVaR(self):
            """
                Return CVaR value
            """
            return self.cvar_thresh

        def get_capacity(self, density):
            """
                Return capacity of the node
            """
            return 
            pass

        def get_gaussian(self):
            pass

    def __init__(self, map_size, obstacles) -> None:
        self.map_size = map_size
        self.obstacles = obstacles
        self.abstract_prm = None
        self.gaussian_prm = None
        pass
    
    def point_collision(self, point):
        pass
        
    def edge_collision(self, p1, p2):
        pass

    def sample_free_space(self)

    def build_prm(self):
        """
            Build Gaussian PRM. Construct regular PRM, then convert the 
            nodes to Gaussian Nodes.
        """

        pass


    def astar_search(self, start, goal):
        """
            A star search on abstract graph
        """
        pass

    def gmm_search(self, start, goal):
        """
            GMM search on Gaussian PRM
        """
        pass