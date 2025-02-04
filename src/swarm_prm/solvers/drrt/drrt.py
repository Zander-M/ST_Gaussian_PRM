"""
    DRRT for continuous space motion planning
    https://arxiv.org/pdf/1903.00994
    This version does not have rewiring behavior
"""

import numpy as np
import matplotlib.pyplot as plt
import heapq

from swarm_prm.solvers.utils.gaussian_prm import GaussianPRM
from swarm_prm.solvers.utils.spatial_hash import SpatialHash

class DRRT:
    def __init__(self, gaussian_prm, hash_size):
        """
            We use the same roadmap for multiple agents. If a Gaussian node
            does not exceed its capacity, we do not consider it a collision.
        """
        self.gaussian_prm = gaussian_prm
        self.hash_size = hash_size
        self.num_agents = gaussian_prm.num_agents
        self.agents = []

    def convert_to_abstract_map(self):
        """
            Convert the Gaussian PRM to an abstract map
        """
        

    def find_nearest_neighbor(self, node):
        """
            Find Nearest Neighbor in the tree
        """
    
    def rewire(self, node):
        """
            Rewire Tree
        """

    def get_cost(self, node):
        """
            Compute cost of node
        """
    
    def Id(self):
        """
            Oracle steering function
        """

    def expand_drrt_star(self):
        """
            Expand RRT Star 
        """
    
    def getParent(self, nodes):
        """
            Get node parent
        """
    
    def get_distance(self, node1, node2):
        """
            Get node distance
        """

    def get_solution(self):
        """
            Get solution per agent
        """
