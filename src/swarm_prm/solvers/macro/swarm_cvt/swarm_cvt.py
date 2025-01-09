"""
    SwarmCVT based on https://arxiv.org/abs/2410.02510
"""
import numpy as np


from swarm_prm.solvers.utils.gaussian_prm import GaussianPRM
from swarm_prm.solvers.utils.gaussian_utils import *

class SwarmCVT:
    def __init__(self, gaussian_prm):
        self.gaussian_prm = gaussian_prm

    def find_shortest_paths(self):
        """
            Find shortest paths between starts and goals.
        """
    
    def find_path_weights(self):
        """
            Find path weights for each node w.r.t. the density/capacity constraint
        """
    
    def get_solution(self):
        """
            Get simple path per agent to 
        """