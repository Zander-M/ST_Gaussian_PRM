"""
    Find the shortest time to travel through the graph by incrementally updating
    Time Expanded Graph (TEG) and checking for the max flow of the graph
"""

import numpy as np
from scipy.sparse.csgraph import maximum_flow

from swarm_prm.solvers.swarm_prm.macro.gaussian_prm import GaussianPRM

INF_FLOW = 1e9 # used as infinite flow edge capacity 

class TEGNode():
    """
        Time expanded graph node
    """
    def __init__(self, capacity, timestep, neighbors) -> None:
        self.capacity = capacity
        self.neighbors = neighbors
        self.timestep = timestep

    def get_neighbors(self):
        """
            Return neighboring nodes with updated 
        """

class TEGGraph():
    """
        Time expanded graph
    """
    def __init__(self) -> None:
        pass

    def construct_initial_teg(self, gaussian_prm:GaussianPRM):
        """
            Contruct initial teg
        """
    
    def update_teg(self):
        """
            Update TEG by adding one timestep.
        """

    def find_max_flow(self):
        """
            Find max flow of the current graph
        """

    def to_csgraph(self):
        """
            Convert TEG graph to csgraph for scipy solution
        """
        pass

