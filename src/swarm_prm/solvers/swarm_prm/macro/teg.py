"""
    Find the shortest time to travel through the graph by incrementally updating
    Time Expanded Graph (TEG) and checking for the max flow of the graph
"""

import numpy as np
from scipy.sparse.csgraph import maximum_flow

from swarm_prm.solvers.swarm_prm.macro.gaussian_prm import GaussianPRM

INF = 1e9 # used as infinite flow edge capacity 

class TEGGraph():
    """
        Time expanded graph
    """
    def __init__(self, nodes, edges, node_capacities) -> None:
        self.nodes = nodes
        self.edges = edges
        self.node_capacities = node_capacities

    def teg_search(self, gaussian_prm:GaussianPRM):
        """
            Find the earliest timestep that reaches the max flow
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

