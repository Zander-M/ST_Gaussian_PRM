"""
    DRRT for continuous space motion planning
    https://arxiv.org/pdf/1903.00994
    This version does not have rewiring behavior
"""

import heapq
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import KDTree

from swarm_prm.solvers.utils.gaussian_prm import GaussianPRM
from swarm_prm.solvers.utils.spatial_hash import SpatialHash
from swarm_prm.solvers.utils.target_assignment import TargetAssignment

class DRRT:
    def __init__(self, gaussian_prm:GaussianPRM, num_agents, max_iter=30000):
        """
            We use the same roadmap for multiple agents. If a Gaussian node
            does not exceed its capacity, we do not consider it a collision.

            State is represented as a list of agent node indices.
        """
        self.gaussian_prm = gaussian_prm
        self.num_agents = num_agents
        self.kd_tree = KDTree(self.gaussian_prm.samples)
        self.roadmap = self.gaussian_prm.roadmap
        self.max_iter = max_iter

        # Initialize problem instance
        self.start_agent_count = [int(w*self.num_agents) for w in self.gaussian_prm.starts_weight]
        self.goal_agent_count = [int(w*self.num_agents) for w in self.gaussian_prm.goals_weight]
        self.ta = TargetAssignment(self.start_agent_count, self.goal_agent_count, self.gaussian_prm)

        # Initialize dRRT 

        self.start, self.goal = self.ta.assign()
            
        self.drrt = {
            "cost": 0,
            "state": self.start, 
            "parent": None
        }

    def connect_to_target(self):
        """
            Connect currect tree to target
        """
        
    def expand_drrt(self):
        """
            Expand RRT Star 
        """

    def sample_new_state(self):
        """
            Sample New state
        """
        
    def find_nearest_neighbor(self, node):
        """
            Find Nearest Neighbor in the tree
        """

    def get_cost(self, node):
        """
            Compute cost of node
            TODO: update cost
        """
        return 0
    
    def Id(self):
        """
            Oracle steering function
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
        best_path = None
        best_cost = float("inf")
        for _ in range(self.max_iter):
            self.expand_drrt()
        path = self.connect_to_target()
        cost = self.get_cost(path)
        if path is not None and cost < best_cost:
            best_path = path
            best_cost = cost
        return best_path, best_cost

    def verify_node(self, node):
        """
            Verify if the new state is valid
        """
        pass

    def verify_connect(self, node1, node2):
        """
            Verify if two states can be connected
        """
        pass