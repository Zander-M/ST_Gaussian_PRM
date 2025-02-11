"""
    Using RRT to find trajectories between two Gaussian states
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import KDTree

from swarm_prm.envs.map import Map, Obstacle
from swarm_prm.solvers.utils.gaussian_utils import *

class RRTNode:
    """
        RRT node
    """
    def __init__(self, g_node:GaussianNode, parent) -> None:
        self.g_node = g_node
        self.parent = parent
        self.cost = 0.0

class GaussianRRT:
    """
        Using RRT to find feasible Gaussian Path between nodes
    """
    def __init__(self, map, start, goal, step_size=0.5,
                 goal_thresh=0.5, sample_goal_prob=0.5,
                 max_iter=500) -> None:
        self.map = map
        self.start = RRTNode(start, None)
        self.goal = RRTNode(goal, None)
        self.step_size = step_size
        self.goal_thresh = goal_thresh
        self.sample_goal_prob = sample_goal_prob
        self.max_iter = max_iter
        self.kd_tree = KDTree([start.g_node.mean]) 

        self.gaussian_nodes = []

        self.simple_path = []
        self.gaussian_path = []
    
    def get_near_nodes(self, new_node, radius):
        """
            get nearest node
        """
        indicies = self.kd_tree.query_ball_point(new_node.g_node.mean, radius)
        return [self.gaussian_nodes[i] for i in indicies]

    def construct_path(self):
        path = []
        node = self.goal
        while node is not None:
            path.append(node.g_node)
            node = node.parent
        return path[::-1]

    def rewire(self, new_node, near_nodes):
        """
            rewire nodes
        """

    def plan(self):
        """
            Randomly sample Gaussian distributions and 
        """
        curr_node = self.start
        iter = 0
        while iter < self.max_iter:
            pass
    
    def steer(self, from_node, to_node):
        """
            Steer node 
        """
        pass

    def sample(self):
        """
            Sample New node
        """
        
        pass

    def visualize_trajectory(self):
        """
            Visualize solution path
        """
        pass

    