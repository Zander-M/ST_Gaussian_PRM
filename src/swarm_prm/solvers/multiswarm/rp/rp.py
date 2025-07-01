"""
    Random Prioritized Planning
"""
import time

import numpy as np

class RandomPriority:
    """
        Random total priority ordering.
    """
    def __init__(self, gaussian_prm, instances, planner, time_limit=180):
        self.gaussian_prm = gaussian_prm
        self.instances = instances
        self.order = np.arange(len(self.instances))
        self.planner = planner
        self.flow_constraints = None
        self.time_limit = time_limit

    def load_instance(self, instance):
        """
            Convert instance parameters to a planning instance
        """
        
    def plan(self):
        """
            Sequentially plan for each instance.
        """
        swarms = []
        solution_found = [False for _ in swarms]
        start_time = time.time()
        while time.time() - start_time < self.time_limit:
            
            capacity_dicts = []
            obstacle_goal_dicts = []
            flow_dicts = []

            for instance in self.instances:
                pass