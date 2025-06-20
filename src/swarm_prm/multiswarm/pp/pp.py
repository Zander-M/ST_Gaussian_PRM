"""
    Prioritized Planning
"""
import numpy as np

class PrioritizedPlannng:
    def __init__(self, gaussian_prm, instances, planner):
        self.gaussian_prm = gaussian_prm
        self.instances = instances
        self.order = np.arange(len(self.instances))
        self.planner = planner
        self.flow_constraints = None

    def update_flow_constraint(self, flow_dict):
        """
            Update flow constraint
        """

    def plan(self):
        """
            Sequentially plan for each instance.
        """
        for instance in self.instances:

            pass