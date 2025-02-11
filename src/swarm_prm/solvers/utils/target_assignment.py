"""
    Assign Start-Goal pairs for each agent using Euclidean distance as heuristics
"""

from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment

from swarm_prm.solvers.utils import GaussianPRM, sample_gaussian

class TargetAssignment:
    def __init__(self, start_agent_count, goal_agent_count, gaussian_prm:GaussianPRM):

        self.start_agent_count = start_agent_count
        self.goal_agent_count = goal_agent_count
        self.gaussian_prm = gaussian_prm

    def assign(self):
        """
            Return start-goal pairs for each agent
        """
        return None, None
        pass