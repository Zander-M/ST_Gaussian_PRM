"""
    Finding single agent trajectory by sampling points in Gaussian
    and finding best matching pairs in each group
"""
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment

from swarm_prm.solvers.utils import gaussian_sampler

class GaussianTrajectorySolver:

    def __init__(self, roadmap, macro_trajectory, agent_radius, 
                 safety_gap = 0.2, ci=0.8,
                 obs_thresh=1, max_dist=8, max_init_attempt=100,
                 ):
        self.roadmap = roadmap
        self.macro_trajectory = macro_trajectory
        self.agent_radius = agent_radius
        self.num_agent = len(self.macro_trajectory)
    
    def assign_groups(self, starts_idx, goals_idx, num_agent_starts, num_agent_goals):
        """
            assign start-goal trajectory for each agent greedily based on distance.
            Return: 
                Groups: [([agents], (start_idx, goal_idx)), ...]
        """

    def find_matching(self, points1, points2):
        """
            compute point matching given start points and goal points
        """
        dist = cdist(points1, points2) # use Euclidean distance
        row_ind, col_ind = linear_sum_assignment(dist)




    def solve(self):
        """
            Find trajectories per timestep. The algorithm works as follows:
            Sample Goal Locations -> Goal Grouping -> Best Matching based on distance 
        """

