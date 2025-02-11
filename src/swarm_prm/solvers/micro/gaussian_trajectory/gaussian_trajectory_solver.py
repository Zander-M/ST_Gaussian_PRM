"""
    Finding single agent trajectory by sampling points in Gaussian
    and finding best matching pairs in each group
"""
import numpy as np
from scipy.spatial import KDTree
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment
from scipy.stats import chi2

def is_within_ci(point, g_node, chi2_thresh):
    delta = point - g_node.get_mean()
    mahalanobis_distance_square = delta.T @ np.linalg.inv(g_node.cov) @ delta
    return mahalanobis_distance_square <= chi2_thresh

def sample_gaussian(g_node, num_points, ci, min_spacing, candidates=None):

    """
        Return Gaussian Samples within the confidence interval
    """
    points = []
    tree = KDTree([g_node.get_mean()])
    chi2_thresh = chi2.ppf(ci, 2)
    if candidates is None:
        num_candidates = 1000
        mean, cov = g_node.get_gaussian()
        candidates = np.random.multivariate_normal(mean, cov, num_candidates)
    for candidate in candidates:
        if len(points) == num_points:
            break
        if  is_within_ci(candidate, g_node, chi2_thresh) \
            and tree.query(candidate)[0] > min_spacing :
            points.append(candidate)
            tree = KDTree(points)
    return np.array(points)

class GaussianTrajectorySolver:

    def __init__(self, gaussian_prm, macro_trajectory, agent_radius, 
                 safety_gap = 0.2, ci=0.8,
                 obs_thresh=1, max_dist=8, max_init_attempt=100,
                 ):
        self.gaussian_prm= gaussian_prm 
        self.macro_trajectory = macro_trajectory
        self.agent_radius = agent_radius
        self.num_agent = len(self.macro_trajectory)
        self.gaussian_candidates = []

        # Generate Candidate points for each Gaussian Node
        for g_node in self.gaussian_prm.gaussian_nodes:
            self.gaussian_candidates.append(np.random.multivariate_normal(g_node.get_mean(), g_node.cov, self.num_agent))

    def solve(self):
        """
            Find trajectories per timestep. We find start-goal pairs based on minimizing
            sum of distances between start and goal points.
        """

