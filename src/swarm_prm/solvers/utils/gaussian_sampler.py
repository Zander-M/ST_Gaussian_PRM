"""
    Given a 2D Gaussian distribution, return 
"""
import numpy as np
from scipy.spatial import KDTree
from scipy.stats import chi2

def is_within_ci(point, mean, cov, chi2_thresh):
    delta = point - mean
    mahalanobis_distance_square = delta.T @ np.linalg.inv(cov) @ delta
    return mahalanobis_distance_square <= chi2_thresh

def gaussian_sampler(mean, cov, num_points, ci, min_spacing):

    """
        Return Gaussian Samples within the confidence interval
    """
    points = [mean]
    num_samples = 5000
    tree = KDTree(points)
    samples = np.random.multivariate_normal(mean, cov, num_samples)
    chi2_thresh = chi2.ppf(ci, 2)
    for sample in samples:
        if len(points) == num_points:
            return points 
        if  is_within_ci(sample, mean, cov, chi2_thresh) \
            and tree.query(sample)[0] > min_spacing :

            points.append(sample)
            tree = KDTree(points)
    return np.array(points)

def visualize_gaussian(mean, cov, samples=None, ci=None):
    """
        Visualize Gaussian Distribution, Samples and Confidence Interval
    """
    pass

def visualize_matching(mean1, cov1, mean2, cov2, samples1, samples2, ci):
    """
        Visualize the matching between two Gaussian Nodes
    """
    pass