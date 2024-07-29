"""
    Gaussian Mixture Model
"""

import numpy as np
from scipy.stats import multivariate_normal, wasserstein_distance_nd

class GaussianMixtureModel:
    """
        Gaussian Mixture Model that represents the distribution of groups of 
        agents on the map. Each GMM is represented with mean, covariance and 
        weight coefficient, such that the weight coefficients sum up to 1.
    """
    def __init__(self, norms=[], covs=[], weights=[]) -> None:
        self.norms = norms
        self.covs = covs
        self.weights = weights
    
    def get_gaussians(self):
        return self.gaussians
    
    def add_gaussian(self, gaussian, weight):
        """
           Adding gaussian distribution to the GMM. Update weights wrt the weight
           parameter
        """

    def get_samples(self, num_samples):
        """
            Samples from the GMM. Represent the 
        """
        pass

    def __len__(self):
        return len(self.norms)
    
    @staticmethod
    def compute_distance(gmm_1, gmm_2):
        """
            Compute the Alternative Wasserstein Distance for two GMMs
            
            Reference:
            Y. Chen, T. T. Georgiou, and A. Tannenbaum, “Optimal transport for
            gaussian mixture models,” IEEE Access, vol. 7, pp. 6269–6278, 2018
        """
        pass

if __name__ == "__main__":
    pass