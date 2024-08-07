"""
    Gaussian Mixture Model
"""

import numpy as np
from scipy.linalg import sqrtm
from scipy.stats import multivariate_normal, wasserstein_distance_nd, gaussian_kde

def gaussian_wasserstein_distance(mean1, cov1, mean2, cov2):
    """
        Return the Wasserstein-2 distance between two gaussian distributions
        Reference: 
        https://djalil.chafai.net/blog/2010/04/30/wasserstein-distance-between-two-gaussians/
    """
    mean_diff = np.linalg.norm(mean1 - mean2)

    sqrt_cov1 = sqrtm(cov1)

    cov_prod = sqrt_cov1 @ cov2 @ sqrt_cov1

    sqrt_cov_prod= sqrtm(cov_prod)

    sqrt_cov_prod = np.real(sqrt_cov_prod)

    trace_term = np.trace(cov1 + cov2 - 2 * sqrt_cov_prod)

    W2_distance = np.sqrt(mean_diff**2 + trace_term)

    return W2_distance
    
class GaussianNode:
    """
        Gaussian Node
    """
    def __init__(self, pos, covariance, type="UNIFORM") -> None:
        self.mean = pos 
        self.type = type
        self.covariance = covariance 

    def get_pos(self):
        """
            Return the location of the point
        """
        return self.pos

    def get_gaussian(self):
        return self.mean, self.covariance

class GaussianMixtureModel:
    """
        Gaussian Mixture Model that represents the distribution of groups of 
        agents on the map. Each GMM is represented with mean, covariance and 
        weight coefficient, such that the weight coefficients sum up to 1.
    """
    def __init__(self, means=[], covs=[], weights=[]) -> None:
        self.means = means 
        self.covs = covs
        self.weights = weights
        self.count = len(self.norms)
    
    def get_gaussians(self):
        return zip(self.norms, self.covs)
    
    def __len__(self):
        return len(self.norms)
    
    @staticmethod
    def compute_distance(gmm_1, gmm_2):
        """
            Compute the Alternative Wasserstein Distance for two GMMs
            
            $$
                D(\rho1, \rho2) = 
                \left\{\min_{\pi \in \Pi(\omega_1, \omega_2)}\sum_{i=1}^{N_1}\sum_{j=1}^{N_2}\left[W_2(g_1^i, g_2^j)\right]^2\pi(i, j)\right\}
            $$
            Reference:
            Y. Chen, T. T. Georgiou, and A. Tannenbaum, “Optimal transport for
            gaussian mixture models,” IEEE Access, vol. 7, pp. 6269–6278, 2018
        """
        
        # compute pairwise Wasserstein Distance
        w_distance = np.zeros((len(gmm_1), len(gmm_2)))

        for i, (mean1, cov1) in enumerate(gmm_1.get_gaussians()):
            for j, (mean2, cov2) in enumerate(gmm_2.get_gaussians()):
                w_distance[i][j] = gaussian_wasserstein_distance(mean1, cov1, mean2, cov2)
        
        




if __name__ == "__main__":
    pass