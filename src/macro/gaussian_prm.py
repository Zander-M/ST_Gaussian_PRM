"""
    Gaussian PRM based on map info.
"""
import numpy as np

class GaussianDist:
    """
        2D Gaussian Distributions
    """
    def __init__(self, mean, cov):
        self.mean = mean
        self.cov = cov 
    


class GaussianMixtureModel:
    """
        Gaussian Mixture Model that represents the distribution of groups of 
        agents on the map. Each GMM is represented with mean, covariance and 
        weight coefficient, such that the weight coefficients sum up to 1.
    """
    def __init__(self) -> None:
        self.gaussians = []
        self.weights = []
        self.n = len(self.gaussians)

class GaussianPRM:
    def __init__(self) -> None:
        pass

    def abstract_prm(self):
        """
            Construct abstract PRM for path search and 
        """
        pass

    def astar_search(self, start, goal):
        pass

    def gmm_search(self):
        pass