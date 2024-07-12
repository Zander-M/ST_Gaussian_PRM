"""
    Gaussian PRM based on map info.
"""
import numpy as np
from scipy.stats import gaussian_kde

class GaussianNode:
    """
        Gaussian Node
    """
    def __init__(self) -> None:
        pass

    def get_capacity(self):
        pass

    def get_gaussian(self):
        pass
    
class GaussianPRM:
    """
        Gaussian PRM
    """
    def __init__(self) -> None:
        pass

    def build_gaussian_prm(self):
        """
            Build Gaussian PRM
        """
        pass

    def abstract_prm(self):
        """
            Construct abstract PRM for path search. Node locations are the center
            of the Gaussian distributions.
        """
        pass

    def astar_search(self, start, goal):
        """
            A star search on abstract graph
        """
        pass

    def gmm_search(self, start, goal):
        """
            GMM search on Gaussian PRM
        """
        pass