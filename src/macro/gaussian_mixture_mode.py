"""
    Gaussian Mixture Model
"""

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

if __name__ == "__main__":
    pass