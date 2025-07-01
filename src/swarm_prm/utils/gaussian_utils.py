"""
    Gaussian Utils
"""

from math import floor
from matplotlib.patches import Ellipse
import numpy as np
from scipy.spatial import KDTree
from scipy.linalg import sqrtm
from scipy.stats import chi2, multivariate_normal
from shapely.geometry import Polygon

def is_within_ci(point, g_node, chi2_thresh):
    delta = point - g_node.get_mean()
    mahalanobis_distance_square = delta.T @ np.linalg.inv(g_node.cov) @ delta
    return mahalanobis_distance_square <= chi2_thresh

def sample_gaussian(g_node, candidates, num_points, ci, min_spacing):

    """
        Return Gaussian Samples within the confidence interval
        and a proper spacing
    """
    points = []
    tree = KDTree([g_node.get_mean()])
    chi2_thresh = chi2.ppf(ci, 2)
    for candidate in candidates:
        if len(points) == num_points:
            break
        if  is_within_ci(candidate, g_node, chi2_thresh) \
            and tree.query(candidate)[0] > min_spacing :
            points.append(candidate)
            tree = KDTree(points)
    return np.array(points)

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

    trace_term = np.trace(cov1 + cov2 - 2 * sqrt_cov_prod)

    W2_distance = np.sqrt(mean_diff**2 + trace_term)

    return W2_distance
    
class GaussianNode:
    """
        Gaussian Node
    """
    def __init__(self, mean, covariance) -> None:
        self.mean = mean 
        self.covariance = covariance 

    def get_mean(self):
        """
            Return the location of the point
        """
        return self.mean

    def get_gaussian(self):
        """
            Return mean and covariance of the Gaussian node
        """
        return self.mean, self.covariance

    def get_samples(self, num_samples):
        """
            Get samples from the distribution.
        """
        return np.random.multivariate_normal(self.mean, 
                                             self.covariance, 
                                             num_samples)

    def get_confidence_ellipse(self, confidence=0.95, num_points=100) -> Polygon:
        """
            Get confidence interval ellipse for collision checking
        """
        # Chi-squared value for 95% confidence interval in 2D
        chi2_val = chi2.ppf(confidence, df=2)

        # Eigens
        eigvals, eigvecs = np.linalg.eigh(self.covariance)
        axes_lengths = np.sqrt(eigvals * chi2_val)

        # points 
        theta = np.linspace(0, 2*np.pi, num_points)
        circle = np.stack([np.cos(theta), np.sin(theta)], axis=1)

        # scale and rotate
        ellipse = circle @ np.diag(axes_lengths) @ eigvecs.T
        ellipse += self.mean
        
        return Polygon(ellipse)

    def is_collision(self, region, threshold=0.95):
        """
            Check collision between confidence ellipse and shapely geometry
        """
        gaussian_poly = self.get_confidence_ellipse(threshold)
        return gaussian_poly.intersects(region)
    
    def visualize(self, ax, threshold=.95, edgecolor="r"):
        """
            Visualize node on the graph
        """
        eigenvalues, eigenvectors = np.linalg.eigh(self.covariance)

        # Sort eigenvalues and eigenvectors by descending eigenvalue
        order = eigenvalues.argsort()[::-1]
        eigenvalues = eigenvalues[order]
        eigenvectors = eigenvectors[:, order]

        # The angle of the ellipse (in degrees)
        angle = np.degrees(np.arctan2(*eigenvectors[:, 0][::-1]))

        # The width and height of the ellipse (2*sqrt(chi2_value*eigenvalue))
        chi2_value = chi2.ppf(threshold, 2)  # 95% confidence interval for 2 degrees of freedom (chi-squared value)
        width, height = 2 * np.sqrt(chi2_value * eigenvalues)
        ellipse = Ellipse(xy=self.mean, width=width, height=height, angle=angle, 
                          edgecolor=edgecolor, fc='None', lw=2) 
        ax.add_patch(ellipse)
        return ellipse
    
    def visualize_gaussian(self, ax, threshold=.95, cmap="Blues"):
        """
            Visualize Gaussian using gradient coloring
        """
        mean, cov = self.get_gaussian()
        x_l = mean[0] - 10
        x_h = mean[0] + 10
        y_l = mean[1] - 10
        y_h = mean[1] + 10
        x = np.linspace(x_l, x_h, 500)
        y = np.linspace(y_l, y_h, 500)
        X, Y = np.meshgrid(x, y)
        pos = np.dstack((X, Y))

        rv = multivariate_normal(mean, cov) 
        Z = rv.pdf(pos)
        chi2_val = chi2.ppf(threshold, 2)
        threshold = rv.pdf(mean) * np.exp(-0.5*chi2_val)
        Z_masked = np.ma.masked_where(Z < threshold, Z)
        ax.imshow(Z_masked, extent=[x_l, x_h, y_l, y_h], origin="lower", cmap=cmap, alpha=0.8)

class GaussianGraphNode(GaussianNode):

    """
        Gaussian Node
    """
    def __init__(self, mean, covariance, type="RANDOM", radius=0., alpha=.99, ) -> None:

        self.type = type
        if self.type == "UNIFORM":
            super().__init__(mean, np.identity(2))
            assert radius > 0, "Radius is required for Uniform Gaussian initialization"
            self.radius = radius
            self.alpha = alpha
            self.set_covariance()

        elif self.type == "RANDOM":
            super().__init__(mean, covariance)
            self.alpha = alpha
            self.radius = radius

    def get_capacity(self, agent_radius, threshold=.3):
        """
            Compute capacity based on safe area and agent radius.
        """
        if self.type == "UNIFORM":
            return floor((self.radius / agent_radius)**2)

        elif self.type == "RANDOM":
            eigenvalues, eigenvectors = np.linalg.eigh(self.covariance)
            
            # Sort eigenvalues and eigenvectors by descending eigenvalue
            order = eigenvalues.argsort()[::-1]
            eigenvalues = eigenvalues[order]
            eigenvectors = eigenvectors[:, order]
            chi2_value = chi2.ppf(self.alpha, 2) 
            width, height = np.sqrt(chi2_value * eigenvalues)

            # apply a small threshold to guarantee feasibility
            return floor(width * height / (agent_radius * agent_radius)*threshold)

    def get_mean(self):
        """
            Return the location of the point
        """
        return self.mean

    def get_radius(self):
        """
            Return the safe radius of the Gaussian Node
        """
        return self.radius

    def get_gaussian(self):
        return self.mean, self.covariance

    def set_alpha(self, alpha):
        self.alpha = alpha 

    def set_covariance(self):
        """
            Update covariance of the Gaussian node such that boundary matches
            the CVaR value. Use numerical method for solving covariance.
        """
        # compute covariance 
        dist_square = chi2.ppf(self.alpha, 2)
        self.covariance = self.radius ** 2 / dist_square * np.identity(2)

class GaussianMixture:
    """
        Gaussian Mixture that represents the distribution of groups of 
        agents on the map. Each GMM is represented with mean, covariance and 
        weight coefficient, such that the weight coefficients sum up to 1.
    """
    def __init__(self, means=[], covs=[], weights=[]) -> None:
        self.means = means 
        self.covs = covs
        self.weights = weights
        self.count = len(self.means)
    
    def get_gaussians(self):
        return zip(self.means, self.covs)
    
    def __len__(self):
        return len(self.means)
    
    @staticmethod
    def compute_distance(gmm_1, gmm_2):
        """
            Compute the Alternative Wasserstein Distance for two GMMs
            
            Reference:
            Y. Chen, T. T. Georgiou, and A. Tannenbaum, “Optimal transport for
            gaussian mixture models,” IEEE Access, vol. 7, pp. 6269–6278, 2018
        """
        r"""
        $$
            D(\rho1, \rho2) = 
            \left\{\min_{\pi \in \Pi(\omega_1, \omega_2)}\sum_{i=1}^{N_1}\sum_{j=1}^{N_2}\left[W_2(g_1^i, g_2^j)\right]^2\pi(i, j)\right\}
        $$
        """
        
        # compute pairwise Wasserstein Distance
        w_distance = np.zeros((len(gmm_1), len(gmm_2)))

        for i, (mean1, cov1) in enumerate(gmm_1.get_gaussians()):
            for j, (mean2, cov2) in enumerate(gmm_2.get_gaussians()):
                w_distance[i][j] = gaussian_wasserstein_distance(mean1, cov1, mean2, cov2)

class GaussianMixtureState:
    """
        Gaussian Mixture State
    """
    def __init__(self, gmm:GaussianMixture, timestep):
        self.gmm = gmm
        self.timestep = timestep

class GaussianMixtureInstance:
    """
        Gaussian instance for the macro problem, representing the start and goals
        of the problem as gaussian mixtures.
    """
    def __init__(self, start:GaussianMixtureState, goal:GaussianMixtureState):
        self.starts = start
        self.goal = goal

