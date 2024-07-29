"""
    Gaussian PRM based on map info.
"""
import ortools.graph.python

from matplotlib import pyplot as plt
import numpy as np
from scipy.optimize import fsolve
from scipy.stats import gaussian_kde
from scipy.spatial import KDTree

from envs.map_objects import Map
from solvers.swarm_prm.macro.gaussian_mixture_model import GaussianMixtureModel

   
class GaussianNode:
    """
        Gaussian Node
    """
    def __init__(self, pos, radius, cvar= 0.99, type="UNIFORM") -> None:
        self.type = type
        self.radius = radius
        self.cvar = cvar
        self.mean = pos 
        self.covariance = np.identity(2)
        self.set_covariance()
        print(self.radius, self.covariance)
        


    def get_capacity(self):
        """
            Compute capacity based on safe area.
        """
        return np.pi*self.radius*self.radius

    def get_pos(self):
        """
            Return the location of the point
        """
        return self.pos

    def get_gaussian(self):
        return self.mean, self.covariance

    def set_cvar(self, cvar):
        self.cvar = cvar

    def set_covariance(self):
        """
            Update covariance of the Gaussian node such that boundary matches
            the CVaR value. Use numerical method for solving covariance.
        """
        # Given values
        t = self.radius  # distance
        p = self.cvar # probability density function value at distance t

        # Function to solve for sigma^2
        def equation(sigma2):
            return p * 2 * np.pi * sigma2 - np.exp(-t**2 / (2 * sigma2))

        # Initial guess for sigma^2
        initial_guess = t**2 / 2

        # Solve for sigma^2
        sigma2_solution, = fsolve(equation, initial_guess)

        # Covariance matrix
        self.covariance = sigma2_solution

class GaussianPRM:
    """
        Gaussian PRM
    """

    def __init__(self, map:Map, num_samples) -> None:
        self.map = map
        self.num_samples = num_samples
        self.samples = []
        self.gaussian_nodes = []
        self.roadmap = []

    def roadmap_construction(self):
        """
            Build Gaussian PRM
        """
        self.sample_free_space() # sample node locations 
        self.build_roadmap() # connect sample nodes, building roadmap
        self.expand_nodes() # expand nodes to gaussian density functions

    def expand_nodes(self):
        """
            Expand node locations to gaussian density functions
        """
        for sample in self.samples:
            # determine max distance
            radius = np.inf
            for obs in self.map.obstacles:
                radius = min(radius, obs.get_dist(sample))
                
            self.gaussian_nodes.append(GaussianNode(sample, radius))

    def sample_free_space(self):
        """
            Sample points on the map uniformly random
            TODO: add Gaussian Sampling perhaps?
        """
        min_x, max_x, min_y, max_y = 0, self.map.width, 0 , self.map.height
        while len(self.samples) < self.num_samples:
            x = np.random.uniform(min_x, max_x)
            y = np.random.uniform(min_y, max_y)
            node = np.array((x, y))
            if not self.map.is_point_collision(node):
                self.samples.append(node)

    def build_roadmap(self, k=10):
        """
            Build Roadmap based on samples
        """

        assert k < len(self.samples),"not enough samples for k = {}".format(k)

        kd_tree = KDTree([(sample[0], sample[1]) for sample in self.samples])
        for i, node in enumerate(self.samples):
            distances, indices = kd_tree.query((node[0], node[1]), k=k+1)

            # Edge must be collision free with the environment
            edges = [(i, idx) for idx, dist in zip(indices[1:], distances[1:]) if dist > 0 \
                     and not self.map.is_line_collision(self.samples[i], self.samples[idx])]
            self.roadmap.extend(edges)

    def get_abstract_prm(self):
        """
            Return abstract PRM
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

    def visualize(self, fname):
        """
            Visualize Gaussian PRM
        """
        fig, ax = plt.subplots()
        for (i, j) in self.roadmap:
            ax.plot([self.samples[i][0], self.samples[j][0]], [self.samples[i][1], self.samples[j][1]], 'gray', linestyle='-', linewidth=0.5)

        for node in self.samples:
            ax.plot(node[0], node[1], 'bo', markersize=2)

        for obs in self.map.obstacles:
            ox, oy = obs.get_pos()
            ax.add_patch(plt.Circle((ox, oy), radius=obs.radius, color="gray"))

        # if path:
            # path_x = [node.x for node in path]
            # path_y = [node.y for node in path]
            # ax.plot(path_x, path_y, 'g-', linewidth=2)

        ax.set_aspect('equal')
        plt.savefig("{}.png".format(fname), dpi=400)

if __name__ == "__main__":
    pass