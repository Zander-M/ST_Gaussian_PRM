"""
    Gaussian PRM based on map info.
"""
from matplotlib import pyplot as plt
import numpy as np
from scipy.optimize import fsolve
from scipy.spatial import KDTree

from swarm_prm.envs.map_objects import Map
from swarm_prm.solvers.swarm_prm.macro.gaussian_mixture import GaussianNode, GaussianMixture

class GaussianGraphNode(GaussianNode):

    """
        Gaussian Node
    """

    def __init__(self, pos, radius, alpha=.99) -> None:

        super().__init__(pos, np.identity(2))
        self.radius = radius
        self.alpha = alpha
        self.set_covariance()

    def get_capacity(self):
        """
            Compute capacity based on safe area.
        """
        return np.pi*self.radius*self.radius

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

            TODO: do we need this?
        """
        # Given values
        t = self.radius  # distance
        p = self.alpha# probability density function value at distance t

        # Function to solve for sigma^2
        def equation(sigma2):
            return p * 2 * np.pi * sigma2 - np.exp(-t**2 / (2 * sigma2))

        # Initial guess for sigma^2
        initial_guess = t**2 / 2

        # Solve for sigma^2
        sigma2_solution, = fsolve(equation, initial_guess)

        # Covariance matrix
        self.covariance = sigma2_solution

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
        self.start = start
        self.goal = goal

class GaussianPRM:
    """
        Gaussian PRM
    """

    def __init__(self, map:Map, instance:GaussianMixtureInstance, num_samples, 
                 alpha=0.9, cvar_threshold=0.02,
                 mc_threshold=0.02,
                 sampling_strategy="UNIFORM", safety_radius=2) -> None:

        # PARAMETERS
        self.map = map
        self.instance = instance
        self.num_samples = num_samples
        self.alpha = alpha
        self.cvar_threshold = cvar_threshold
        self.sampling_strategy = sampling_strategy
        self.safety_radius = safety_radius

        self.samples = []
        self.gaussian_nodes = []
        self.roadmap = []
        self.shortest_paths = []


    def load_instance(self):
        """
            Load problem instance, adding start and target GMM nodes to the roadmap
            TODO: implement this
        """
        pass
        
    def roadmap_construction(self):
        """
            Build Gaussian PRM
        """
        self.sample_free_space() # sample node locations 
        if self.sampling_strategy != "SWARMPRM":
            # Swarm PRM sample Gaussian nodes so no need to covert to gaussian nodes
            self.expand_nodes() # expand nodes to gaussian density functions
        self.load_instance() # adding problem instance nodes to roadmap
        self.build_roadmap() # connect sample Gaussian nodes, building roadmap

    def expand_nodes(self):
        """
            Expand node locations to gaussian density functions
        """
        for sample in self.samples:
            # determine max distance
            radius = np.inf
            for obs in self.map.obstacles:
                radius = min(radius, obs.get_dist(sample))
                
            self.gaussian_nodes.append(GaussianGraphNode(sample, radius))

    def sample_free_space(self):
        """
            Sample points on the map uniformly random
            TODO: add SwarmPRM sampling strategy
            TODO: add Gaussian Sampling perhaps?
        """

        if self.sampling_strategy == "UNIFORM":
            min_x, max_x, min_y, max_y = 0, self.map.width, 0 , self.map.height
            while len(self.samples) < self.num_samples:
                x = np.random.uniform(min_x, max_x)
                y = np.random.uniform(min_y, max_y)
                node = np.array((x, y))
                if not self.map.is_point_collision(node):
                    self.samples.append(node)

        elif self.sampling_strategy == "UNIFORM_WITH_RADIUS":
            min_x, max_x, min_y, max_y = 0, self.map.width, 0 , self.map.height
            while len(self.samples) < self.num_samples:
                x = np.random.uniform(min_x, max_x)
                y = np.random.uniform(min_y, max_y)
                node = np.array((x, y))
                if not self.map.is_radius_collision(node, self.safety_radius):
                    self.samples.append(node)               

        elif self.sampling_strategy == "GAUSSIAN":
            assert False, "Unimplemented Gaussian sampling strategy"
            pass

        elif self.sampling_strategy == "SWARMPRM":
            """
                Random sampling strategy as shown in SwarmPRM. Sample location 
                and covariance and check if the sample satisfies the CVaR condition
            """
            assert False, "Unimplemented Random sampling strategy"

        elif self.sampling_strategy == "SPACE_TRIANGULATION":
            """
                Apply space triangulation and prune unsuitable points.
            """
            assert False, "Unimplemented space triangulation sampling strategy"
            pass
        else:
            assert False, "Unimplemented sampling strategy"

    def build_roadmap(self, r=10):
        """
            Build Roadmap based on samples. Default connect radius is 10
        """

        kd_tree = KDTree([(sample[0], sample[1]) for sample in self.samples])
        for i, node in enumerate(self.samples):
            indices = kd_tree.query_ball_point(node, r=r)

            # Edge must be collision free with the environment
            edges = [(i, idx) for idx in indices \
                     if not self.map.is_gaussian_collision(self.gaussian_nodes[i], self.gaussian_nodes[idx])]
            self.roadmap.extend(edges)

    def get_csgraph(self):
        """
            Convert Roadmap to scipy csgraph for graph search algorithms
            TODO: Implement this 
        """
        pass

    def astar_search(self, start, goal):
        """
            A star search on abstract graph
        """
        pass

    def gmm_search(self):
        """
            Use linear programming to find the weight assigned to different solution
            paths.
        """

    def visualize_map(self, fname):
        """
            Visualize Gaussian PRM
        """
        fig, ax = plt.subplots()
        for (i, j) in self.roadmap:
            ax.plot([self.samples[i][0], self.samples[j][0]], [self.samples[i][1], self.samples[j][1]], 'gray', linestyle='-', linewidth=0.5)

        # for node in self.samples:
            # ax.plot(node[0], node[1], 'bo', markersize=2)

        for gaussian_node in self.gaussian_nodes:
            pos = gaussian_node.get_mean()
            ax.plot(pos[0], pos[1], 'bo', markersize=2)
            # radius = gaussian_node.get_radius()
            # ax.add_patch(plt.Circle(pos, radius=radius, color="cyan"))

        for obs in self.map.obstacles:
            ox, oy = obs.get_pos()
            ax.add_patch(plt.Circle((ox, oy), radius=obs.radius, color="gray"))

        # if path:
            # path_x = [node.x for node in path]
            # path_y = [node.y for node in path]
            # ax.plot(path_x, path_y, 'g-', linewidth=2)

        ax.set_aspect('equal')
        ax.set_xlim(left=0, right=self.map.width)
        ax.set_ylim(bottom=0, top=self.map.height)
        plt.savefig("{}.png".format(fname), dpi=400)

    def visualize_solution(self):
        """
            Visualize solution paths
        """
        pass

if __name__ == "__main__":
    pass