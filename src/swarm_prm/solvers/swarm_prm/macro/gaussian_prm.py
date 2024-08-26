"""
    Gaussian PRM based on map info.
"""
from matplotlib import pyplot as plt
from matplotlib.patches import Ellipse
import numpy as np

from scipy.spatial import KDTree, Delaunay
from scipy.stats import chi2

from swarm_prm.envs.roadmap import Roadmap
from swarm_prm.solvers.swarm_prm.macro.gaussian_mixture import GaussianNode, GaussianMixture

class GaussianGraphNode(GaussianNode):

    """
        Gaussian Node
    """

    def __init__(self, mean, covariance, type="UNIFORM", alpha=.99, radius=None) -> None:

        if type == "UNIFORM":
            super().__init__(mean, np.identity(2))
            assert radius is not None, "Radius is required for Uniform Gaussian initialization"
            self.radius = radius
            self.alpha = alpha
            self.set_covariance()

        elif type == "RANDOM":
            super().__init__(mean, covariance)
            self.alpha = alpha
            self.radius = radius

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
        """
        # compute covariance 
        dist_square = chi2.ppf(self.alpha, 2)
        self.covariance = self.radius ** 2 / dist_square * np.identity(2)

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

class GaussianPRM:
    """
        Gaussian PRM
    """

    def __init__(self, map:Roadmap, instance:GaussianMixtureInstance, num_samples, 
                 alpha=0.95, cvar_threshold=-8,
                 mc_threshold=0.02,
                 safety_radius=2,
                 swarm_prm_covariance_scaling=5,
                 ) -> None:

        # PARAMETERS
        self.map = map
        self.instance = instance
        self.num_samples = num_samples
        self.alpha = alpha
        self.cvar_threshold = cvar_threshold
        self.safety_radius = safety_radius

        # Monte Carlo Sampling strategy
        self.mc_threshold = mc_threshold

        # SwarmPRM Sampling strategy
        self.swarm_prm_covariance_scalling= swarm_prm_covariance_scaling

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
        self.load_instance() # adding problem instance nodes to roadmap
        self.build_roadmap(roadmap_method="TRIANGULATION") # connect sample Gaussian nodes, building roadmap

    def sample_free_space(self, sampling_strategy="UNIFORM", collision_check_method="CVAR"):
        """
            Sample points on the map uniformly random
            TODO: add Gaussian Sampling perhaps?
        """

        if sampling_strategy == "UNIFORM":
            min_x, max_x, min_y, max_y = 0, self.map.width, 0 , self.map.height
            while len(self.samples) < self.num_samples:
                x = np.random.uniform(min_x, max_x)
                y = np.random.uniform(min_y, max_y)
                node = np.array((x, y))
                if not self.map.is_point_collision(node):
                    radius = np.inf
                    for obs in self.map.obstacles:
                        radius = min(radius, obs.get_dist(node))
                    g_node = GaussianGraphNode(node, None, radius=radius)
                    self.samples.append(node)
                    self.gaussian_nodes.append(g_node)

        elif sampling_strategy == "UNIFORM_WITH_RADIUS":
            min_x, max_x, min_y, max_y = 0, self.map.width, 0 , self.map.height
            while len(self.samples) < self.num_samples:
                x = np.random.uniform(min_x, max_x)
                y = np.random.uniform(min_y, max_y)

                # covariance will be automatically updated.
                node = np.array((x, y))
                if not self.map.is_radius_collision(node, self.safety_radius):
                    radius = np.inf
                    for obs in self.map.obstacles:
                        radius = min(radius, obs.get_dist(node))
                    g_node = GaussianGraphNode(node, None, radius=radius)
                    self.samples.append(node)               
                    self.gaussian_nodes.append(g_node)

        elif sampling_strategy == "SWARMPRM":
            """
                Random sampling strategy as shown in SwarmPRM. Sample location 
                and covariance and check if the sample satisfies the CVaR condition
            """

            min_x, max_x, min_y, max_y = 0, self.map.width, 0 , self.map.height
            while len(self.samples) < self.num_samples:
                x = np.random.uniform(min_x, max_x)
                y = np.random.uniform(min_y, max_y)
                node = np.array((x, y))
                if self.map.is_radius_collision(node, self.safety_radius):
                    continue
                
                mean = np.array((x, y))
                a = np.random.uniform(low=-self.swarm_prm_covariance_scalling, 
                                      high = self.swarm_prm_covariance_scalling,
                                      size=[2, 2])
                cov = a.T@ a # construct a random positive semidefinite cov matrix

                g_node = GaussianGraphNode(mean, cov, type="RANDOM")
                if not self.map.is_gaussian_collision(g_node,
                                                      collision_check_method=collision_check_method,
                                                      alpha=self.alpha, cvar_threshold=self.cvar_threshold
                                                      ):
                    self.samples.append(mean)
                    self.gaussian_nodes.append(g_node)

        elif sampling_strategy == "GAUSSIAN":
            assert False, "Unimplemented Gaussian sampling strategy"
        else:
            assert False, "Unimplemented sampling strategy"

    def build_roadmap(self, kd_radius=10, roadmap_method="KDTREE", collision_check_method="CVAR"):
        """
            Build Roadmap based on samples. Default connect radius is 10
        """

        if roadmap_method == "KDTREE":
            kd_tree = KDTree([(sample[0], sample[1]) for sample in self.samples])
            for i, node in enumerate(self.samples):
                indices = kd_tree.query_ball_point(node, r=kd_radius)

                # Edge must be collision free with the environment
                edges = [(i, idx) for idx in indices \
                         if not self.map.is_gaussian_trajectory_collision(
                             self.gaussian_nodes[i],
                             self.gaussian_nodes[idx],
                             collision_check_method=collision_check_method)]
                self.roadmap.extend(edges)

        elif roadmap_method == "TRIANGULATION":
            tri = Delaunay([(sample[0], sample[1]) for sample in self.samples])
            for i, simplex in enumerate(tri.simplices):
                for i in range(-1, 2):

                    if (simplex[i], simplex[i+1]) not in self.roadmap \
                        and (simplex[i+1], simplex[i]) not in self.roadmap \
                        and not self.map.is_gaussian_trajectory_collision(
                             self.gaussian_nodes[simplex[i]],
                             self.gaussian_nodes[simplex[i+1]],
                             collision_check_method=collision_check_method):
                        self.roadmap.append((simplex[i], simplex[i+1]))

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

    def visualize_roadmap(self, fname):
        """
            Visualize Gaussian PRM
        """
        fig, ax = plt.subplots(figsize=(10, 10))

        # Nodes and Paths 

        for node in self.samples:
            ax.plot(node[0], node[1], 'bo', markersize=2)

        for (i, j) in self.roadmap:
            ax.plot([self.samples[i][0], self.samples[j][0]], [self.samples[i][1], self.samples[j][1]], 'gray', linestyle='-', linewidth=0.5)

        for obs in self.map.obstacles:
            ox, oy = obs.get_pos()
            ax.add_patch(plt.Circle((ox, oy), radius=obs.radius, color="black"))

        ax.set_aspect('equal')
        ax.set_xlim(left=0, right=self.map.width)
        ax.set_ylim(bottom=0, top=self.map.height)
        plt.savefig("{}.png".format(fname), dpi=400)
        plt.show()
    
    def visualize_g_nodes(self, fname):
        """
            Visualize Gaussian Nodes on the map
        """
        fig, ax = plt.subplots(figsize=(10, 10))

        # for (i, j) in self.roadmap:
            # ax.plot([self.samples[i][0], self.samples[j][0]], [self.samples[i][1], self.samples[j][1]], 'gray', linestyle='-', linewidth=0.5)

        # for node in self.samples:
            # ax.plot(node[0], node[1], 'bo', markersize=2)

        # Visualize G nodes
        cmap = plt.get_cmap('tab10')
        for i, gaussian_node in enumerate(self.gaussian_nodes):
            mean, cov = gaussian_node.get_gaussian()
            # Compute the eigenvalues and eigenvectors of the covariance matrix
            eigenvalues, eigenvectors = np.linalg.eigh(cov)

            # Sort eigenvalues and eigenvectors by descending eigenvalue
            order = eigenvalues.argsort()[::-1]
            eigenvalues = eigenvalues[order]
            eigenvectors = eigenvectors[:, order]

            # The angle of the ellipse (in degrees)
            angle = np.degrees(np.arctan2(*eigenvectors[:, 0][::-1]))

            # The width and height of the ellipse (2*sqrt(chi2_value*eigenvalue))
            chi2_value = chi2.ppf(0.99, 2)  # 95% confidence interval for 2 degrees of freedom (chi-squared value)
            width, height = 2 * np.sqrt(chi2_value * eigenvalues)
            ellipse = Ellipse(xy=mean, width=width, height=height, angle=angle, 
                              edgecolor=cmap(i%10), fc='None', lw=2)
            ax.add_patch(ellipse)

        for obs in self.map.obstacles:
            ox, oy = obs.get_pos()
            ax.add_patch(plt.Circle((ox, oy), radius=obs.radius, color="black"))

        # if path:
            # path_x = [node.x for node in path]
            # path_y = [node.y for node in path]
            # ax.plot(path_x, path_y, 'g-', linewidth=2)

        ax.set_aspect('equal')
        ax.set_xlim(left=0, right=self.map.width)
        ax.set_ylim(bottom=0, top=self.map.height)
        plt.savefig("{}.png".format(fname), dpi=400)
        plt.show()

    def visualize_solution(self):
        """
            Visualize solution paths
        """
        pass

if __name__ == "__main__":
    pass