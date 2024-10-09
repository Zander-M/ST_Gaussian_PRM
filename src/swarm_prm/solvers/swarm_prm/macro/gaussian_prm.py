"""
    Gaussian PRM based on map info.
"""
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Circle
import numpy as np
from scipy.spatial import KDTree, Delaunay, Voronoi
from scipy.stats.qmc import Halton
from shapely.geometry import Polygon

from swarm_prm.solvers.swarm_prm.macro.gaussian_utils import *
from swarm_prm.envs.instance import Instance

class GaussianPRM:
    """
        Gaussian PRM
    """

    def __init__(self, instance:Instance, num_samples, 
                 alpha=0.95, cvar_threshold=-8,
                 mc_threshold=0.02,
                 safety_radius=2,
                 swarm_prm_covariance_scaling=5,
                 cvt_iteration=10
                 ) -> None:

        # PARAMETERS
        self.instance = instance
        self.map = self.instance.map
        self.starts = self.instance.starts
        self.goals = self.instance.goals
        self.starts_weight = self.instance.starts_weight
        self.goals_weight = self.instance.goals_weight

        self.num_samples = num_samples
        self.alpha = alpha
        self.cvar_threshold = cvar_threshold
        self.safety_radius = safety_radius

        # Monte Carlo Sampling strategy
        self.mc_threshold = mc_threshold

        # SwarmPRM Sampling strategy
        self.swarm_prm_covariance_scalling= swarm_prm_covariance_scaling

        # CVT Map construction strategy
        self.cvt_iteration = cvt_iteration

        # Map related 
        self.samples = []
        self.gaussian_nodes = []
        self.roadmap = []

        self.shortest_paths = []
        self.starts_idx = []
        self.goals_idx = []

    def load_instance(self):
        """
            Load problem instance, adding start and target GMM nodes to the roadmap
        """

        self.starts_idx = [i for i in range(len(self.samples), len(self.samples) + len(self.starts))]
        self.samples.extend([start.get_mean() for start in self.starts])
        self.gaussian_nodes.extend(self.starts)

        self.goals_idx = [i for i in range(len(self.samples), len(self.samples) + len(self.goals))]
        self.samples.extend([goal.get_mean() for goal in self.goals])
        self.gaussian_nodes.extend(self.goals)

    def roadmap_construction(self):
        """
            Build Gaussian PRM
        """

        self.sample_free_space() # sample node locations 
        self.load_instance() # adding problem instance nodes to roadmap
        self.build_roadmap(roadmap_method="TRIANGULATION") # connect sample Gaussian nodes, building roadmap

    def cvt_roadmap_construction(self):
        """
            Iteratively update roadmap using CVT optimization
        """
        def cvt_compute_centroids(vor, points, bounding_polygon, obstacles):
            """
                Compute centroids of Voronoi regions
            """
            new_points = []
            for point, region_index in zip(points, vor.point_region):
                region = vor.regions[region_index]
                if not region or -1 in region:
                    new_points.append(point)  # Keep original point for infinite regions
                    continue
        
                # Create a polygon for the Voronoi cell
                region_vertices = [vor.vertices[i] for i in region]
                cell_polygon = Polygon(region_vertices)

                # If the region intersects with any obstacle, keep the original point
                if any(cell_polygon.intersects(obs) for obs in obstacles):
                    new_points.append(point)
                else:
                    # Calculate the centroid of the cell
                    if cell_polygon.is_valid and not cell_polygon.is_empty:
                        centroid = cell_polygon.centroid
                        # Ensure the centroid is inside the bounding polygon
                        if bounding_polygon.contains(centroid):
                            new_points.append(centroid.coords[0])
                        else:
                            # If centroid is outside, keep the original point
                            # Alternatively, you could project it back onto the bounding polygon boundary
                            closest_point = bounding_polygon.exterior.interpolate(bounding_polygon.exterior.project(centroid))
                            new_points.append(closest_point.coords[0])
                    else:
                        new_points.append(point)  # Fallback to original point if invalid

                return np.array(new_points)
    
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
                    g_node = GaussianGraphNode(node, None, type="UNIFORM", radius=radius)
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
                    g_node = GaussianGraphNode(node, None, type="UNIFORM", radius=radius)
                    self.samples.append(node)               
                    self.gaussian_nodes.append(g_node)

        elif sampling_strategy == "UNIFORM_HALTON":
            min_x, max_x, min_y, max_y = 0, self.map.width, 0 , self.map.height
            halton_sampler = Halton(d=2) # 2d Halton sampler

            while len(self.samples) < self.num_samples:
                num_to_generate = max(self.num_samples * 2, 100)
                samples = halton_sampler.random(num_to_generate)
                nodes = samples * np.array([self.map.width, self.map.height])

                # covariance will be automatically updated.
                for node in nodes:
                    if not self.map.is_point_collision(node):
                        radius = np.inf
                        for obs in self.map.obstacles:
                            radius = min(radius, obs.get_dist(node))
                        g_node = GaussianGraphNode(node, None, type="UNIFORM", radius=radius)
                        self.samples.append(node)               
                        self.gaussian_nodes.append(g_node)
                        if len(self.samples) >= self.num_samples:
                            break

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
        self.new_node_idx = len(self.samples)

    def build_roadmap(self, radius=10, roadmap_method="KDTREE", collision_check_method="CVAR"):
        """
            Build Roadmap based on samples. Default connect radius is 10
        """

        # add start nodes and goal nodes to the graph

        if roadmap_method == "KDTREE":
            kd_tree = KDTree([(sample[0], sample[1]) for sample in self.samples])
            for i, node in enumerate(self.samples):
                indices = kd_tree.query_ball_point(node, radius, 2)

                # Edge must be collision free with the environment
                edges = [(i, idx) for idx in indices \
                         if not self.map.is_gaussian_trajectory_collision(
                             self.gaussian_nodes[i],
                             self.gaussian_nodes[idx],
                             collision_check_method=collision_check_method)]
                self.roadmap.extend(edges)

        elif roadmap_method == "TRIANGULATION":
            tri = Delaunay(self.samples)
            for i, simplex in enumerate(tri.simplices):
                for i in range(-1, 2):
                    if (simplex[i], simplex[i+1]) not in self.roadmap \
                        and (simplex[i+1], simplex[i]) not in self.roadmap \
                        and np.linalg.norm(self.samples[simplex[i]]-self.samples[simplex[i+1]]) < radius \
                        and not self.map.is_gaussian_trajectory_collision(
                             self.gaussian_nodes[simplex[i]],
                             self.gaussian_nodes[simplex[i+1]],
                             collision_check_method=collision_check_method):
                        self.roadmap.append((simplex[i], simplex[i+1]))

        elif roadmap_method == "VORONOI":
            """
                Construct roadmap by creating voronoi cells based on sampled points
            """
            pass

    def visualize_roadmap(self, fname="test_gaussian_prm"):
        """
            Visualize Gaussian PRM
        """
        fig, ax = plt.subplots(figsize=(10, 10))

        # Nodes and Paths 

        for node in self.samples:
            ax.plot(node[0], node[1], 'co', markersize=2)
        for (i, j) in self.roadmap:
            ax.plot([self.samples[i][0], self.samples[j][0]], [self.samples[i][1], self.samples[j][1]], 'gray', linestyle='-', linewidth=0.5)

        # Starts and goals
        for start in self.instance.starts:
            pos = start.get_mean()
            ax.plot(pos[0], pos[1], 'bo', markersize=3)
            start.visualize(ax=ax, edgecolor="blue")

        for goal in self.instance.goals:
            pos = goal.get_mean()
            ax.plot(pos[0], pos[1], 'go', markersize=3)
            goal.visualize(ax=ax, edgecolor="green")

        for obs in self.map.obstacles:
            if obs.obs_type == "CIRCLE": 
                x, y = obs.get_pos()
                # ax.plot(x, y, 'ro', markersize=3)
                ax.add_patch(Circle((x, y), radius=obs.radius, color="black"))
            elif obs.obs_type == "POLYGON":
                x, y = obs.geom.exterior.xy
                ax.fill(x, y, fc="black")

        ax.set_aspect('equal')
        ax.set_xlim(left=0, right=self.map.width)
        ax.set_ylim(bottom=0, top=self.map.height)
        plt.savefig("{}.png".format(fname))
        return fig, ax
    
    def visualize_g_nodes(self, fname="test_g_nodes"):
        """
            Visualize Gaussian Nodes on the map
        """
        fig, ax = plt.subplots(figsize=(10, 10))

        # Visualize G nodes
        cmap = plt.get_cmap('tab10')
        for i, gaussian_node in enumerate(self.gaussian_nodes):
            gaussian_node.visualize(ax, edgecolor=cmap(i%10))

        for obs in self.map.obstacles:
            if obs.obs_type == "CIRCLE": 
                x, y = obs.get_pos()
                # ax.plot(x, y, 'ro', markersize=3)
                ax.add_patch(Circle((x, y), radius=obs.radius, color="black"))
            elif obs.obs_type == "POLYGON":
                x, y = obs.geom.exterior.xy
                ax.fill(x, y, fc="black")

        ax.set_aspect('equal')
        ax.set_xlim(left=0, right=self.map.width)
        ax.set_ylim(bottom=0, top=self.map.height)
        plt.savefig("{}.png".format(fname), dpi=400)
        plt.show()

    def get_solution(self, flow_dict, timestep, num_agent):
        """
            Return macro solution path per agent
        """
        paths = []
        for _ in range(num_agent):
            paths.append([])
            u = "SS"
            while u != "SG":
                for v in flow_dict[u]:
                    if flow_dict[u][v] > 0:
                        if v == "SG":
                            flow_dict[u][v] -= 1
                            u = v
                            break
                        else:
                            idx = int(v.split("_")[0])
                            paths[-1].append(idx)
                            flow_dict[u][v] -= 1
                            u = v
                            break

            # padding paths to solution length, agent waits at goal
            wait_timestep = timestep - len(paths[-1])
            paths[-1] = paths[-1] + [paths[-1][-1]] * wait_timestep


        simple_paths = [] # positions
        gaussian_paths = [] # Gaussian nodes
        for path in paths:
            simple_paths.append([self.samples[i] for i in path])
            gaussian_paths.append([self.gaussian_nodes[i] for i in path])
            

        return simple_paths, gaussian_paths

    def visualize_solution(self, flow_dict, timestep, num_agent):
        """
            Visualize solution path per timestep provided the flow dict
        """
        node_idx = [i for i in range(len(self.samples))]
        for t in range(timestep):
            fig, ax = self.visualize_roadmap()

            # plot path at each timestep
            for i in node_idx:
                u = '{}_{}'.format(i, t)
                for j in node_idx:
                    v = '{}_{}'.format(j, t+1)
                    if v in flow_dict[u] and flow_dict[u][v] != 0:
                        ax.plot([self.samples[i][0], self.samples[j][0]], 
                                [self.samples[i][1], self.samples[j][1]], 
                                'r', linestyle='-', linewidth=flow_dict[u][v]/num_agent * 20)

    def animate_solution(self, flow_dict, timestep, num_agent):
        """
            Animate solution paths
        """
        node_idx = [i for i in range(len(self.samples))]

        def get_segments(timestep):
            data = []
            for t in range(timestep):
                segments = []
                for i in node_idx:
                    u = '{}_{}'.format(i, t)
                    for j in node_idx:
                        v = '{}_{}'.format(j, t+1)
                        if v in flow_dict[u] and flow_dict[u][v] != 0:
                            # [x, y, capacity]
                            segments.append([
                                [self.samples[i][0], self.samples[j][0]], 
                                [self.samples[i][1], self.samples[j][1]],
                                flow_dict[u][v]
                                ])
                data.append(segments)
            return data

        # Plotting map
        data = get_segments(timestep)
        fig, ax = self.visualize_roadmap()

        # Animate functions
        lines = []

        def init():
            for line in lines:
                line.remove()
            lines.clear()
            return []
        
        def update(frame):
            for line in lines:
                line.remove()
            lines.clear()

            segments = data[frame]
            for x, y, capacity in segments:
                line, = ax.plot(x, y, color='r', lw=capacity/num_agent*20)
                lines.append(line)
            return lines
        anim = FuncAnimation(fig, update, frames=timestep, init_func=init,
                             blit=True, interval=100)
        anim.save("test_solution_path.gif", writer='pillow', fps=6)


