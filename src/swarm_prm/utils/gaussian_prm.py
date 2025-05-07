"""
    Gaussian PRM based on map info.
"""
from collections import defaultdict 
import copy 

from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Circle
import numpy as np
from scipy.spatial import KDTree, Delaunay 
from scipy.stats.qmc import Halton
from shapely.geometry import Point, Polygon

from swarm_prm.utils.gaussian_utils import *
from swarm_prm.utils.cvt import CVT

class GaussianPRM:
    """
        Gaussian PRM
    """

    def __init__(self, instance, num_samples, 
                 alpha=0.95, cvar_threshold=-8,
                 mc_threshold=0.02,
                 safety_radius=2.0,
                 swarm_prm_covariance_scaling=5,
                 cvt_iteration=10,
                 hex_radius=2
                 ) -> None:

        # PARAMETERS
        self.instance = instance
        self.num_samples = num_samples
        self.raw_map = self.instance.roadmap
        self.starts = self.instance.starts
        self.goals = self.instance.goals

        self.alpha = alpha
        self.cvar_threshold = cvar_threshold
        self.safety_radius = safety_radius

        # Monte Carlo Sampling strategy
        self.mc_threshold = mc_threshold

        # SwarmPRM Sampling strategy
        self.swarm_prm_covariance_scalling= swarm_prm_covariance_scaling

        # CVT Map construction strategy
        self.cvt_iteration = cvt_iteration

        # Hexagon Map construction
        self.hex_radius = hex_radius

        # Map related 
        self.samples = []
        self.gaussian_nodes = []
        self.roadmap = []
        self.roadmap_cost = []

        self.shortest_paths = []

        # Add starts and goals to the map
        self.starts_idx = [i for i in range(len(self.samples), len(self.samples) + len(self.starts))]
        self.samples.extend([start.get_mean() for start in self.starts])
        self.gaussian_nodes.extend(self.starts)

        self.goals_idx = [i for i in range(len(self.samples), len(self.samples) + len(self.goals))]
        self.samples.extend([goal.get_mean() for goal in self.goals])
        self.gaussian_nodes.extend(self.goals)

    def add_sample(self, sample):
        """
            Explicitly add samples to road map. For Debugging only
        """
        self.samples.append(sample)
    
    def add_gaussian_node(self, mean, cov):
        """
            Explicitly add Gaussian Graph nodes to road map. For Debugging only
        """
        self.add_sample(mean)
        g_node = GaussianGraphNode(mean, cov)
        self.gaussian_nodes.append(g_node)

    def add_edge(self, idx1, idx2):
        """
            Explicitly add edge between indicies. For Debugging only
        """
        self.roadmap.append((idx1, idx2))

    def roadmap_construction(self):
        """
            Build Gaussian PRM
        """
        self.sample_free_space() # sample node locations 
        self.build_roadmap(roadmap_method="TRIANGULATION") # connect sample Gaussian nodes, building roadmap

    def sample_free_space(self, sampling_strategy="CVT", collision_check_method="CVAR"):
        """
            Sample points on the map uniformly random
            TODO: add Gaussian Sampling perhaps?
        """

        if sampling_strategy == "UNIFORM":
            min_x, max_x, min_y, max_y = 0, self.raw_map.width, 0 , self.raw_map.height
            while len(self.samples) < self.num_samples:
                x = np.random.uniform(min_x, max_x)
                y = np.random.uniform(min_y, max_y)
                node = np.array((x, y))
                if not self.raw_map.is_point_collision(node):
                    radius = np.inf
                    for obs in self.raw_map.obstacles:
                        radius = min(radius, obs.get_dist(node))
                    g_node = GaussianGraphNode(node, None, type="UNIFORM", radius=radius)
                    self.samples.append(node)
                    self.gaussian_nodes.append(g_node)

        elif sampling_strategy == "CVT":
            """
                Perform Centroidal Voronoi Tesellation for choosing 
                Gaussian Points
            """
            cvt_instance = CVT(self.raw_map, self.num_samples, iteration=500)
            samples, gaussian_nodes = cvt_instance.get_CVT()
            self.samples.extend(samples)
            self.gaussian_nodes.extend(gaussian_nodes)

        elif sampling_strategy == "UNIFORM_WITH_RADIUS":
            min_x, max_x, min_y, max_y = 0, self.raw_map.width, 0 , self.raw_map.height
            while len(self.samples) < self.num_samples:
                x = np.random.uniform(min_x, max_x)
                y = np.random.uniform(min_y, max_y)

                # covariance will be automatically updated.
                node = np.array((x, y))
                if not self.raw_map.is_radius_collision(node, self.safety_radius):
                    radius = np.inf
                    for obs in self.raw_map.obstacles:
                        radius = min(radius, obs.get_dist(node))
                    g_node = GaussianGraphNode(node, None, type="UNIFORM", radius=radius)
                    self.samples.append(node)               
                    self.gaussian_nodes.append(g_node)

        elif sampling_strategy == "UNIFORM_HALTON":
            min_x, max_x, min_y, max_y = 0, self.raw_map.width, 0 , self.raw_map.height
            halton_sampler = Halton(d=2) # 2d Halton sampler

            while len(self.samples) < self.num_samples:
                num_to_generate = max(self.num_samples * 2, 100)
                samples = halton_sampler.random(num_to_generate)
                nodes = samples * np.array([self.raw_map.width, self.raw_map.height])

                # covariance will be automatically updated.
                for node in nodes:
                    if not self.raw_map.is_point_collision(node):
                        radius = np.inf
                        for obs in self.raw_map.obstacles:
                            radius = min(radius, obs.get_dist(node))
                        g_node = GaussianGraphNode(node, None, type="UNIFORM", radius=radius)
                        self.samples.append(node)               
                        self.gaussian_nodes.append(g_node)
                        if len(self.samples) >= self.num_samples:
                            break

        elif sampling_strategy == "UNIFORM_HALTON_RADIUS":
            min_x, max_x, min_y, max_y = 0, self.raw_map.width, 0 , self.raw_map.height
            halton_sampler = Halton(d=2) # 2d Halton sampler

            while len(self.samples) < self.num_samples:
                num_to_generate = max(self.num_samples * 2, 100)
                samples = halton_sampler.random(num_to_generate)
                nodes = samples * np.array([self.raw_map.width, self.raw_map.height])

                # covariance will be automatically updated.
                for node in nodes:
                    if not self.raw_map.is_radius_collision(node, self.safety_radius):
                        radius = np.inf
                        for obs in self.raw_map.obstacles:
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

            min_x, max_x, min_y, max_y = 0, self.raw_map.width, 0 , self.raw_map.height
            while len(self.samples) < self.num_samples:
                x = np.random.uniform(min_x, max_x)
                y = np.random.uniform(min_y, max_y)
                node = np.array((x, y))
                if self.raw_map.is_radius_collision(node, self.safety_radius):
                    continue
                
                mean = np.array((x, y))
                a = np.random.uniform(low=-self.swarm_prm_covariance_scalling, 
                                      high = self.swarm_prm_covariance_scalling,
                                      size=[2, 2])
                cov = a.T@ a # construct a random positive semidefinite cov matrix

                g_node = GaussianGraphNode(mean, cov, type="RANDOM")
                if not self.raw_map.is_gaussian_collision(g_node,
                                                      collision_check_method=collision_check_method,
                                                      alpha=self.alpha, cvar_threshold=self.cvar_threshold
                                                      ):
                    self.samples.append(mean)
                    self.gaussian_nodes.append(g_node)
            
        elif sampling_strategy == "HEXAGON":
            """
                Hexagonal map construction
            """

            # Function to create a hexagon centered at (x, y) with a given size (radius)
            def create_hexagon(center_x, center_y, size):
                return Polygon([(center_x + size * np.cos(2 * np.pi * i / 6), 
                                 center_y + size * np.sin(2 * np.pi * i / 6)) for i in range(6)])

            # Square map boundary
            map_boundary = Polygon([(0, 0), 
                                    (0, self.raw_map.height), 
                                    (self.raw_map.width, self.raw_map.height), 
                                    (self.raw_map.width, 0)])  

            hex_width = self.hex_radius * 2  # Distance between two horizontal vertices of a hexagon
            hex_height = np.sqrt(3) * self.hex_radius# Vertical distance between hexagon vertices

            # Create hexagonal grid points, reject points 
            grid_points = []
            for i in range(0, self.raw_map.width):  # Adjust range based on map size and hex size
                for j in range(0, self.raw_map.height):
                    x_offset = i * hex_width * 3 / 4  # Horizontal spacing
                    y_offset = j * hex_height + (i % 2) * (hex_height / 2)  # Offset every other row
                    hex_center = Point(x_offset, y_offset)
                    if map_boundary.contains(hex_center):
                        grid_points.append((x_offset, y_offset))

            for x, y in grid_points:
                hexagon = create_hexagon(x, y, self.hex_radius)
                if not self.raw_map.is_geometry_collision(hexagon):
                    node = np.array((x, y))
                    self.samples.append(node)
                    g_node = GaussianGraphNode(node, None, type="UNIFORM", 
                                               radius=self.hex_radius)
                    self.gaussian_nodes.append(g_node)

        elif sampling_strategy == "GAUSSIAN":
            assert False, "Unimplemented Gaussian sampling strategy"
        else:
            assert False, "Unimplemented sampling strategy"
        self.new_node_idx = len(self.samples)

    def build_roadmap(self, radius=50, roadmap_method="KDTREE", collision_check_method="CVAR"):
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
                         if not self.raw_map.is_gaussian_trajectory_collision(
                             self.gaussian_nodes[i],
                             self.gaussian_nodes[idx],
                             collision_check_method=collision_check_method)]
                self.roadmap.extend(edges)

        elif roadmap_method == "TRIANGULATION":
            boundary_points = self.raw_map.get_boundary_points(self.raw_map.obstacles, 10)
            points = np.concat(( self.samples, boundary_points))
            tri = Delaunay(points)
            for i, simplex in enumerate(tri.simplices):
                for i in range(-1, 2):
                    if  not (boundary_points == points[simplex[i]]).all(1).any()  \
                        and not (boundary_points == points[simplex[i+1]]).all(1).any()\
                        and (simplex[i], simplex[i+1]) not in self.roadmap \
                        and (simplex[i+1], simplex[i]) not in self.roadmap \
                        and np.linalg.norm(self.samples[simplex[i]]-self.samples[simplex[i+1]]) < radius \
                        and not self.raw_map.is_line_collision(self.gaussian_nodes[simplex[i]].mean, 
                                                           self.gaussian_nodes[simplex[i+1]].mean) \
                        and not self.raw_map.is_gaussian_trajectory_collision(
                             self.gaussian_nodes[simplex[i]],
                             self.gaussian_nodes[simplex[i+1]],
                             collision_check_method=collision_check_method, num_samples=20):
                        self.roadmap.append((int(simplex[i]), int(simplex[i+1])))
                        # Add path cost
                        g_node1 = self.gaussian_nodes[int(simplex[i])]
                        g_node2 = self.gaussian_nodes[int(simplex[i+1])]
                        self.roadmap_cost.append(gaussian_wasserstein_distance(
                            g_node1.mean, g_node1.covariance,
                            g_node2.mean, g_node2.covariance))

    def get_bounding_polygon(self):
        """
            Get bounding polygons of the map
        """
        return self.raw_map.get_bounding_polygon()

    def get_macro_solution(self, flow_dict):
        """
            Get macro solution indexed by timestep
        """
        macro_solution = {}

        for in_node in flow_dict:
            if in_node == ("SS", None):
                continue

            u, t = in_node
            for out_node, flow_value in flow_dict[in_node].items():
                if flow_value > 0:
                    if out_node == ('SG', None):
                        continue
                    v, _ = out_node

                    if t not in macro_solution:
                        macro_solution[t] = {}
                    
                    if u not in macro_solution[t]:
                        macro_solution[t][u] = []

                    macro_solution[t][u].append((v, flow_value))
        return macro_solution

    def get_obstacles(self):
        """
            Get obstacles in the space
        """
        return self.raw_map.get_obstacles()

    def get_solution(self, flow_dict, timestep, num_agent):
        """
            Return macro solution path per agent in DFS style
            TODO: Fix bug here
        """
        paths = []
        copy_dict = copy.deepcopy(flow_dict)
        for i in range(num_agent):
            paths.append([])
            u = ("SS", None) 
            while u != ("SG", None):
                for v in copy_dict[u]:
                    if copy_dict[u][v] > 0:
                        if v == ("SG", None):
                            u = v
                            break
                        else:
                            idx = v[0]
                            paths[-1].append(idx)
                            copy_dict[u][v] -= 1
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

# Visualization functions

    def visualize_map(self):
        """
            Visualize map only
        """
        fig, ax = plt.subplots(figsize=(10, 10))
        for start in self.instance.starts:
            pos = start.get_mean()
            ax.plot(pos[0], pos[1], 'bo', markersize=3)
            start.visualize_gaussian(ax=ax, cmap="Reds")
            # start.visualize(ax=ax, edgecolor="blue")

        for goal in self.instance.goals:
            pos = goal.get_mean()
            ax.plot(pos[0], pos[1], 'go', markersize=3)
            goal.visualize_gaussian(ax=ax, cmap="Blues")
            # goal.visualize(ax=ax, edgecolor="green")

        for obs in self.raw_map.obstacles:
            if obs.obs_type == "CIRCLE": 
                x, y = obs.get_pos()
                # ax.plot(x, y, 'ro', markersize=3)
                ax.add_patch(Circle((x, y), radius=obs.radius, color="black"))
            elif obs.obs_type == "POLYGON":
                x, y = obs.geom.exterior.xy
                ax.fill(x, y, fc="black")

        ax.set_aspect('equal')
        ax.set_xlim(left=0, right=self.raw_map.width)
        ax.set_ylim(bottom=0, top=self.raw_map.height)
        return fig, ax


    def visualize_solution(self, flow_dict, timestep, num_agent):
        """
            Visualize solution path per timestep provided the flow dict
        """
        node_idx = [i for i in range(len(self.samples))]
        for t in range(timestep):
            fig, ax = self.visualize_roadmap()

            # plot path at each timestep
            for i in node_idx:
                u = (i, t)
                for j in node_idx:
                    v = (j, t+1)
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
                    u = (i, t)
                    for j in node_idx:
                        v = (j, t+1)
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
            start.visualize_gaussian(ax=ax, cmap="Reds")
            # start.visualize(ax=ax, edgecolor="blue")

        for goal in self.instance.goals:
            pos = goal.get_mean()
            ax.plot(pos[0], pos[1], 'go', markersize=3)
            goal.visualize_gaussian(ax=ax, cmap="Blues")
            # goal.visualize(ax=ax, edgecolor="green")

        for obs in self.raw_map.obstacles:
            if obs.obs_type == "CIRCLE": 
                x, y = obs.get_pos()
                # ax.plot(x, y, 'ro', markersize=3)
                ax.add_patch(Circle((x, y), radius=obs.radius, color="black"))
            elif obs.obs_type == "POLYGON":
                x, y = obs.geom.exterior.xy
                ax.fill(x, y, fc="black")

        ax.set_aspect('equal')
        ax.set_xlim(left=0, right=self.raw_map.width)
        ax.set_ylim(bottom=0, top=self.raw_map.height)
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
        
        for start in self.starts:
            start.visualize_gaussian(ax=ax, cmap="Reds")

        for goal in self.goals:
            goal.visualize_gaussian(ax=ax, cmap="Blues")

        for obs in self.raw_map.obstacles:
            if obs.obs_type == "CIRCLE": 
                x, y = obs.get_pos()
                # ax.plot(x, y, 'ro', markersize=3)
                ax.add_patch(Circle((x, y), radius=obs.radius, color="black"))
            elif obs.obs_type == "POLYGON":
                x, y = obs.geom.exterior.xy
                ax.fill(x, y, fc="black")

        ax.set_aspect('equal')
        ax.set_xlim(left=0, right=self.raw_map.width)
        ax.set_ylim(bottom=0, top=self.raw_map.height)
        return fig, ax
