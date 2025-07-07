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

from st_gaussian_prm.utils.gaussian_utils import *
from st_gaussian_prm.utils.cvt import CVT

### Sampling Methods

def uniform_sampling(raw_map, num_samples, **kwargs):
    """
        Sample uniformly in the configuration space
    """ 
    samples = []
    gaussian_nodes = []
    safety_radius = kwargs.get("safety_radius", None)
    check_collision = (
        lambda n: raw_map.is_radius_collision(n, safety_radius)
        if safety_radius is not None
        else raw_map.is_point_collision(n)
    )
    
    while len(samples) < num_samples:
        x = np.random.uniform(0, raw_map.width)
        y = np.random.uniform(0, raw_map.height)
        node = np.array((x, y))

        if check_collision(node):
            continue
        radius = np.inf
        for obs in raw_map.obstacles:
            radius = min(radius, obs.get_dist(node))
        g_node = GaussianGraphNode(node, None, type="UNIFORM", radius=radius)
        samples.append(node)
        gaussian_nodes.append(g_node)
    return samples, gaussian_nodes

def halton_sampling(raw_map, num_samples, **kwargs):
    """
        Halton sampling in the configuration space
    """
    samples = []
    gaussian_nodes = []
    halton_sampler = Halton(d=2) # 2d Halton sampler

    check_collision = (
        lambda n: raw_map.is_radius_collision(n, safety_radius)
        if safety_radius is not None
        else raw_map.is_point_collision(n)
    )

    while len(samples) < num_samples:
        num_to_generate = max(num_samples * 2, 100)
        sample_points = halton_sampler.random(num_to_generate)
        nodes = sample_points * np.array([raw_map.width, raw_map.height])
        safety_radius = kwargs.get("safety_radius", None)
        # covariance will be automatically updated.
        for node in nodes:
            if check_collision(node):
                continue
                # safety radius check
            radius = np.inf
            for obs in raw_map.obstacles:
                radius = min(radius, obs.get_dist(node))
            g_node = GaussianGraphNode(node, None, type="UNIFORM", radius=radius)
            samples.append(node)               
            gaussian_nodes.append(g_node)
            if len(samples) >= num_samples:
                break
        return samples, gaussian_nodes
            
def cvt_sampling(raw_map, num_samples, **kwargs):
    """
        Use Centroidal Voronoi Tessellation to sample nodes
    """
    cvt_instance = CVT(raw_map, num_samples, iteration=kwargs["iteration"], confidence_interval=kwargs["confidence_interval"])
    return cvt_instance.get_CVT()

def hexagon_sampling(raw_map, num_samples, **kwargs):
    """
        Hexagonal sampling points
    """
    samples = []
    gaussian_nodes = []

    # Function to create a hexagon centered at (x, y) with a given size (radius)
    def create_hexagon(center_x, center_y, size):
        return Polygon([(center_x + size * np.cos(2 * np.pi * i / 6), 
                         center_y + size * np.sin(2 * np.pi * i / 6)) for i in range(6)])

    # Square map boundary
    map_boundary = Polygon([(0, 0), 
                            (0, raw_map.height), 
                            (raw_map.width, raw_map.height), 
                            (raw_map.width, 0)])  

    hex_width = kwargs["hex_radius"] * 2  # Distance between two horizontal vertices of a hexagon
    hex_height = np.sqrt(3) * kwargs["hex_radius"]# Vertical distance between hexagon vertices

    # Create hexagonal grid points, reject points 
    grid_points = []
    for i in range(0, raw_map.width):  # Adjust range based on map size and hex size
        for j in range(0, raw_map.height):
            x_offset = i * hex_width * 3 / 4  # Horizontal spacing
            y_offset = j * hex_height + (i % 2) * (hex_height / 2)  # Offset every other row
            hex_center = Point(x_offset, y_offset)
            if map_boundary.contains(hex_center):
                grid_points.append((x_offset, y_offset))

    for x, y in grid_points:
        hexagon = create_hexagon(x, y, kwargs["hex_radius"])
        if not raw_map.is_geometry_collision(hexagon):
            node = np.array((x, y))
            samples.append(node)
            g_node = GaussianGraphNode(node, None, type="UNIFORM", 
                                       radius=kwargs["hex_radius"])
            gaussian_nodes.append(g_node)
    return samples, gaussian_nodes

sampling_methods = {
    "UNIFORM": uniform_sampling,
    "HALTON": halton_sampling,
    "CVT": cvt_sampling,
    "HEXAGON": hexagon_sampling
}

sampling_config = {
    "UNIFORM": {"safety_radius": 2.0},
    "HALTON": {"safety_radius": 2.0},
    "CVT": {"iteration": 500, "confidence_interval": 0.95},
    "HEXAGON": {"hex_radius": 2.0, }
}

### Gaussian PRM

class GaussianPRM:
    """
        Gaussian PRM
    """

    def __init__(self, obstacle_map, num_samples, 
                 alpha=0.95, cvar_threshold=-8,
                 swarm_prm_covariance_scaling=5,
                 cvt_iteration=10,
                 hex_radius=2
                 ) -> None:

        # PARAMETERS
        self.obstacle_map = obstacle_map
        self.num_samples = num_samples

        self.alpha = alpha
        self.cvar_threshold = cvar_threshold

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
        self.sample_free_space(sampling_strategy="CVT") # sample node locations 
        self.build_roadmap() # connect sample Gaussian nodes, building roadmap

    def sample_free_space(self, sampling_strategy="CVT"):
        """
            Sample points on the map uniformly random
        """
        sampling_method = sampling_methods[sampling_strategy]
        config = sampling_config[sampling_strategy]
        self.samples, self.gaussian_nodes = sampling_method(self.obstacle_map, self.num_samples, **config)

    def build_roadmap(self, radius=50):
        """
            Build Roadmap based on samples. Default connect radius is 50
        """
        # add start nodes and goal nodes to the graph
        boundary_points = self.obstacle_map.get_boundary_points(self.obstacle_map.obstacles, 10)
        points = np.concat(( self.samples, boundary_points))
        tri = Delaunay(points)
        for i, simplex in enumerate(tri.simplices):
            for i in range(-1, 2):
                if  not (boundary_points == points[simplex[i]]).all(1).any()  \
                    and not (boundary_points == points[simplex[i+1]]).all(1).any()\
                    and (simplex[i], simplex[i+1]) not in self.roadmap \
                    and (simplex[i+1], simplex[i]) not in self.roadmap \
                    and np.linalg.norm(self.samples[simplex[i]]-self.samples[simplex[i+1]]) < radius \
                    and not self.obstacle_map.is_line_collision(self.gaussian_nodes[simplex[i]].mean, 
                                                       self.gaussian_nodes[simplex[i+1]].mean) \
                    and not self.obstacle_map.is_gaussian_trajectory_collision(
                         self.gaussian_nodes[simplex[i]],
                         self.gaussian_nodes[simplex[i+1]]):
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
        return self.obstacle_map.get_bounding_polygon()

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

    def get_node_index(self, regions):
        """
            Get node indices contained in the provided polygon. Used to
            filter start nodes and goal nodes
        """
        indices = []
        for region in regions:
            for i, sample in enumerate(self.samples):
                if region.contains(Point(sample)):
                    indices.append(i)
        return indices
    
    def get_overlapping_index(self, regions):
        """
            Get node indices intersecting with the provided polygon. Used to
            avoid swarm-dynamic obstacle collisions.
        """
        indices = []
        for region in regions:
            for i, gaussian_node in enumerate(self.gaussian_nodes):
                if gaussian_node.is_collision(region):
                    indices.append(i)
        return indices
    
    def get_obstacles(self):
        """
            Get obstacles in the space
        """
        return self.obstacle_map.get_obstacles()

# Visualization functions

    def visualize_map(self):
        """
            Visualize map only
        """
        fig, ax = plt.subplots(figsize=(10, 10))
        for obs in self.obstacle_map.obstacles:
            if obs.obs_type == "CIRCLE": 
                x, y = obs.get_pos()
                # ax.plot(x, y, 'ro', markersize=3)
                ax.add_patch(Circle((x, y), radius=obs.radius, color="black"))
            elif obs.obs_type == "POLYGON":
                x, y = obs.geom.exterior.xy
                ax.fill(x, y, fc="black")

        ax.set_aspect('equal')
        ax.set_xlim(left=0, right=self.obstacle_map.width)
        ax.set_ylim(bottom=0, top=self.obstacle_map.height)
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

    def visualize_roadmap(self):
        """
            Visualize Gaussian PRM
        """
        fig, ax = plt.subplots(figsize=(10, 10))

        # Nodes and Paths 

        for node in self.samples:
            ax.plot(node[0], node[1], 'co', markersize=2)
        for (i, j) in self.roadmap:
            ax.plot([self.samples[i][0], self.samples[j][0]], [self.samples[i][1], self.samples[j][1]], 'gray', linestyle='-', linewidth=0.5)

        for obs in self.obstacle_map.obstacles:
            if obs.obs_type == "CIRCLE": 
                x, y = obs.get_pos()
                # ax.plot(x, y, 'ro', markersize=3)
                ax.add_patch(Circle((x, y), radius=obs.radius, color="black"))
            elif obs.obs_type == "POLYGON":
                x, y = obs.geom.exterior.xy
                ax.fill(x, y, fc="black")

        ax.set_aspect('equal')
        ax.set_xlim(left=0, right=self.obstacle_map.width)
        ax.set_ylim(bottom=0, top=self.obstacle_map.height)
        return fig, ax
    
    def visualize_g_nodes(self):
        """
            Visualize Gaussian Nodes on the map
        """
        fig, ax = plt.subplots(figsize=(10, 10))

        # Visualize G nodes
        cmap = plt.get_cmap('tab10')
        for i, g_node in enumerate(self.gaussian_nodes):
            # g_node.visualize(ax=ax, edgecolor=cmap(i%10))
            g_node.visualize(ax=ax, edgecolor='gray')
            x, y = g_node.get_mean()
            # ax.text(x, y, str(i), fontsize=8, ha='center', va='center', color='black')
        
        for obs in self.obstacle_map.obstacles:
            if obs.obs_type == "CIRCLE": 
                x, y = obs.get_pos()
                # ax.plot(x, y, 'ro', markersize=3)
                ax.add_patch(Circle((x, y), radius=obs.radius, color="black"))
            elif obs.obs_type == "POLYGON":
                x, y = obs.geom.exterior.xy
                ax.fill(x, y, fc="black")

        ax.set_aspect('equal')
        ax.set_xlim(left=0, right=self.obstacle_map.width)
        ax.set_ylim(bottom=0, top=self.obstacle_map.height)
        return fig, ax
