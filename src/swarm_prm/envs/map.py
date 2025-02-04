"""
    Roadmap objects
"""

from matplotlib import pyplot as plt
from matplotlib.patches import Circle
import numpy as np
import os
from scipy.stats import norm
from shapely.geometry import LineString, Point, Polygon
import yaml

from swarm_prm.solvers.utils.gaussian_utils import GaussianNode

##### Map       #####

class Map:
    def __init__(self, width, height, map_name=None) -> None:
        self.width = width
        self.height = height
        self.obstacles:list[Obstacle] = []
        self.map_name = map_name

    def get_map_size(self):
        """
            Return map size
        """
        return np.array([self.width, self.height])
    
    def add_obstacle(self, obstacle):
        """
            Add obstacles to the environment
        """
        self.obstacles.append(obstacle)
    
    def get_clear_radius(self, point):
        """
            Return clear radius from the selected point. Equivalent to the distance
            to the closest obstacle
        """
        clear_radius = np.inf
        for obs in self.obstacles:
            clear_radius = min(clear_radius, obs.get_dist(point))
        return clear_radius
    
    def get_closest_obstacle(self, point):
        """
            Return the closest obstacle from the point. 
        """
        dist = [obs.get_dist(point) for obs in self.obstacles]
        return self.obstacles[np.argsort(dist)[0]]

    def get_bounding_polygon_shapely(self):
        """
           Get bounding polygon of the space 
        """
        return Polygon([(0, 0), (self.width, 0), (self.width, self.height), (0, self.height)])

    def get_obstacles(self):
        """
            Return obstacles
        """
        return self.obstacles
    
    def get_obstacles_shapely(self):
        """
            Return obstacles in shapely geometry form
        """
        return [obs.geom for obs in self.obstacles]

    def is_line_collision(self, line_start, line_end):
        """
            Check if a line collides with the environment
        """

        # boundary check

        if line_start[0] < 0 or line_start[0] > self.width:
            return True

        if line_start[1] < 0 or line_start[1] > self.height:
            return True

        if line_end[0] < 0 or line_end[0] > self.width:
            return True

        if line_end[1] < 0 or line_end[1] > self.height:
            return True

        # obstacle check
        for obs in self.obstacles:
            if obs.is_line_colliding(line_start, line_end):
                return True
        return False

    def is_point_collision(self, point):
        """
            Check if a point collides with the environment.
        """
        # Boundary Checks
        if point[0] < 0 or point[0] > self.width:
            return True

        if point[1] < 0 or point[1] > self.height:
            return True

        # Obstacle Checks
        for obs in self.obstacles:
            if obs.is_point_colliding(point):
                return True
        return False

    def is_radius_collision(self, point, radius):
        """
            Check if a point is a least radius away from obstacles in the environment 
        """
        # Boundary Checks
        if point[0] < radius or point[0] > self.width - radius:
            return True

        if point[1] < radius or point[1] > self.height - radius:
            return True

        # Obstacle Checks
        for obs in self.obstacles:
            if obs.is_radius_colliding(point, radius):
                return True
        return False

    def is_gaussian_trajectory_collision(self, start:GaussianNode, goal:GaussianNode,
                                        num_samples=10, 
                                        mc_threshold=0.9, 
                                        collision_check_method="MONTE_CARLO") -> bool:
        """
            Linearly interpolate between two Gaussian distributions and check if the trajectory
            collides with the environment. Return True if it collides with the environment
        """
        mean1, cov1 = start.get_gaussian()
        mean2, cov2 = goal.get_gaussian()
        for step in range(num_samples):
            mean = (step/num_samples)* mean1 + (1-(step/num_samples)) * mean2 
            cov = (step/num_samples)* cov1 + (1-(step/num_samples)) * cov2 
            g_node = GaussianNode(mean, cov)

            if self.is_gaussian_collision(g_node, mc_threshold=mc_threshold, 
                                          collision_check_method=collision_check_method):
                return True
        return False
        
    def is_gaussian_collision(self, g_node:GaussianNode, 
                              collision_check_method="MONTE_CARLO", 
                              mc_num_samples=100, mc_threshold=0.9,
                              alpha=0.9, cvar_threshold=-0.02) -> bool:
        """
            Perform collision checks of the distribution will collide
            with the environment.  
            MONTE-CARLO: using sampling to check if distribution collides with environment
            CVAR: using closest point to obstacle to check if obstacle will collide
        """
        if collision_check_method == "MONTE_CARLO":
            samples = g_node.get_samples(mc_num_samples)
            count = 0
            for sample in samples:
                count += 1 if self.is_point_collision(sample) else 0
            ratio = 1 - count / mc_num_samples
            return ratio < mc_threshold

        elif collision_check_method == "CVAR":

            for obs in self.obstacles:
                if obs.is_gaussian_colliding(g_node, alpha, cvar_threshold):
                    return True
            return False
        else: 
            assert False, "Unimplemented Collision Checking Method"
    
    def is_geometry_collision(self, geom):
        """
            Check if Geometry collides with the obstacles on the map
        """
        if geom.centroid.xy[0][0] < 0 or geom.centroid.xy[0][0] > self.width:
            return True

        if geom.centroid.xy[1][0] < 0 or geom.centroid.xy[1][0] > self.height:
            return True

        # obstacle check
        for obs in self.obstacles:
            if obs.is_geometry_colliding(geom):
                return True
        return False

    def visualize(self, fig=None, ax=None):
        """
            Visualize map
        """
        if ax is None:
            fig, ax = plt.subplots()
        ax.set_xlim((0., self.width))
        ax.set_ylim((0., self.height))
        ax.set_aspect('equal')
        for obs in self.get_obstacles():
            if obs.obs_type == "CIRCLE": 
                x, y = obs.get_pos()
                ax.add_patch(Circle((x, y), radius=obs.radius, color="black"))
            elif obs.obs_type == "POLYGON":
                x, y = obs.geom.exterior.xy
                ax.fill(x, y, fc="black")
        return fig, ax

    def get_openfoam_config(self):
        """
            TODO: Convert current map to openfoam initial conditions
        """
        pass

##### Obstacles #####

class Obstacle:
    """
        Obstacles on the map
    """

    def __init__(self, pos, obs_type, *args):
        """
            Obstacle base class
            For Circle obstacle, pass in radius as args
            For Polygon obstacle, pass in ABSOLUTE coordinates of the points
        """

        self.pos = pos
        self.obs_type = obs_type 
        if self.obs_type == "CIRCLE":
            self.geom = Point(pos).buffer(args[0]) 
            self.radius = args[0]
        elif self.obs_type == "POLYGON":
            self.geom = Polygon(args[0])
            self.pos = np.array([self.geom.centroid.x, self.geom.centroid.y])
        else:
            assert False, "Obstacle must be either circle or polygon."

    def get_pos(self):
        """
            Return position
        """
        return self.pos

    def get_dist(self, point):
        """
            check point to obstacle distance
        """
        point_geom = Point(point)
        return point_geom.distance(self.geom)

    def get_edge_segments(self, segment_length=2):
        """
            Get edge segments for delaunay triangulation
            Return: points, segmetns, pos 
        """
        if self.obs_type == "CIRCLE":
            n_points = int(2 * self.radius * np.pi // segment_length)

            # calculate points on the circle
            i = np.arange(n_points)
            theta = np.linspace(0, 2 * np.pi, n_points, endpoint=False)
            pts = np.stack([np.cos(theta), np.sin(theta)], axis=1) * self.radius + self.get_pos()
            segs = np.stack([i, i+1], axis=1) % n_points
            return pts, segs, self.get_pos()

        elif self.obs_type == "POLYGON":

            def _split_edge(edge, segment_length):
                """Split a LineString edge into segments of specified length."""
                pts = []

                num_segments = int(edge.length // segment_length)
                if edge.length % segment_length != 0:
                    num_segments += 1
                # Create each segment
                for i in range(num_segments):
                    pts.append(np.array(edge.interpolate(i * segment_length).xy).reshape(2))

                return pts
            all_pts = []

            # Extract and split each edge of the polygon
            for i in range(len(self.geom.exterior.coords) - 1):
                # Each edge of the polygon as a LineString
                edge = LineString([self.geom.exterior.coords[i], self.geom.exterior.coords[i + 1]])
                pts = _split_edge(edge, segment_length)
                all_pts.extend(pts)          

            idx = np.arange(len(all_pts))

            segs = np.stack([idx, idx+1], axis=1) % len(all_pts)

            return np.array(all_pts), segs, self.get_pos()
        else: 
            assert False, "Unimplemented obstacle type."

    def is_point_colliding(self, point):
        """
            check if point collides with obstacle 
        """
        point_geom = Point(point)
        return self.geom.contains(point_geom)

    def is_line_colliding(self, line_start, line_end):
        """
            check if line collides with obstacle 
        """
        line = LineString([line_start, line_end])
        return self.geom.intersects(line)

    def is_radius_colliding(self, point, radius):
        """
            check if obstacles is with the radius distance from the point
        """
        pt = Point(point).buffer(radius)
        return self.geom.intersects(pt)
    
    def is_gaussian_colliding(self, g_node:GaussianNode, alpha, threshold):
        """
            check if Gaussian distribution is too close to the obstacles
        """
        mean = -self.get_dist(g_node.get_mean())
        v_normal = (self.pos - g_node.get_mean()) / np.linalg.norm(self.pos - g_node.get_mean())
        variance = v_normal.T @ g_node.covariance @ v_normal
        ita = norm(mean, variance)
        cvar = mean + ita.pdf(ita.ppf(1-alpha))/alpha * variance # type: ignore
        return cvar > threshold
    
    def is_geometry_colliding(self, geom):
        """
            check if provided geometry 
        """
        return self.geom.intersects(geom)

##### Map Generator #####

class MapGenerator:
    """
        Map generator. Generate yaml file representing the map config.
    """
    def __init__(self, width, height, radius_min=5, radius_max=10, map_count=10, 
                obs_count=10, map_fname="map", roadmap_dir="data/envs/maps") -> None:
        self.width = width
        self.height = height
        self.radius_min= radius_min
        self.radius_max = radius_max
        self.roadmap_fname= map_fname
        self.roadmap_count = map_count
        self.roadmap_dir = roadmap_dir
        self.obs_count = obs_count
        self.roadmap_names = []
        self.roadmaps = []
        for i in range(self.roadmap_count):
            self.roadmap_names.append("{}_{:01d}.yaml".format(self.roadmap_fname, i))
            map_instance = Map(self.width, self.height)
            self.add_obstacles(map_instance)
            self.roadmaps.append(map_instance)

    def add_obstacles(self, map_instance):
        """
            Generate random obstacles on the map
            Adding circular obstacles for now
            num: number of random obstacles

        """
        for _ in range(self.obs_count):
            x = np.random.random() * self.width
            y = np.random.random() * self.height
            rand = np.random.random()
            radius = rand * self.radius_min + (1-rand) * self.radius_max
            obs = Obstacle((x, y), "CIRCLE", radius)
            map_instance.add_obstacle(obs)

    def to_yaml(self):
        """
            save map to yaml file
            FIXME: does not work with polygon obstacles
        """
        for i, map_instance in enumerate(self.roadmaps):
            map_dict = dict()
            map_dict["height"] = map_instance.height
            map_dict["width"] = map_instance.width
            map_dict["obstacles"] = []
            for obs in map_instance.obstacles:
                d = dict()
                x, y = obs.get_pos()
                d["x"] = x
                d["y"] = y
                if obs.obs_type == "CIRCLE":
                    d["radius"] = obs.radius
                map_dict["obstacles"].append(d)
            with open(os.path.join(self.roadmap_dir, self.roadmap_names[i]), "w") as f:
                yaml.dump(map_dict, f, sort_keys=False)

##### Map Loader #####

class MapLoader:
    """
        Return map object given yaml file
    """
    def __init__(self, fname) -> None:
        self.fname = fname
        with open(fname, "r") as f:
            data = f.read()
        map_dict = yaml.load(data, Loader=yaml.SafeLoader)
        roadmap = Map(map_dict["width"], map_dict["height"])
        for obs in map_dict["obstacles"]:
            roadmap.add_obstacle(Obstacle([obs["x"], obs["y"]], "CIRCLE", obs["radius"]))
        self.map = roadmap
    
    def get_map(self):
        """
            Get roadmap
        """
        return self.map
