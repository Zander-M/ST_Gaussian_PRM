"""
    Roadmap objects
"""

from abc import abstractmethod
from matplotlib import pyplot as plt
import numpy as np
import os
from scipy.stats import norm
import yaml

from swarm_prm.solvers.swarm_prm.macro.gaussian_utils import GaussianNode

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
        clear_radius = np.Infinity
        for obs in self.obstacles:
            clear_radius = min(clear_radius, obs.get_dist(point))
        return clear_radius

    def get_obstacles(self):
        """
            Return obstacles
        """
        return self.obstacles

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

    def visualize(self, fig=None, ax=None):
        """
            Visualize map
        """
        if ax is None:
            fig, ax = plt.subplots()
        ax.set_xlim([0, self.width])
        ax.set_ylim([0, self.height])
        for obs in self.get_obstacles():
            x, y = obs.get_pos()
            # ax.plot(x, y, 'ro', markersize=3)
            ax.add_patch(plt.Circle((x, y), radius=obs.radius, color="black"))
        ax.set_aspect('equal')
        return fig, ax

##### Obstacles #####

class Obstacle:
    """
        Obstacles on the map
    """

    def __init__(self, pos, obs_type):
        self.pos = pos
        self.obs_type= obs_type 

    def get_pos(self):
        """
            Return position
        """
        return self.pos

    @abstractmethod
    def get_dist(self, point) -> float:
        """
            check point to obstacle distance
        """
        assert False, "get_dist not implemented"

    @abstractmethod
    def is_point_colliding(self, point) -> bool:
        """
            check if point collides with obstacle 
        """
        assert False, "is_point_colliding not implemented"

    @abstractmethod
    def is_line_colliding(self, line_start, line_end) -> bool:
        """
            check if line collides with obstacle 
        """
        assert False, "is_line_colliding not implemented"

    @abstractmethod
    def is_radius_colliding(self, point, radius) -> bool:
        """
            check if obstacles is with the radius distance from the point
        """
        assert False, "is_radius_colliding not implemented"
    
    @abstractmethod
    def is_gaussian_colliding(self, g_node:GaussianNode, cvar, threshold) -> bool:
        """
            check if Gaussian distribution is too close to the obstacles
        """

        assert False, "is_gaussian_colliding not implemented"

class CircleObstacle(Obstacle):
    """
        Circular obstacle
    """
    def __init__(self, pos, radius):
        super().__init__(pos, "CIRCLE")
        self.radius = radius

    def get_dist(self, point):
        """
            distance to circle
        """
        return np.linalg.norm(point-self.pos)-self.radius

    def is_point_colliding(self, point) -> bool:
        return self.get_dist(point) <= 0 

    def is_line_colliding(self, line_start, line_end):
    
        # Compute the line vector
        line_vec = line_end - line_start
        line_length = np.linalg.norm(line_vec)
    
        # Compute the vector from the start of the line to the point
        point_vec = self.pos - line_start
    
        # Project point_vec onto line_vec to find the nearest point on the line
        t = np.dot(point_vec, line_vec) / line_length**2
    
        # Restrict t to the range [0, 1] to stay within the line segment
        t = max(0, min(1, t))
    
        # Find the nearest point on the line segment
        nearest_point = line_start + t * line_vec
    
        # Compute the distance from the point to the nearest point on the line segment
        distance = np.linalg.norm(self.pos - nearest_point)
    
        return distance <= self.radius
    
    def is_radius_colliding(self, point, radius) -> bool:
        return self.get_dist(point) <= radius 

    def is_gaussian_colliding(self, g_node: GaussianNode, alpha, threshold) -> bool:
        """
            Using CVaR and threshold to test if node is too close to obstacle.
            Return True if CVaR is greater than the threshold.
            Reference: SwarmPRM
        """

        mean = -self.get_dist(g_node.get_mean())
        v_normal = (self.pos - g_node.get_mean()) / np.linalg.norm(self.pos - g_node.get_mean())
        variance = v_normal.T @ g_node.covariance @ v_normal
        ita = norm(mean, variance)
        cvar = mean + ita.pdf(ita.ppf(1-alpha))/alpha * variance
        return cvar > threshold

class PolygonObstacle(Obstacle):
    """
        Polygonal obstacle.
        Vertices are relative to the absolute pos
        TODO: fix gjk
        TODO: fix collision check
    """
    def __init__(self, pos, nums):
        super().__init__(pos, "Polygon")
        self.polygon= []

    def dist(self, point):
        """
            distance between a point and the polygon
            computed with gjk algorithm 
        """

        def dot(v1, v2):
            return np.dot(v1, v2)

        def support(points, d):
            """ 
            Find the furthest point in the direction d from the origin in the set of points.
            """
            return max(points, key=lambda p: dot(p, d))

        def perpendicular(v):
            return np.array([-v[1], v[0]])

        simplex = []
        direction = point - self.polygon[0]  # Initial direction from the point to the first vertex of the polygon

        while True:
            # Get a new point in the direction
            new_point = support(self.polygon, direction)
    
            # If the point we got is not past the origin in the direction, the distance is zero
            if dot(new_point, direction) <= 0:
                return 0

            simplex.append(new_point)

            if len(simplex) == 3:
                # Get the edges of the triangle
                a, b, c = simplex
                ab = b - a
                ac = c - a
                ao = -a

                # Perpendicular vectors to the edges
                ab_perp = perpendicular(ab)
                ac_perp = perpendicular(ac)

                if dot(ab_perp, ao) > 0:
                    simplex = [a, b]
                    direction = ab_perp
                else:
                    simplex = [a, c]
                    direction = ac_perp
            else:
                a, b = simplex
                ab = b - a
                ao = -a

                direction = perpendicular(ab)
                if dot(direction, ao) < 0:
                    direction = -direction

            if len(simplex) == 3:
                break

        # Calculate the distance from the point to the polygon
        a, b, c = simplex
        ab = b - a
        ac = c - a
        ao = -a

        ab_perp = perpendicular(ab)
        ac_perp = perpendicular(ac)

        if dot(ab_perp, ao) > 0:
            return np.linalg.norm(ab_perp)
        else:
            return np.linalg.norm(ac_perp)

    def is_line_colliding(self, line_start, line_end) -> bool:
        return super().is_line_colliding(line_start, line_end)

    def is_point_colliding(self, line_start, line_end) -> bool:
        return super().is_line_colliding(line_start, line_end)

    def is_radius_colliding(self, point, radius) -> bool:
        return super().is_radius_colliding(point, radius)
    
    def is_gaussian_colliding(self, g_node: GaussianNode) -> bool:
        return super().is_gaussian_colliding(g_node)

##### Map Generator #####

class MapGenerator:
    """
        Map generator. Generate yaml file representing the map config.
    """
    def __init__(self, width, height, radius_min=5, radius_max=10, map_count=10, 
                obs_count=10, map_fname="map", map_dir="data/envs/maps") -> None:
        self.width = width
        self.height = height
        self.radius_min= radius_min
        self.radius_max = radius_max
        self.roadmap_fname= map_fname
        self.roadmap_count = map_count
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
            obs = CircleObstacle((x, y), radius)
            map_instance.add_obstacle(obs)

    def to_yaml(self):
        """
            save map to yaml file
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
            roadmap.add_obstacle(CircleObstacle([obs["x"], obs["y"]], obs["radius"]))
        self.map = roadmap
    
    def get_map(self):
        """
            Get roadmap
        """
        return self.map
