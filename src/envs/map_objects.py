"""
    Map objects
"""

from abc import abstractmethod
import math
from collections import namedtuple

import numpy as np

# Name tuple for point

Point = namedtuple('Point', ["x", "y"])

##### Map       #####

class Map:
    def __init__(self, width, height) -> None:
        self.width = width
        self.height = height
        self.obstacles = []

    def add_obstacle(self, obstacle):
        """
            Add obstacles to the environment
        """
        self.obstacles.append(obstacle)
    
    def collision_check(self, point):
        """
            Check if a point collides with the environment.
        """

        # Boundary Checks

        # Obstacle Checks

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
    def dist(self, point) -> float:
        """
            check point to obstacle distance
        """
        pass

    @abstractmethod
    def is_colliding(self, point) -> bool:
        """
            check if point collides with obstacle 
        """
        pass

class CircleObstacle(Obstacle):
    """
        Circular obstacle
    """
    def __init__(self, pos, radius):
        super().__init__(pos, "Circle")
        self.radius = radius

    def dist(self, point):
        """
            distance to circle
        """
        p_x, p_y = point
        return math.sqrt((self.pos[0]-p_x)**2 + (self.pos[1]-p_y)**2) - self.radius
    
    def is_colliding(self, point) -> bool:
        return self.dist(point) <= 0

class PolygonObstacle(Obstacle):
    """
        Polygonal obstacle.
        Vertices are relative to the absolute pos
        TODO: fix gjk
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
