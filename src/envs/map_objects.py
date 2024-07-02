"""
    Map objects
"""

import math

class Obstacle:
    def __init__(self, obstacle_type, *args):
        self.obstacle_type = obstacle_type
        self.params = args

    def is_point_inside(self, point):
        if self.obstacle_type == 'polygon':
            return self.point_in_polygon(point, *self.params)
        elif self.obstacle_type == 'sphere':
            return self.point_in_sphere(point, *self.params)
        else:
            raise ValueError("Unsupported obstacle type")

    def point_in_polygon(self, point, vertices):
        # Check if a point is inside a polygon using ray-casting algorithm
        x, y = point
        inside = False
        vertices_count = len(vertices)

        j = vertices_count - 1
        for i in range(vertices_count):
            xi, yi = vertices[i]
            xj, yj = vertices[j]

            intersect = ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi) + xi)
            if intersect:
                inside = not inside

            j = i

        return inside

    def point_in_sphere(self, point, center, radius):
        # Check if a point is inside a sphere
        px, py = point
        cx, cy, cz = center
        distance = math.sqrt((px - cx) ** 2 + (py - cy) ** 2)

        return distance <= radius

class Maps:
    def __init__(self, width, height, obs_density) -> None:
        pass

class Instance:
    def __init__(self, map) -> None:
        self.map = map
        self.objects = []
        pass