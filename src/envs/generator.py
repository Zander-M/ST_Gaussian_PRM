"""
    Generate map and instance. Problem instance includes agent starts and goals.
    All the obstacles on the map are static.
"""

import numpy as np

from map_objects import *

class InstanceGenerator:
    def __init__(self, map) -> None:
        self.map = map

class MapGenerator:
    """
        Map generator. Generate yaml file representing the map config.
    """
    def __init__(self, width, height) -> None:
        self.width = width
        self.height = height
        self.obstacles = []

    def gen_obstacles(self, num):
        """
            Generate random obstacles on the map
            num: number of random obstacles
        """
        for _ in range(num):
            pass

    def add_obstacle(self, pos, obs_type, *args):
        """
            Add obstacles to the map
        """ 
        pass

    def to_yaml(self):
        """
            save map to yaml file
        """
        pass

if __name__ == "__main__":
    map_generator = MapGenerator()
    instance_generator = InstanceGenerator()