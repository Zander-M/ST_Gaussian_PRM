"""
    Generate map and instance. Problem instance includes agent starts and goals.
    All the obstacles on the map are static.
"""

import numpy as np
import yaml

from envs.map_objects import *

class InstanceGenerator:
    def __init__(self, map, num_of_agents) -> None:
        self.map = map
        self.num_of_agents = num_of_agents
        

    def create_instance(self):
        """
            add random start and goal GMMs for agents on the map
        """
        pass

    def to_yaml(self):
        """
            Dump the config to yaml
        """
        pass
        

class MapGenerator:
    """
        Map generator. Generate yaml file representing the map config.
    """
    def __init__(self, width, height, radius_min=5, radius_max=10, map_count=10, 
                obs_count=10, map_fname="map", map_dir="data/envs/") -> None:
        self.width = width
        self.height = height
        self.radius_min= radius_min
        self.radius_max = radius_max
        self.map_fname= map_fname
        self.map_count = map_count
        self.obs_count = obs_count
        self.map_names = []
        self.maps = []
        for i in range(self.map_count):
            self.map_names.append("{}_{:01d}.yaml".format(self.map_fname, i))
            map_instance = Map(self.width, self.height)
            self.add_obstacles(map_instance)
            self.maps.append(map_instance)

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
        for i, map_instance in enumerate(self.maps):
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
            with open(self.map_names[i], "w") as f:
                yaml.dump(map_dict, f, sort_keys=False)

if __name__ == "__main__":
    map_generator = MapGenerator(100, 100)
    map_generator.to_yaml()
    # instance_generator = InstanceGenerator()