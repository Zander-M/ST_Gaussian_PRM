"""
    Generate map and instance. Problem instance includes agent starts and goals.
    All the obstacles on the map are static.
"""


import numpy as np
import os
import yaml

from envs.map_objects import *

# seeding the environment
np.random.seed(0)

class InstanceGenerator:
    def __init__(self, map:Map, num_agents, num_starts, num_goals, agent_radius=5,
                 instance_count=10, instance_dir="data/envs/instances") -> None:

        self.map = map
        self.num_agents = num_agents

        self.num_starts = num_starts
        self.num_goals = num_goals
        self.instance_count = instance_count
        self.instance_dir = instance_dir

        self.starts = []
        self.goals = []

        self.starts_num_agents = []
        self.goals_num_agents = []
        self.starts_weights = []
        self.goals_weights = []

        self.agent_radius = agent_radius

    
    def add_start(self, mean:np.ndarray, covariance:np.ndarray):
        """
            Add start node
        """
        start_dict = {}
        start_dict["mean"] = mean.to_list()
        start_dict["covariance"] = covariance.to_list()
        self.starts.append(start_dict)

    def add_goal(self, mean:np.ndarray, covariance:np.ndarray):
        """
            Add goal node
        """
        goal_dict = {}
        goal_dict["mean"] = mean.to_list()
        goal_dict["covariance"] = covariance.to_list()
        self.goals.append(goal_dict)

    def create_instance(self):
        """
            add random start and goal GMMs for agents on the map
        """
        min_x, max_x, min_y, max_y = 0, self.map.width, 0, self.map.height
        while len(self.starts) < self.num_starts:
            x = np.random.uniform(min_x, max_x)
            y = np.random.uniform(min_y, max_y)
            node = np.array((x, y))
            if self.map.is_point_collision(node):
                continue
            if self.map.get_clear_radius(node) < 10:
                continue
            mean = np.array((x, y))
            cov = 100 * np.identity(2)
            self.add_start(mean, cov)

        while len(self.goals) < self.num_goals:
            x = np.random.uniform(min_x, max_x)
            y = np.random.uniform(min_y, max_y)
            node = np.array((x, y))
            if self.map.is_point_collision(node):
                continue
            if self.map.get_clear_radius(node) < 10:
                continue
            mean = np.array((x, y))
            cov = 100 * np.identity(2)
            self.add_goal(mean, cov)

        # Randomly assign the agents to the starts and the goals
        start_idx = 1
        for _ in range(self.num_starts-1):
            temp = np.random.randint(start_idx, self.num_agents)
            self.starts_num_agents.append(temp-start_idx)
            start_idx = temp
        self.starts_num_agents.append(self.num_agents-start_idx)

        goal_idx = 1
        for _ in range(self.num_goal-1):
            temp = np.random.randint(goal_idx, self.num_agents)
            self.goals_num_agents.append(temp-goal_idx)
            goal_idx = temp
        self.goal_num_agents.append(self.num_agents-goal_idx)
        self.update_weights()


    def update_weights(self):
        """
            Update start/goal weights
        """
        self.starts_weights = np.array(self.starts_num_agents)/self.num_agents
        self.goals_weights = np.array(self.goals_num_agents)/self.num_agents

    def to_yaml(self):
        """
            Dump the config to yaml
        """
        instance_dict = dict()

        instance_dict["starts"] = self.starts
        instance_dict["goals"] = self.goals
        
        instance_dict["starts_num_agents"] = self.starts_num_agents
        instance_dict["goals_num_agents"] = self.goals_num_agents

        instance_dict["starts_weights"] = self.starts_weights
        instance_dict["goals_weights"] = self.goals_weights

        instance_dict["num_agents"] = self.num_agents
        instance_dict["map_name"] = self.map.map_name

        instance_name = "{}_agent{}.yaml".format(self.map.map_name, self.num_agents) 
        with open(os.path.join(self.instance_dir, instance_name), "w") as f:
            yaml.dump(instance_dict, f, sort_keys=False)

    def visualize_instance(self):
        """
            Visualize problem instance on the map
        """
        pass

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
            with open(os.path.join(self.map_dir, self.map_names[i]), "w") as f:
                yaml.dump(map_dict, f, sort_keys=False)

if __name__ == "__main__":
    # Generate Maps
    # map_generator = MapGenerator(100, 100)
    # map_generator.to_yaml()
    
    # Generate Instance
    instance_generator = InstanceGenerator()
    instance_generator.to_yaml()