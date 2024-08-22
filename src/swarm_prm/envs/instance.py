"""
    Instance objects
"""
import numpy as np 
import os
import yaml

from swarm_prm.envs.roadmap import Roadmap, RoadmapLoader
from swarm_prm.solvers.swarm_prm.macro.gaussian_prm import GaussianNode

##### Instance #####

class Instance():
    def __init__(self, roadmap) -> None:
        pass

    def add_start(self, start:GaussianNode):
        pass

    def visualize(self):
        """
            Visualize instance
        """
        pass

##### Instance Generator #####

class InstanceGenerator:
    def __init__(self, map:Roadmap, num_agents, num_starts, num_goals, agent_radius=5,
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

##### Instance Loader #####

class InstanceLoader:
    """
        Return instance object given yaml file
    """
    def __init__(self, fname) -> None:
        self.fname = fname
        
        with open(fname, "r") as f:
            data = f.read()
        instance_dict = yaml.load(data, Loader=yaml.SafeLoader)
        instance = Instance()
        # roadmap = Map(map_dict["width"], map_dict["height"])
        # for obs in map_dict["obstacles"]:
            # roadmap.add_obstacle(CircleObstacle([obs["x"], obs["y"]], obs["radius"]))
        roadmap_loader = RoadmapLoader(instance_dict["map_fname"])
        self.roadmap = roadmap_loader.get_map()
 
        pass

    def get_instance(self):
        pass

    def visualize(self):
        """
            Visualize instance
        """
        pass