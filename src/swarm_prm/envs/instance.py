"""
    Instance objects
"""
from matplotlib import pyplot as plt
import numpy as np 
import os
from typing import List
import yaml

from swarm_prm.envs.map import Map, MapLoader
from swarm_prm.solvers.macro.swarm_prm.gaussian_utils import GaussianGraphNode

##### Instance #####

class Instance:
    def __init__(self, map:Map, starts:List[GaussianGraphNode], goals:List[GaussianGraphNode],
                 starts_weight, goals_weight, num_agent) -> None:
        self.map = map
        self.starts = starts
        self.goals = goals
        self.starts_weight = starts_weight
        self.goals_weight = goals_weight
        self.num_agent = num_agent

    def add_start(self, start):
        self.starts.append(start)

    def add_goal(self, goal):
        self.goals.append(goal)

    def visualize(self):
        """
            Visualize instance
        """
        fig, ax = plt.subplots()
        self.map.visualize(ax=ax)

        for start in self.starts:
            start.visualize(ax, edgecolor="green")

        for goal in self.goals:
            goal.visualize(ax, edgecolor="blue")
        
        return fig, ax


##### Instance Generator #####

class InstanceGenerator:
    def __init__(self, map:Map, num_agents, num_starts, num_goals, agent_radius=5,
                 instance_count=10, instance_dir="data/envs/instances") -> None:

        self.roadmap = map
        self.num_agents = num_agents

        self.num_starts = num_starts
        self.num_goals = num_goals
        self.instance_count = instance_count
        self.instance_dir = instance_dir

        self.starts = []
        self.goals = []

        self.starts_weights = []
        self.goals_weights = []

        self.agent_radius = agent_radius
    
    def add_start(self, mean:np.ndarray, covariance:np.ndarray):
        """
            Add start node
        """
        start_dict = {}
        start_dict["mean"] = mean.tolist() 
        start_dict["covariance"] = covariance.flatten().tolist()
        self.starts.append(start_dict)

    def add_goal(self, mean:np.ndarray, covariance:np.ndarray):
        """
            Add goal node
        """
        goal_dict = {}
        goal_dict["mean"] = mean.tolist()
        goal_dict["covariance"] = covariance.flatten().tolist()
        self.goals.append(goal_dict)

    def create_instance(self):
        """
            add random start and goal GMMs for agents on the map
        """
        min_x, max_x, min_y, max_y = 0, self.roadmap.width, 0, self.roadmap.height
        while len(self.starts) < self.num_starts:
            x = np.random.uniform(min_x, max_x)
            y = np.random.uniform(min_y, max_y)
            node = np.array((x, y))
            if self.roadmap.is_point_collision(node):
                continue

            # Using a threshhold to guarantee available space for start locations
            if self.roadmap.get_clear_radius(node) < 10:
                continue
            mean = np.array((x, y))
            cov = 100 * np.identity(2)
            self.add_start(mean, cov)

        while len(self.goals) < self.num_goals:
            x = np.random.uniform(min_x, max_x)
            y = np.random.uniform(min_y, max_y)
            node = np.array((x, y))
            if self.roadmap.is_point_collision(node):
                continue
            if self.roadmap.get_clear_radius(node) < 10:
                continue
            mean = np.array((x, y))
            cov = 100 * np.identity(2)
            self.add_goal(mean, cov)

        # Randomly assign the agents to the starts and the goals
        start_idx = 1
        starts_num_agents = []
        for _ in range(self.num_starts-1):
            temp = np.random.randint(start_idx, self.num_agents)
            starts_num_agents.append(temp-start_idx)
            start_idx = temp
        starts_num_agents.append(self.num_agents-start_idx)

        goal_idx = 1
        goals_num_agents = []
        for _ in range(self.num_goals-1):
            temp = np.random.randint(goal_idx, self.num_agents)
            goals_num_agents.append(temp-goal_idx)
            goal_idx = temp
        goals_num_agents.append(self.num_agents-goal_idx)
        self.starts_weights = np.array(starts_num_agents)/self.num_agents
        self.goals_weights = np.array(goals_num_agents)/self.num_agents

    def to_yaml(self):
        """
            Dump the config to yaml
        """
        instance_dict = dict()

        instance_dict["starts"] = self.starts
        instance_dict["goals"] = self.goals
        
        instance_dict["starts_weights"] = self.starts_weights
        instance_dict["goals_weights"] = self.goals_weights

        instance_dict["num_agents"] = self.num_agents
        instance_dict["map_name"] = self.roadmap.map_name

        instance_name = "{}_agent{}.yaml".format(self.roadmap.map_name, self.num_agents) 
        with open(os.path.join(self.instance_dir, instance_name), "w") as f:
            yaml.dump(instance_dict, f, sort_keys=False)

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
        roadmap_loader = MapLoader(instance_dict["roadmap_fname"])
        roadmap = roadmap_loader.get_map()

        starts = []
        for start in instance_dict["starts"]:
            mean = np.array(start["mean"]) 
            cov = np.array(start["covariance"]).reshape((2, 2)) 
            starts.append(GaussianGraphNode(mean, cov))

        goals = []
        for goal in instance_dict["goals"]:
            mean = np.array(goal["mean"]).astype(np.float32) 
            cov = np.array(goal["covariance"]).astype(np.float32).reshape((2, 2)) 
            goals.append(GaussianGraphNode(mean, cov))
        
        starts_weight = np.array(instance_dict["starts_weights"])
        goals_weight = np.array(instance_dict["goals_weights"])

        num_agent = instance_dict["num_agents"]
        
        self.instance = Instance(roadmap, starts, goals, starts_weight, goals_weight, num_agent)
 
    def get_instance(self):
        return self.instance
