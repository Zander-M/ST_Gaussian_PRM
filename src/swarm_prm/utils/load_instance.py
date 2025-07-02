"""
    Load planning instance provided instance dict
"""
# Load map and instance
import os
import pickle
from collections import Counter

import numpy as np 
from swarm_prm.solvers.macro import SOLVER_REGISTRY

def load_instance(instance_config, solver_config):
    """
        Load instance and return a solver object
    """
    map_fname = "{}_{}.pkl".format(instance_config["map_type"], instance_config["num_samples"])
    fname = os.path.join("../maps", map_fname)
    with open(fname, "rb") as f:
        gaussian_prm = pickle.load(f) 

    node_capacities = [g_node.get_capacity(instance_config["agent_radius"]) for g_node in gaussian_prm.gaussian_nodes]
    print("average node capacity: ", np.average(node_capacities))

    # Randomly choose starts and goals from the candidates, and assign them a weight
    # Make sure all agents can fit in the start and goal configuration

    starts_idx_set = gaussian_prm.get_node_index(instance_config["start_regions"])
    goals_idx_set = gaussian_prm.get_node_index(instance_config["goal_regions"])

    starts_idx = np.random.choice(starts_idx_set, instance_config["num_starts"], replace=False).tolist() # type:ignore
    goals_idx = np.random.choice(goals_idx_set, instance_config["num_goals"], replace=False).tolist() # type:ignore 

    # Create weight pool

    starts_pool = []
    goals_pool = []

    for start_idx in starts_idx:
        starts_pool += [start_idx] * gaussian_prm.gaussian_nodes[start_idx].get_capacity(instance_config["agent_radius"])

    for goal_idx in goals_idx:
        goals_pool += [goal_idx] * gaussian_prm.gaussian_nodes[goal_idx].get_capacity(instance_config["agent_radius"])

    # Decide agent based on capacity

    instance_capacity = min(len(starts_pool), len(goals_pool))
    num_agents = int(instance_capacity * instance_config["capacity_percentage"])

    # Distribute agents to goals
    start_per_agent = np.random.choice(starts_pool, num_agents, replace=False).tolist()
    goal_per_agent = np.random.choice(goals_pool, num_agents, replace=False).tolist()

    start_counts = Counter(start_per_agent)
    goal_counts = Counter(goal_per_agent)

    starts_idx = []
    starts_agent_count = []
    goals_idx = []
    goals_agent_count = []

    for start_idx, count in start_counts.items():
        starts_idx.append(start_idx)
        starts_agent_count.append(count)

    for goal_idx, count in goal_counts.items():
        goals_idx.append(goal_idx)
        goals_agent_count.append(count)

    solver_cls = SOLVER_REGISTRY[solver_config["solver_name"]]
    return solver_cls(gaussian_prm, instance_config["agent_radius"], 
              starts_agent_count=starts_agent_count, 
              goals_agent_count=goals_agent_count,
              starts_idx=starts_idx, goals_idx=goals_idx,
              num_agents=num_agents, time_limit=180)
