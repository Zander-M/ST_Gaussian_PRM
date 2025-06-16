"""
    Single agent experiments
"""

import argparse
import csv
from collections import Counter
from datetime import datetime
import itertools
import json
import os
import pickle
import shutil
import time

import numpy as np
from shapely.geometry import Polygon

# Solvers 
from swarm_prm.solvers.macro import SOLVER_REGISTRY

# Scripts
# from plot_result import plot_result

def run_solver(instance_config):
    """
        Run Experiment
    """

    # Stats

    num_success = 0
    suboptimality = []
    runtimes = []

    # Repeat experiments
    for _ in range(instance_config["num_tries"]):
    # Random Instance  
        starts_idx, starts_agent_count, goals_idx, goals_agent_count =  \
        create_random_planning_instance(instance_config)

        # Solver
        solver_cls = SOLVER_REGISTRY[instance_config["solver"]]
        solver = solver_cls(
            instance_config["gaussian_prm"],
            instance_config["agent_radius"],
            starts_agent_count = starts_agent_count,
            goals_agent_count = goals_agent_count,
            starts_idx = starts_idx,
            goals_idx = goals_idx,
            num_agents=instance_config["num_agents"],
            time_limit=instance_config["time_limit"]
            )
        start_time = time.time()
        solution = solver.solve()
        runtime = time.time() - start_time
        if solution["success"]:
            num_success += 1
            suboptimality.append(None) # TODO: create suboptimality evaluation here
            runtimes.append(runtime)
        else:
            suboptimality.append(0)
            runtimes.append(0)
    return {
        "success_rate": num_success / instance_config["num_tries"],
        # "average_suboptimality": np.sum(suboptimality) / num_success,
        "average_runtime": np.sum(runtimes) / num_success
    }

def load_config(config_path):
    """
        Load config
    """
    config = {}
    with open(config_path, "r") as f:
       config = json.load(f)
    # import pprint
    # pprint.pprint(config)
    return config
    
def create_result_folder(output_dir):
    time_tag = datetime.now().strftime('%Y%m%d_%H%M%S')
    folder_name = f"{time_tag}"
    result_path = os.path.join(output_dir, folder_name)
    os.makedirs(result_path, exist_ok=True)
    return result_path

def create_random_planning_instance(instance_config):
    """
        Create random planning instance
    """
    gaussian_prm = instance_config["gaussian_prm"]
    start_regions = []
    goal_regions = []
    for vertices in instance_config["start_regions"]:
        start_regions.append(Polygon(vertices))
    for vertices in instance_config["goal_regions"]:
        goal_regions.append(Polygon(vertices))

    starts_idx_set = gaussian_prm.get_node_index(start_regions)
    goals_idx_set = gaussian_prm.get_node_index(goal_regions)

    # Sample starts and goals.
    # Resample if capacity is smaller than num_agents.

    starts_idx = np.random.choice(starts_idx_set, instance_config["num_starts"], replace=False)
    starts_idx = [starts_idx] if instance_config["num_starts"] == 1 else starts_idx.tolist() # type:ignore
    start_capacity = np.sum([gaussian_prm.gaussian_nodes[idx].get_capacity(instance_config["agent_radius"]) 
                             for idx in starts_idx])
    while start_capacity < instance_config["num_agents"]:
        starts_idx = np.random.choice(starts_idx_set, instance_config["num_starts"], replace=False)
        starts_idx = [starts_idx] if instance_config["num_starts"] == 1 else starts_idx.tolist() # type:ignore
        start_capacity = np.sum([gaussian_prm.gaussian_nodes[idx].get_capacity(instance_config["agent_radius"]) 
                                 for idx in starts_idx])

    goals_idx = np.random.choice(goals_idx_set, instance_config["num_goals"], replace=False)
    goals_idx = [goals_idx] if instance_config["num_goals"] == 1 else goals_idx.tolist()     # type:ignore
    goal_capacity = np.sum([gaussian_prm.gaussian_nodes[idx].get_capacity(instance_config["agent_radius"]) 
                             for idx in goals_idx])
    while goal_capacity < instance_config["num_agents"]:
        goals_idx = np.random.choice(starts_idx_set, instance_config["num_starts"], replace=False)
        goals_idx = [starts_idx] if instance_config["num_starts"] == 1 else starts_idx.tolist() # type:ignore
        goal_capacity = np.sum([gaussian_prm.gaussian_nodes[idx].get_capacity(instance_config["agent_radius"]) 
                                 for idx in goals_idx])

    # Create weight pool
    starts_pool = []
    goals_pool = []

    for start_idx in starts_idx:
        starts_pool += [start_idx] * gaussian_prm.gaussian_nodes[start_idx].get_capacity(instance_config["agent_radius"])

    for goal_idx in goals_idx:
        goals_pool += [goal_idx] * gaussian_prm.gaussian_nodes[goal_idx].get_capacity(instance_config["agent_radius"])

    start_per_agent = np.random.choice(starts_pool, instance_config["num_agents"], replace=False).tolist()
    goal_per_agent = np.random.choice(goals_pool, instance_config["num_agents"], replace=False).tolist()

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
    return starts_idx, starts_agent_count, goals_idx, goals_agent_count


def run_experiment(config, result_path):
    """
        Compare performance of algorithms on different instances
        Metric: 
        Solution Time: time to find a solution
        Solution Length: average solution length
    """
    experiment_config = config["experiment_config"]
    env_configs = config["env_configs"]
    solver_configs = config["solver_configs"]

    combinations = list(itertools.product(
        experiment_config["num_agents"],
        env_configs,
        solver_configs,
    ))

    # Run experiment
    results = []
    for num_agents, env_config, solver_config in combinations:
        instance_config = {
            "time_limit": experiment_config["time_limit"],
            "num_tries": experiment_config["num_tries"],
            "num_agents": num_agents,
        }
        instance_config.update(env_config)
        instance_config.update(solver_config)

        print("Running {}, Agents: {}, Map: {}".format(
            solver_config["solver"],
            num_agents, 
            env_config["map_type"],
            ))
        # Load Map
        map_fname = "{}_{}.pkl".format(env_config["map_type"], env_config["num_samples"])

        fname = os.path.join("../../../maps", map_fname)
        with open(fname, "rb") as f:
            instance_config["gaussian_prm"] = pickle.load(f) 
        result = run_solver(instance_config)

        # Add experiment config
        result.update({
            "solver": solver_config["solver"],
            "map_type": env_config["map_type"],
            "num_agents": num_agents,
        })
        results.append(result)
    
    # Store Experiment Results
    keys = sorted(results[0].keys())
    csv_path = os.path.join(result_path, "results.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(results)
    print(f"Result written to {csv_path}")
    return csv_path

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Gaussian PRM makespan experiment.")
    parser.add_argument(
        "--config",
        type=str,
        default="config.json",
        help="Path to config JSON file"
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default="results",
        help="Directory for experiment results"
    )

    parser.add_argument(
        "--show_fig",
        type=bool,
        default=False,
        help="Display Performance Charts"
    )
    args = parser.parse_args()

    ## Copy config file to result folder
    result_path = create_result_folder(args.output_dir)
    shutil.copy(args.config, result_path)
    config = load_config(args.config)
    result_path = run_experiment(config, result_path)
    # plot_result(result_path, args.show_fig)

    
