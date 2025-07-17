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

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from shapely.geometry import Polygon

# Solvers 
from st_gaussian_prm.solvers.macro import SOLVER_REGISTRY

# Scripts

markers = {
    "TEGSolver": "*",
    "LPSolver": "o"
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

    starts_idx = np.random.choice(starts_idx_set, instance_config["num_starts"], replace=False)
    starts_idx = [starts_idx] if instance_config["num_starts"] == 1 else starts_idx.tolist() # type:ignore
    start_capacity = np.sum([gaussian_prm.gaussian_nodes[idx].get_capacity(instance_config["agent_radius"]) 
                             for idx in starts_idx])

    goals_idx = np.random.choice(goals_idx_set, instance_config["num_goals"], replace=False)
    goals_idx = [goals_idx] if instance_config["num_goals"] == 1 else goals_idx.tolist()     # type:ignore
    goal_capacity = np.sum([gaussian_prm.gaussian_nodes[idx].get_capacity(instance_config["agent_radius"]) 
                             for idx in goals_idx])
    
    # Decide num_agents based on the maximum number of agents that can fit in start and goal location
    num_agents = np.floor(instance_config["capacity_percentage"]*np.min((start_capacity, goal_capacity))).astype(np.int32)

    # Create weight pool
    starts_pool = []
    goals_pool = []

    for start_idx in starts_idx:
        starts_pool += [start_idx] * gaussian_prm.gaussian_nodes[start_idx].get_capacity(instance_config["agent_radius"])

    for goal_idx in goals_idx:
        goals_pool += [goal_idx] * gaussian_prm.gaussian_nodes[goal_idx].get_capacity(instance_config["agent_radius"])

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
    return starts_idx, starts_agent_count, goals_idx, goals_agent_count, num_agents

def plot_results(result_path, show_fig):
    """
    Plot per-map experiment results: Runtime and Success Rate vs Capacity Percentage.
    Saves figures per map.
    """
    # Load CSV
    df = pd.read_csv(os.path.join(result_path, "results.csv"))

    # Ensure results directory exists
    os.makedirs(result_path, exist_ok=True)

    # Plot per map
    for map_type, map_group in df.groupby('map_type'):
        # --- Runtime Plot ---
        plt.figure(figsize=(10, 6))
        for solver, solver_group in map_group.groupby('solver'):
            group_sorted = solver_group.sort_values('capacity_percentage')
            plt.plot(group_sorted['capacity_percentage'], group_sorted['average_runtime'],
                     marker=markers.get(solver, 'x'), markersize=10, label=solver) # type:ignore
        plt.title(f"Average Runtime vs Capacity Percentage ({map_type})")
        plt.xlabel("Capacity Percentage")
        plt.ylabel("Average Runtime (s)")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        if show_fig:
            plt.show()
        plt.savefig(os.path.join(result_path, f"{map_type}_runtime_vs_capacity.png"))
        plt.close()

        # --- Success Rate Plot ---
        plt.figure(figsize=(10, 6))
        for solver, solver_group in map_group.groupby('solver'):
            group_sorted = solver_group.sort_values('capacity_percentage')
            plt.plot(group_sorted['capacity_percentage'], group_sorted['success_rate'],
                     marker=markers.get(solver, 'x'), markersize=10, label=solver) # type:ignore
        plt.title(f"Success Rate vs Capacity Percentage ({map_type})")
        plt.xlabel("Capacity Percentage")
        plt.ylabel("Success Rate")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        if show_fig:
            plt.show()
        plt.savefig(os.path.join(result_path, f"{map_type}_success_vs_capacity.png"))
        plt.close()

        # --- Relative Quality Plot ---
        # TODO: make sure color matches
        df_tegsolver = df[(df["map_type"] == map_type) & (df["solver"] == "TEGSolver")]

        plt.figure(figsize=(10, 6))
        group_sorted = df_tegsolver.sort_values('capacity_percentage')
        plt.plot(group_sorted['capacity_percentage'], group_sorted['relative_cost'],
                 marker="o", label="Relative Transport Cost")
        plt.plot(group_sorted['capacity_percentage'], group_sorted['relative_makespan'],
                 marker="o", label="Relative Makespan")
        plt.title(f"Relative Solution Quality vs Capacity Percentage ({map_type})")
        plt.xlabel("Capacity Percentage")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        if show_fig:
            plt.show()
        
        plt.savefig(os.path.join(result_path, f"{map_type}_relative_solution_quality_vs_capacity.png"))
        plt.close()

def run_experiment(config, result_path):
    """
        Run experiments using the same random instances with different solvers
    """
    experiment_config = config["experiment_config"]
    env_configs = config["env_configs"]
    solver_configs = config["solver_configs"]

    combinations = list(itertools.product(
        experiment_config["capacity_percentage"],
        env_configs,
    ))

    results = []

    for capacity_percentage, env_config in combinations:
        # Load Map
        map_fname = "{}_{}.pkl".format(env_config["map_type"], env_config["num_samples"])
        map_path = os.path.join("../../../maps", map_fname)
        with open(map_path, "rb") as f:
            gaussian_prm = pickle.load(f)

        print(f"Loaded map: {map_fname}")

        for trial in range(experiment_config["num_tries"]):
            print(f"Trial {trial+1}/{experiment_config['num_tries']} - Capacity: {capacity_percentage}")

            # Sample a single random instance
            instance_config_sample = {
                "gaussian_prm": gaussian_prm,
                "agent_radius": env_config["agent_radius"],
                "start_regions": env_config["start_regions"],
                "goal_regions": env_config["goal_regions"],
                "num_starts": env_config["num_starts"],
                "num_goals": env_config["num_goals"],
                "capacity_percentage": capacity_percentage,
            }

            starts_idx, starts_agent_count, goals_idx, goals_agent_count, num_agents = \
                create_random_planning_instance(instance_config_sample)

            swarmprm_result = None

            for solver_config in solver_configs:
                solver_name = solver_config["solver"]
                print(f"  Running solver: {solver_name}")

                # Compose instance config
                instance_config = {
                    "gaussian_prm": gaussian_prm,
                    "agent_radius": env_config["agent_radius"],
                    "time_limit": experiment_config["time_limit"],
                    "capacity_percentage": capacity_percentage,
                    "starts_idx": starts_idx,
                    "starts_agent_count": starts_agent_count,
                    "goals_idx": goals_idx,
                    "goals_agent_count": goals_agent_count,
                    "num_agents": num_agents,
                    "solver": solver_name,
                }

                # Run solver
                solver_cls = SOLVER_REGISTRY[solver_name]
                solver = solver_cls(
                    gaussian_prm,
                    instance_config["agent_radius"],
                    starts_agent_count=starts_agent_count,
                    goals_agent_count=goals_agent_count,
                    starts_idx=starts_idx,
                    goals_idx=goals_idx,
                    num_agents=num_agents,
                    time_limit=instance_config["time_limit"],
                    **solver_config["solver_config"]
                )

                start_time = time.time()
                solution = solver.solve()
                runtime = time.time() - start_time

                cost = solution.get("cost", 0)
                timestep = solution.get("timestep", 0)

                if solver_name == "LPSolver" and solution["success"]:
                    swarmprm_result = {
                        "cost": cost,
                        "timestep": timestep
                    }

                relative_cost = cost / swarmprm_result["cost"] if (solution["success"] and swarmprm_result) else None
                relative_makespan = timestep / swarmprm_result["timestep"] if (solution["success"] and swarmprm_result and swarmprm_result["timestep"] > 0) else None

                result = {
                    "solver": solver_name,
                    "map_type": env_config["map_type"],
                    "capacity_percentage": capacity_percentage,
                    "trial": trial,
                    "success": solution["success"],
                    "valid": solution["valid"],
                    "runtime": runtime if solution["success"] else 0,
                    "cost": cost,
                    "timestep": timestep,
                    "relative_cost": relative_cost,
                    "relative_makespan": relative_makespan,
                }
                results.append(result)

    # Save raw results
    df = pd.DataFrame(results)
    df.to_csv(os.path.join(result_path, "raw_results.csv"), index=False)

    # Aggregate summary
    df_summary = df.groupby(["solver", "map_type", "capacity_percentage"]).agg({
        "valid": "mean",
        "runtime": lambda x: np.mean([v for v in x if v > 0]) if any(x) else 0,
        "relative_cost": lambda x: np.nanmean(x),
        "relative_makespan": lambda x: np.nanmean(x),
    }).reset_index()

    df_summary.rename(columns={
        "valid": "success_rate",
        "runtime": "average_runtime"
    }, inplace=True)

    csv_path = os.path.join(result_path, "results.csv")
    df_summary.to_csv(csv_path, index=False)
    print(f"Results saved to {csv_path}")


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
    # result_path = create_result_folder(args.output_dir)
    # shutil.copy(args.config, result_path)
    # config = load_config(args.config)
    # run_experiment(config, result_path)
    result_path = "results/20250717_102710"
    plot_results(result_path, args.show_fig)
