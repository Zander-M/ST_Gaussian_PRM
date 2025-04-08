"""
    Compare Performance of algorithm on different algorithms 
"""

import argparse
from datetime import datetime
import os
import csv
import pickle
import json
import itertools
import shutil
import time

# Solvers 
from swarm_prm.utils import get_agent_assignment
from swarm_prm.solvers.macro import SOLVER_REGISTRY

# Scripts
from plot_result import plot_result

def run_solver(config):
    """
        Run Experiment with solver using config
    """
    solver_cls = SOLVER_REGISTRY[config["solver"]]
    solver = solver_cls(
        config["gaussian_prm"],
        config["agent_radius"],
        starts_agent_count=config["starts_agent_count"],
        goals_agent_count=config["goals_agent_count"],
        num_agents=config["num_agents"],
        time_limit=config["time_limit"]
        )
    start_time = time.time()
    solution = solver.solve()
    runtime = time.time() - start_time
    if solution["success"]:
        return {
            "success": True,
            "solver": config["solver"],
            "runtime": runtime,
            "solution_timestep": solution["timestep"],
            "solution_cost": solution["cost"]
        }

    else:
        return {
            "success": False
        }
    
def load_config(config_path):
    """
        Load config
    """
    config = {}
    with open(config_path, "r") as f:
       config = json.load(f)


    return config
    
def create_result_folder(output_dir):
    time_tag = datetime.now().strftime('%Y%m%d_%H%M%S')
    folder_name = f"{time_tag}"
    result_path = os.path.join(output_dir, folder_name)
    os.makedirs(result_path, exist_ok=True)
    return result_path

def run_experiment(config, result_path):
    """
        Compare performance of algorithms on different instances
        Metric: 
        Solution Time: time to find a solution
        Solution Length: average solution length
    """
    combinations = list(itertools.product(
        config["solvers"],
        config["map_type"],
        config["num_samples"],
        config["agent_config"]
    ))

    # Run experiment
    results = []
    for solver, map_type, num_samples, [num_agents, agent_radius] in combinations:
        experiment_config = {
            "solver": solver,
            "map_type": map_type,
            "num_samples": num_samples,
            "num_agents": num_agents,
            "starts_agent_count": get_agent_assignment(num_agents, config["starts_weight"]),
            "goals_agent_count": get_agent_assignment(num_agents, config["goals_weight"]),

            # Following parameters are shared across experiments
            "starts_weight": config["starts_weight"],
            "goals_weight": config["goals_weight"],
            "agent_radius": agent_radius,
            "time_limit": config["time_limit"]
        }

        print(f"Running {solver}, Agents: {num_agents}, Map: {map_type}, Samples: {num_samples}")

        # Load Map
        map_fname = "{}_{}.pkl".format(map_type, num_samples)
        fname = os.path.join("../maps", map_fname)
        with open(fname, "rb") as f:
            experiment_config["gaussian_prm"] = pickle.load(f) 

        result = run_solver(experiment_config)

        # Add experiment config
        result.update({
            "solver": solver,
            "map_type": map_type,
            "num_samples": num_samples,
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
    plot_result(result_path, args.show_fig)

    
