"""
    Load config yaml, and return instance_config, solver_config and configs
"""

import yaml
from shapely.geometry import Polygon

def load_config(yaml_path):
    with open(yaml_path, 'r') as f:
        data = yaml.safe_load(f)

    # Extract top-level config values
    instance_name = data['experiment']['instance_name']
    solver_name = data['experiment']['solver_name']
    time_limit = data['experiment']['time_limit']
    capacity_constraint = data['experiment']['capacity_constraint']

    # Process instances
    instances = {}
    for key, inst in data['instances'].items():
        instance = inst.copy()
        instance['start_regions'] = [Polygon(coords) for coords in inst['start_regions']]
        instance['goal_regions'] = [Polygon(coords) for coords in inst['goal_regions']]
        instances[key] = instance

    # Build final configs
    instance_config = instances[instance_name]
    solver_config = {"solver_name": solver_name}
    experiment_config = {
        "time_limit": time_limit,
        "capacity_constraint": capacity_constraint
    }
    apf_config = data.get("apf_config", {})

    return instance_config, solver_config, experiment_config, apf_config

# Example usage:
if __name__ == "__main__":
    instance_config, solver_config, configs, apf_config = load_config("experiment_config.yaml")
    print("Instance Config:", instance_config)
    print("Solver Config:", solver_config)
    print("General Configs:", configs)
    print("APF Config:", apf_config)

