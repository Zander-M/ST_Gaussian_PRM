"""
    Random Prioritized Planning
"""
import time

import numpy as np

class RandomPriority:
    """
        Random total priority ordering.
    """
    def __init__(self, gaussian_prm, instances, planner, time_limit=180):
        self.gaussian_prm = gaussian_prm
        self.instances = instances
        self.order = np.arange(len(self.instances))
        self.planner = planner
        self.flow_constraints = None
        self.time_limit = time_limit

    def plan(self):
        """
            Sequentially plan for each instance.
        """
        solution_found = [False for _ in self.instances]
        start_time = time.time()

        while time.time() - start_time < self.time_limit:
            paths = []
            if np.all(solution_found):
                print(f"solution found, solution time: {time.time() - start_time}.")
                return {
                    "success": True,
                    "paths": paths
                }

            order = np.random.permutation(len(self.instances))
            constraint_dicts = {
                "capacity_dicts": [],
                "obstacle_goal_dicts": [],
                "flow_dicts": [],
                "max_timestep": 0
            }
            for idx in order:
                print(f"Planning for swarm {idx}")
                solution = self.instances[idx].solve(**constraint_dicts)
                if solution["success"]:
                    constraint_dicts["capacity_dicts"].append(solution["capacity_dict"])
                    constraint_dicts["obstacle_goal_dicts"].append(solution["goal_state_dict"])
                    constraint_dicts["flow_dicts"].append(solution["flow_dict"])
                    constraint_dicts["max_timestep"] = max(constraint_dicts["max_timestep"], solution["timestep"])
                    # store solution paths
                    paths.append(solution["paths"])
                else:
                    print("No solution found. Randomizing order.")
                    break
        print("Timelimit Exceeded.") 
        return {"success": False}
                
