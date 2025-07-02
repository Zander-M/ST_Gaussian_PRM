"""
    Random Prioritized Planning
"""
import time

import numpy as np

class RandomPriority:
    """
        Random total priority ordering.
    """
    def __init__(self, instances, time_limit=180):

        self.instances = instances
        self.time_limit = time_limit

    def solve(self):
        """
            Sequentially plan for each instance.
        """
        solution_found = [False for _ in self.instances]
        start_time = time.time()
        order = np.random.permutation(len(self.instances))
        paths = []
        constraint_dicts = {
                "occupancy_sets": [],
                "obstacle_goal_dicts": [],
                "flow_dicts": [],
                "max_timestep": 0
            }
        while time.time() - start_time < self.time_limit:
            if np.all(solution_found):
                print(f"solution found, solution time: {time.time() - start_time}.")
                solution_length = max([len(path[0]) for path in paths])
                padded_paths = []
                for swarm_idx in order:
                    swarm_paths = paths[swarm_idx]
                    for path in swarm_paths:
                        padded_path = path + [path[-1]]*(solution_length-len(path))
                        padded_paths.append(padded_path)

                return {
                    "success": True,
                    "paths": padded_paths
                }


            for idx in order:
                print(f"Planning for swarm {idx}")
                solution = self.instances[idx].solve(**constraint_dicts)
                if solution["success"]:
                    constraint_dicts["occupancy_sets"].append(solution["occupancy_set"])
                    constraint_dicts["obstacle_goal_dicts"].append(solution["goal_state_dict"])
                    constraint_dicts["flow_dicts"].append(solution["flow_dict"])
                    constraint_dicts["max_timestep"] = max(constraint_dicts["max_timestep"], solution["timestep"])
                    # store solution paths
                    paths.append(solution["paths"])
                    solution_found[idx] = True
                else:
                    print("No solution found. Randomizing order.")
                    solution_found = [False for _ in self.instances]
                    order = np.random.permutation(len(self.instances))
                    paths = []
                    constraint_dicts = {
                                    "occupancy_sets": [],
                                    "obstacle_goal_dicts": [],
                                    "flow_dicts": [],
                                    "max_timestep": 0
                                }
                    break
        print("Timelimit Exceeded.") 
        return {"success": False}
                
