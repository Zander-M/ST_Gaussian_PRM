"""
    Map Generation
"""

# Imports
import argparse
import os
import pickle

import matplotlib.pyplot as plt
import numpy as np

from swarm_prm.utils.gaussian_prm import GaussianPRM
from swarm_prm.utils.gaussian_utils import GaussianGraphNode
from swarm_prm.envs.roadmap import Roadmap, Obstacle 
from swarm_prm.envs.instance import Instance

MAP_PATH = "../maps"

# Curated Test Examples
maps = {
        "empty":{
                "roadmap" : Roadmap(100, 100), 
                "obstacles" : [], 
                "starts" : np.array([[10, 10], [10, 90]]), 
                "goals" : np.array([[90, 90], [90, 10]])             
                },
        "corridor":{
                "roadmap" : Roadmap(100, 100), 
                "obstacles" : [
                                Obstacle(None, "POLYGON", [(30, 0), (30, 40), (70, 40), (70, 0)]),
                                Obstacle(None, "POLYGON", [(30, 100), (30, 60), (70, 60), (70, 100)])
                            ], 
                "starts" : np.array([[10, 10], [10, 90]]), 
                "goals" : np.array([[90, 90], [90, 10]])             
                },
        "corridor_exchange":{
                "roadmap" : Roadmap(100, 100), 
                "obstacles" : [
                                Obstacle(None, "POLYGON", [(30, 0), (30, 40), (70, 40), (70, 0)]),
                                Obstacle(None, "POLYGON", [(30, 100), (30, 60), (70, 60), (70, 100)])
                            ], 
                "starts" : np.array([[10, 10], [90, 10]]), 
                "goals" : np.array([[90, 90], [10, 90]])             
                },
        "obstacle":{
                "roadmap" : Roadmap(200, 160), 
                "obstacles" : [
                                Obstacle(None, "POLYGON", [(50, 0), (60, 75), (75, 75), (90, 40), (90, 0)]),
                                Obstacle(None, "POLYGON", [(50, 130), (75, 127), (80, 100), (55, 103)]),
                                Obstacle(None, "POLYGON", [(100, 150), (140, 150), (140, 125), (110, 125)]),
                                Obstacle(None, "POLYGON", [(145, 25), (125, 50), (135, 100), (150, 100), (160, 75), (150, 25)])
                            ],

                "starts" : np.array([[25, 25], [25, 125]]), 
                "goals" : np.array([[175, 125], [175, 50]])
                },
        "cross":{
                "roadmap" : Roadmap(100, 100), 
                "obstacles" : [
                                Obstacle(None, "POLYGON", [(0, 0), (30, 00), (30, 30), (0, 30)]),
                                Obstacle(None, "POLYGON", [(70, 0), (100, 00), (100, 30), (70, 30)]),
                                Obstacle(None, "POLYGON", [(0, 100), (0, 70), (30, 70), (30, 100)]),
                                Obstacle(None, "POLYGON", [(70, 70), (100, 70), (100, 100), (70, 100)]),
                            ], 
                "starts" : np.array([[20, 50], [50, 20]]),
                "goals" : np.array([[80, 50], [50, 80]])
                },
        "multiagent": {
            "roadmap": Roadmap(200, 200),
            "obstacles": [], 
            "starts" : np.array([[70, 50], [70, 150], [130, 50], [130, 150]]), 
            "goals" : np.array([[25, 50], [25, 150], [175, 50], [175, 150]])
        }
        }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate different types of maps with specified sample sizes.")
    
    parser.add_argument(
        "--map_type",
        choices=["empty", "corridor", "obstacle", "cross", "corridor_exchange"],
        nargs="+",
        default="empty",
        help="Type of the map to generate."
    )
    
    parser.add_argument(
        "--num_samples",
        nargs="+",
        type=int,
        default=100, 
        help="Number of samples to use for the map."
    )

    parser.add_argument(
        "--map_path",
        default=MAP_PATH,
        help="Path to map file"
    )

    args = parser.parse_args()
    for map_type in args.map_type:
        for num_sample in args.num_samples:
            roadmap = maps[map_type]["roadmap"]
            for obs in maps[map_type]["obstacles"]:
                roadmap.add_obstacle(obs)
            g_starts = [GaussianGraphNode(start, None, "UNIFORM", radius=10) for start in maps[map_type]["starts"]]
            g_goals = [GaussianGraphNode(goal, None, "UNIFORM", radius=10) for goal in maps[map_type]["goals"]]
            instance = Instance(roadmap, g_starts, g_goals)
            gaussian_prm = GaussianPRM(instance, num_sample)
            map_fname = "{}_{}".format(map_type, num_sample)
            path = os.path.join(args.map_path, map_fname)
            gaussian_prm.roadmap_construction()

            # save map  
            fig, _ =gaussian_prm.visualize_roadmap(path)
            plt.savefig(f"{path}_roadmap.png", )
            plt.close()
            fig, _ =gaussian_prm.visualize_g_nodes(path)
            plt.savefig(f"{path}_gaussian_nodes.png", )
            plt.close()
            with open(f"{path}.pkl", "wb") as f:
                pickle.dump(gaussian_prm, f)

