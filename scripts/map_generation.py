"""
    Map Generation
"""

# Imports
import argparse
import csv
import os
import pickle
import time

import matplotlib.pyplot as plt
import numpy as np

from swarm_prm.utils.gaussian_prm import GaussianPRM
from swarm_prm.envs.obstacle_map import ObstacleMap, Obstacle 

MAP_PATH = "../maps"

plt.ioff()

# Curated Test Examples
maps = {
        "empty":{
                "obstacle_map" : ObstacleMap(100, 100), 
                "obstacles" : [], 
                },
        "corridor":{
                "obstacle_map" : ObstacleMap(100, 100), 
                "obstacles" : [
                                Obstacle(None, "POLYGON", [(30, 0), (30, 40), (70, 40), (70, 0)]),
                                Obstacle(None, "POLYGON", [(30, 100), (30, 60), (70, 60), (70, 100)])
                            ], 
                },
        "corridor_narrow":{
                "obstacle_map" : ObstacleMap(100, 100), 
                "obstacles" : [
                                Obstacle(None, "POLYGON", [(30, 0), (30, 45), (70, 45), (70, 0)]),
                                Obstacle(None, "POLYGON", [(30, 100), (30, 55), (70, 55), (70, 100)])
                            ], 
                },
        "corridor_exchange":{
                "obstacle_map" : ObstacleMap(100, 100), 
                "obstacles" : [
                                Obstacle(None, "POLYGON", [(30, 0), (30, 40), (70, 40), (70, 0)]),
                                Obstacle(None, "POLYGON", [(30, 100), (30, 60), (70, 60), (70, 100)])
                            ], 
                },
        "obstacle":{
                "obstacle_map" : ObstacleMap(200, 160), 
                "obstacles" : [
                                Obstacle(None, "POLYGON", [(50, 0), (60, 75), (75, 75), (90, 40), (90, 0)]),
                                Obstacle(None, "POLYGON", [(50, 130), (75, 127), (80, 100), (55, 103)]),
                                Obstacle(None, "POLYGON", [(100, 150), (140, 150), (140, 125), (110, 125)]),
                                Obstacle(None, "POLYGON", [(145, 25), (125, 50), (135, 100), (150, 100), (160, 75), (150, 25)])
                            ],
                },
        "cross":{
                "obstacle_map" : ObstacleMap(100, 100), 
                "obstacles" : [
                                Obstacle(None, "POLYGON", [(0, 0), (30, 00), (30, 30), (0, 30)]),
                                Obstacle(None, "POLYGON", [(70, 0), (100, 00), (100, 30), (70, 30)]),
                                Obstacle(None, "POLYGON", [(0, 100), (0, 70), (30, 70), (30, 100)]),
                                Obstacle(None, "POLYGON", [(70, 70), (100, 70), (100, 100), (70, 100)]),
                            ], 
                },
        "swarm": {
            "obstacle_map" : ObstacleMap(200, 160),
            "obstacles" : [
                Obstacle(None, "POLYGON", [(50, 160), (50, 80), (70, 80), (70, 160)]),
                Obstacle(None, "POLYGON", [(70, 160), (70, 120), (150, 120), (150, 160)]),
                Obstacle(None, "POLYGON", [(50, 40), (50, 0), (130, 0), (130, 40)]),
                Obstacle(None, "POLYGON", [(130, 80), (130, 0), (150, 0), (150, 80)]),
                Obstacle(None, "POLYGON", [(90, 100), (90, 80), (110, 80), (115, 100)]),
                Obstacle(None, "POLYGON", [(90, 80), (85, 60), (110, 60), (110, 80)]),
            ]
        },
        "multiagent": {
            "obstacle_map": ObstacleMap(200, 200),
            "obstacles": [], 
        }
        }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate different types of maps with specified sample sizes.")
    
    parser.add_argument(
        "--map_type",
        choices=["empty", "corridor", "obstacle", "cross", "corridor_exchange", "corridor_narrow", "swarm"],
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
    time_logs = []
    for map_type in args.map_type:
        for num_sample in args.num_samples:
            obstacle_map = maps[map_type]["obstacle_map"]
            for obs in maps[map_type]["obstacles"]:
                obstacle_map.add_obstacle(obs)

            start_time = time.time()
            gaussian_prm = GaussianPRM(obstacle_map, num_sample)
            map_fname = "{}_{}".format(map_type, num_sample)
            path = os.path.join(args.map_path, map_fname)
            gaussian_prm.roadmap_construction()
            construction_time = time.time() - start_time
            time_logs.append([map_type, num_sample, construction_time]) 

            # save map  
            fig, _ =gaussian_prm.visualize_roadmap()
            plt.savefig(f"{path}_roadmap.png", )
            plt.close()
            fig, _ =gaussian_prm.visualize_g_nodes()
            plt.savefig(f"{path}_gaussian_nodes.png", )
            plt.close()
            with open(f"{path}.pkl", "wb") as f:
                pickle.dump(gaussian_prm, f)
    time_log_fname = os.path.join(args.map_path, "time_log.csv")
    with open(time_log_fname, "w") as f:
        writer = csv.writer(f)
        writer.writerow(["map", "num_samples", "time"])
        for time_log in time_logs:
            writer.writerow(time_log)


