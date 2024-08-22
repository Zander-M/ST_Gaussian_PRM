"""
    Load map and test instances
"""

from matplotlib import pyplot as plt
import yaml

from swarm_prm.envs.roadmap import *


  
        


if __name__ == "__main__":
    fname = "../../data/envs/map_3.yaml"
    map_info = RoadmapLoader(fname)
    map_info.visualize("test_map")