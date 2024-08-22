"""
    Generate map and instance. Problem instance includes agent starts and goals.
    All the obstacles on the map are static.
"""


import numpy as np
import os
import yaml

from swarm_prm.envs.roadmap import *
from swarm_prm.solvers.swarm_prm.macro.gaussian_prm import GaussianNode

# seeding the environment
np.random.seed(0)

if __name__ == "__main__":
    pass
    # Generate Maps
    # map_generator = MapGenerator(100, 100)
    # map_generator.to_yaml()
    
    # Generate Instance
    # instance_generator = InstanceGenerator()
    # instance_generator.to_yaml()