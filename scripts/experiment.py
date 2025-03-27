"""
    Compare Performance of algorithm on different algorithms 
"""

import argparse
import csv
import time
import matplotlib.pyplot as plt

# Solvers 

from swarm_prm.solvers.macro.teg import TEG
from swarm_prm.solvers.macro.drrt import DRRT
from swarm_prm.solvers.macro.lp import LP

def experiment(args):
    """
        Compare performance of algorithms on different instances
        Metric: 
        Solution Time: time to find a solution
        Solution Length: average solution length
    """
    pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--algorithm"
    )

    parser.add_argument(
        "--map_type"
    )
    parser.add_argument(
        "--num_agent",
        type=int
    )

    args = parser.parse_args()
