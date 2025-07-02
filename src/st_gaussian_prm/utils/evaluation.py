"""
    Evaluate solution quality
    We assume the agents have a maximum velocity
    All transitions between states are piecewise-linear.
    Different edge lengths are normalized.
    Wait action take the same duration as the longest travel time.
"""
import time
from collections import defaultdict, Counter

import matplotlib.pyplot as plt

from st_gaussian_prm.solvers.micro import EvaluationSolver

def paths_to_macro(paths):
    """
        Convert individual agent solutions to time-indexed paths
    """
    timestep = len(paths[0]) # solution timesteps
    macro_sol = defaultdict(lambda:defaultdict(list))
    for t in range(timestep-1):
        transitions = [(p[t], p[t+1]) for p in paths]
        count = Counter(transitions)
        for (u, v), c in count.items():
            macro_sol[t][u].append((v, c))
    return macro_sol

def evaluate_path_cost(gaussian_nodes, macro_solution, timestep, num_agent,
                       starts_idx, goals_idx,
                       starts_agent_count, goals_agent_count):
    """
        Given Gassian nodes and time-indexed macro solution,
        evaluate time taken (makespan) to complete the plan
    """
    pass

def animate_path(ax, gaussian_path):
    pass