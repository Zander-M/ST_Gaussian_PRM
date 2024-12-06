"""
    CBS with capacity constraint implementation using NetworkX

    We extend the notion of from MAPF to Swarm coordination:
    
    * EdgeConflict: agents can only travel in one direction, and only one group
    of swarm can use one edge at a timestep.

    * CapacityConflict: at any timesteps, the number of agents inside each node
    cannot exceed the node capacity limit.
"""

from collections import defaultdict
from queue import PriorityQueue

from matplotlib import pyplot as plt
import networkx as nx

from swarm_prm.solvers.macro.teg.gaussian_prm import GaussianPRM

class CBS_NX:
    def __init__(self, gaussian_prm:GaussianPRM, agent_radius, target_flow, max_timestep=100) -> None:
        self.gaussian_prm = gaussian_prm
        self.agent_radius = agent_radius
        self.target_flow = target_flow
        self.max_timestep = max_timestep
        self.roadmap_graph = self.build_roadmap_graph()
        self.nodes = [i for i in range(len(self.gaussian_prm.samples))]

    
    def build_roadmap_graph(self):
        """
            Build roadmap based on Gaussian PRM
        """
        pass

    def find_solution(self):
        """
            Find Solution using CBS
        """
        pass

    def get_conflicts(self):
        """
            Get conflicts of the current solution
        """
        pass


    def construct_constraints(self, ct, timestep):
        """
            Construct constraints at timestep
        """

    def plan(self, ct):
        """
            Compute trajectory per agent
        """