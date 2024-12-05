"""
    Prioritized Planning for Gaussian PRM
"""
from collections import defaultdict

from swarm_prm.solvers.macro.swarm_teg.gaussian_prm import GaussianPRM
from swarm_prm.solvers.macro.swarm_teg.abstract_graph import AbstractGraph 
from swarm_prm.solvers.macro.swarm_teg.stastar import STAStar

class PrioritizedPlanningMaxFlow:
    """
        Prioritized Planning for finding earliest timestep to reach target flow
    """
    def __init__(self, gaussian_prm:GaussianPRM, agent_radius, target_flow) -> None:
        self.gaussian_prm = gaussian_prm
        self.agent_radius = agent_radius
        self.target_flow = target_flow
        self.graph= AbstractGraph(gaussian_prm, agent_radius)
        self.nodes = [i for i in range(len(self.gaussian_prm.samples))]

    def solve(self):
        """
            Solve for agent trajectories
        """
        curr_flow = 0
        constraints = defaultdict(dict)
        paths = []

        # compute number of agents at starts
        starts_flow = []
        for i in range(len(self.gaussian_prm.starts_idx)):
           starts_flow.append(int(self.target_flow * self.gaussian_prm.starts_weight[i])) 
        
        while curr_flow < self.target_flow:

            # Choose start nodes with positive capacity
            start = -1
            for i, source_flow in enumerate(starts_flow):
                if source_flow > 0:
                    start = i
                
            path = STAStar(self.nodes, self.graph, constraints).search(start)
            flow = self.graph.get_path_flow(path)
            flow = min(starts_flow[start], flow)

            # update flow dict

            self.graph.update_flow(path, flow)
            constraints = self.update_constraints(constraints, path)
            paths.append((path, flow))
            curr_flow += flow
            starts_flow[start] -= flow

        return paths

    def update_constraints(self, constraints, path):
        """
            Convert Path to constraints

            Edge Constraint: Agents move in the same direction across the same edge
            Capacity Constraint: Cannot move into node with no available capacity

            Constraints are negative actions. The agents cannot take the actions
            listed in the constraints

            Constriant format:
            {t:[(v1, v2), ...]...}
        """
        # Edge Constraints
        for t, (u, v) in enumerate(zip(path[:-1], path[1:])):
            if (v, u) not in constraints[t]:
                constraints[t][(v, u)] = ""

        # Capacity Constraints
        # Forbid agents to enter node that is full
        for t, node in enumerate(path):
            if self.graph.get_node_capacity(node, t) == 0:
                neighbors = self.graph.get_neighbors(node)
                for neighbor in neighbors:
                    constraints[t-1][(neighbor, node)] = ""

        return constraints


