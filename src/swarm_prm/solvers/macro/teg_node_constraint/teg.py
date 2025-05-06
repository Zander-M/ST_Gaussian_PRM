"""
    Max flow on Time Expanded Graph. Result from previous timestep is reused
    to speed up the algorithm.
"""

import time

from collections import defaultdict, deque
from swarm_prm.solvers.macro import TEGSolver, register_solver
from swarm_prm.solvers.macro.teg_node_constraint.max_flow import MaxFlow

# distinguish nodes

IN_NODE = 0
OUT_NODE = 1 

@register_solver("TEGNodeConstraintSolver")
class TEGNodeConstraintSolver(TEGSolver):
    def build_teg(self, timestep):
        """
            Build TEG based on timestep. If a node is used by one swarm,
            forbid other swarms to use this node.
        """
        teg = defaultdict(dict)
        super_source = ("SS", None, OUT_NODE)
        super_sink = ("SG", None, IN_NODE)

        # Adding super source and super goal to the graph
        for i, start_idx in enumerate(self.gaussian_prm.starts_idx):
            teg[super_source][(start_idx, 0, IN_NODE)] = self.starts_agent_count[i]
            teg[(start_idx, 0, IN_NODE)][(start_idx, 0, OUT_NODE)] = self.node_capacity[start_idx]

        for i, goal_idx in enumerate(self.gaussian_prm.goals_idx):
            teg[(goal_idx, timestep, OUT_NODE)][super_sink] = self.goals_agent_count[i]

        # adding graph edges
        for t in range(timestep):
            for u in self.roadmap:

                # Edge for capacity constraints. We have capacities occupied by previous agents. Capacity Constraint
                teg[(u, t+1, IN_NODE)][(u, t+1, OUT_NODE)] = self.node_capacity[u]  
                for capacity_dict in self.capacity_dicts:
                    # if node is used, forbid the swarm to use this node
                    if (u, t+1) in capacity_dict:
                        teg[(u, t+1, IN_NODE)][(u, t+1, OUT_NODE)] = 0
                        break

                # Edge between states
                for v in self.roadmap[u]:
                    # check if inverse edge between different nodes exists. Edge Constraint
                    if (u != v):
                        edge_exists = False
                        for flow_dict in self.flow_dicts:
                            if (u, t+1) in flow_dict[(v, t)]: # type: ignore
                                # print("edge_exist!") # TESTT
                                edge_exists = True
                                break
                        if not edge_exists:
                            teg[(u, t, OUT_NODE)][(v, t+1, IN_NODE)] = float("inf")
                    else:
                        teg[(u, t, OUT_NODE)][(v, t+1, IN_NODE)] = float("inf")

        return super_source, super_sink, teg 

