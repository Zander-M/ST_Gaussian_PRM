"""
    Prioritized Planning for Gaussian PRM
"""

import numpy as np
import matplotlib.pyplot as plt

import heapq

from swarm_prm.solvers.macro.swarm_prm.gaussian_prm import GaussianPRM


from collections import defaultdict
from matplotlib import pyplot as plt
import networkx as nx
from swarm_prm.solvers.macro.swarm_prm.gaussian_prm import GaussianPRM

class AbstractGraph:
    """
        Abstract Graph for A star search
    """
    def __init__(self, gaussian_prm, agent_radius):
        self.gaussian_prm = gaussian_prm
        self.grid_size = self.gaussian_prm.hex_radius
        self.agent_radius = agent_radius
        self.nodes = self.gaussian_prm.samples 
        self.nodes_idx = [i for i in range(len(self.nodes))]

        self.nodes = self.gaussian_prm.samples 
        self.graph = self.build_graph()
        self.starts_idx = self.gaussian_prm.start_idx
        self.goals_idx = self.gaussian_prm.start_idx
        self.heuristic = self.compute_heuristic()

    def build_graph(self, method="MIN_CAPACITY"):
        """
            Build Graph with capacity constraints
        """
        graph = defaultdict(list)

        if method == "MIN_CAPACITY":
            for edge in self.gaussian_prm.roadmap:
                u, v = edge
                capacity = min(self.gaussian_prm.gaussian_nodes[u].get_capacity(self.agent_radius),
                               self.gaussian_prm.gaussian_nodes[v].get_capacity(self.agent_radius))
                graph[u].append((v, capacity))
                graph[v].append((u, capacity))

        elif method == "VERTEX_CAPACITY":
            assert False, "Unimplemented roadmap graph construction method."
        return graph
    
    def get_starts(self):
        """
            Return start nodes
        """
        return self.starts_idx

    def update_capacity(self, constraints):
        """
            Update node constriants based on constriants
        """

    def get_path_capacity(self, path):
        """
            Get path flow based on map
        """

    
    def get_neighbors(self, node_idx):
        """
            Get neighbors of a node
        """
        return [node[0] for node in self.graph[node_idx]]
        
    def compute_heuristic(self):
        """
            Eucledian Distance normalized with grid size
        """
        heuristic = {}
        for node_idx in self.nodes_idx:
            heuristic[node_idx] = min(
                [np.linalg.norm(self.nodes[goal_idx], self.nodes[node_idx]) \
                for goal_idx in self.gaussian_prm.goals_idx])\
                / self.grid_size
        return heuristic

    def get_heuristic(self, node_idx):
        """
            Get heuristic of the next node
            Use minimum Eucledian distance normalized by grid size to the closest goals 
        """
        return self.heuristic[node_idx]

class ST_AStar:
    """
        Single agent spatio-temporal A star search on PRM for max flow w.r.t.
        the constriants

        We assume agents are always planning from super source to super sink

    """

    def __init__(self, nodes, graph:AbstractGraph, constraints):
        """
            nodes:
                Node indexes
            
            graph: 
                Abstract Graph

            constraints:
                Constriants indexed by timestep.
                Format:
                {timestep:[(v1, v2), ...], ...}
        """
        self.nodes = nodes
        self.graph = graph
        self.constraints = constraints
        self.ss = len(self.graph.nodes)
        self.sg = len(self.graph.nodes) + 1

    def a_star(self):
        """
            A Star search
        """
        open_heap = []

        # Add starts
        actions = []
        for start_idx in self.graph.starts_idx:
            actions.append([None, start_idx])

        constrained_actions = self.apply_constriants(actions, 0)
        for _, start_idx in constrained_actions:
            curr_state_dict = {
                "t": 0,
                "parent": None,
                "node_idx": start_idx
            }
            f_value = self.graph.get_heuristic(start_idx)
            heapq.heappush(open_heap, (f_value, curr_state_dict))

        while open_heap:
            # pop from open list, take one step and add new node to open list
            _, curr_state_dict =heapq.heappop(open_heap)
            curr_node_idx = curr_state_dict["node_idx"]

            # if reaching one of the goals, 
            if self.goal_test(curr_node_idx):
                break

            # all neighboring nodes + wait
            next_nodes = self.graph.get_neighbors(curr_node_idx) + [curr_node_idx]
            actions = [[curr_node_idx, next_node] for next_node in next_nodes]
            constrained_actions = self.apply_constriants(actions, timestep)
            for action in constrained_actions:
                state_dict = {
                    "t" : curr_state_dict["t"] + 1,
                    "parent" : curr_state_dict,
                    "node_idx" : action[1]
                }
                f_value = curr_state_dict["t"] + self.graph.get_heuristic(action[1])
                heapq.heappush(open_heap, (f_value, state_dict))
        
        # Construct path based on trajectory
        path = []
        while curr_state_dict["parent"] is not None:
            path.append(curr_state_dict["node_idx"])
            curr_state_dict = curr_state_dict["parent"]
        return path
    
    def apply_constriants(self, actions, timestep):
        """
            return valid actions that respects the constriants

            actions: [(v1, v2), ...] 
                Travel from v1 to v2
            
            timestep:
                Timestep of of the action
        """
        t_constriants = self.get_constraint_at_timestep(timestep)
        return [action for action in actions if action[1] not in t_constriants[action[0]]]
    
    def get_constraint_at_timestep(self, timestep):
        """
            return constriants indexed by node idx at timestep t
        """
        t_constraints = defaultdict(list)
        for constraint in self.constraints[timestep]:
            t_constraints[constraint[0]].append(constraint[1])
        return t_constraints

    def goal_test(self, node_idx):
        """
            Test if goal is reached
        """
        return node_idx in self.graph.goals_idx




class PrioritizedPlanning_MaxFlow:
    """
        Prioritized Planning for finding earliest timestep to reach target flow
    """
    def __init__(self, gaussian_prm:GaussianPRM, agent_radius, target_flow) -> None:
        self.gaussian_prm = gaussian_prm
        self.agent_radius = agent_radius
        self.target_flow = target_flow
        self.abstract_graph= AbstractGraph(gaussian_prm, agent_radius)
        self.nodes = [i for i in range(len(self.gaussian_prm.samples))]
