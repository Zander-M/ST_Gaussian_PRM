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

class ST_AStar:
    """
        Single agent spatio-temporal A star search on PRM for max flow w.r.t.
        the constriants

        We assume agents are always planning from super source to super sink

    """

    def __init__(self, nodes, graph, constraints):
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

    def a_star(self):
        """
            A Star search
        """
        timestep = 0
        open_heap = []
        node_idx = -1
        f_value = timestep + self.graph.get_heuristic(node_idx)
        heapq.heappush(open_heap, (f_value, timestep, node_idx))
        while open_heap:
            _, timestep, node_idx =heapq.heappop(open_heap)
            timestep += 1
            next_nodes = self.graph.get_neighbors(node_idx) + [node_idx]
            actions = [[node_idx, next_node] for next_node in next_nodes]
            constrained_actions = self.apply_constriants(actions, timestep)

            pass
    
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
        self.graph = self.build_capacity_graph()
        self.starts_idx = self.gaussian_prm.start_idx
        self.goals_idx = self.gaussian_prm.start_idx
        self.heuristic = self.compute_heuristic()

    def build_capacity_graph(self, method="MIN_CAPACITY"):
        """
            Build Graph with capacity constraints
        """
        graph = defaultdict(list)
        ss = "SS" # super start
        sg = "SG" # super goal

        # Add super start/goal

        for start_idx in self.gaussian_prm.starts_idx:
            capacity = self.gaussian_prm.gaussian_nodes[start_idx].get_capacity(self.agent_radius)
            graph[ss].append((start_idx, capacity))

        for goal_idx in self.gaussian_prm.goals_idx:
            capacity = self.gaussian_prm.gaussian_nodes[goal_idx].get_capacity(self.agent_radius)
            graph[goal_idx].append((sg, capacity))

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

    def update_capacity(self, constraints):
        """
            Update node constriants based on constriants
        """

    def get_path_capacity(self, path):
        """
            Get path flow based on map
        """

    def build_graph(self):
        """
            Build abstract graph from Gaussian PRM
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


    def find_earliest_timestep(self):
        """
            Find earliest timestep such that the graph reaches target flow
        """
        timestep = 0
        max_flow = 0
        flow_dict = {}
        while timestep < self.max_timestep:
            super_source, super_sink, teg, restricted_edges = self.build_teg(timestep)
            max_flow, flow_dict = nx.maximum_flow(teg, super_source, super_sink)
            print("timestep:", timestep, "max_flow:", max_flow)
            if max_flow == self.target_flow:
                return max_flow, flow_dict, timestep, teg, restricted_edges
            else:
                timestep += 1

        return None, None, None, None, None
    
    def flow_to_trajectory(self, flow_dict):
        """
            Convert Flow to Trajectories per agent
        """
        trajectory = []
        return trajectory

    def visualize_teg(self, teg, restricted_edges):
        """
            Visualize TEG 
        """
        node_labels = {node: node for node in teg.nodes()}
        edge_labels = nx.get_edge_attributes(teg, 'capacity')

        # Draw the graph
        plt.figure(figsize=(10, 8))  # Set the size of the figure
        teg = teg.to_undirected()
        pos = nx.bfs_layout(teg, "VS")  # type: ignore # Compute positions using the spring layout
        pos["SS"] = pos["VS"]

        # Hide visualization source and extra edges
        teg = nx.restricted_view(teg, ["VS"], restricted_edges)


        # Draw nodes and edges
        nx.draw(teg, pos, with_labels=True, node_color='lightblue', node_size=300, font_size=16, edge_color='gray')
        nx.draw_networkx_edge_labels(teg, pos, edge_labels=edge_labels)


        # Display the plot
        plt.title("Graph Visualization")
        plt.xlabel("X Coordinate")
        plt.ylabel("Y Coordinate")
        plt.grid(True)
        plt.show()