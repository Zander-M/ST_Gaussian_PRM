"""
    TEG implementation using NetworkX
"""

from collections import defaultdict
from matplotlib import pyplot as plt
import networkx as nx
from swarm_prm.solvers.swarm_prm.macro.gaussian_prm import GaussianPRM

class TEGGraph_NX:
    def __init__(self, gaussian_prm:GaussianPRM, agent_radius, target_flow, max_timestep=100) -> None:
        self.gaussian_prm = gaussian_prm
        self.agent_radius = agent_radius
        self.target_flow = target_flow
        self.max_timestep = max_timestep
        self.roadmap_graph = self.build_roadmap_graph()
        self.nodes = [i for i in range(len(self.gaussian_prm.samples))]

    def build_teg(self, timestep):
        """
            Build TEG based on timestep
        """
        teg = nx.DiGraph()
        super_source = "SS"
        super_goal = "SG"

        # Adding timestep -1 node for visualization purpose
        # Visualization source and restricted_edges are hidden
        # duirng TEG visualization

        vis_source = "VS" 
        restricted_edges  = []
        for vis_idx in range(1, 8):
            edge = (vis_source, '{}_{}'.format(vis_idx, 0))
            teg.add_edge(edge[0], edge[1])
            restricted_edges.append(edge)

        # Adding super source and super goal to the graph

        for start_idx in [1, 2]:
            teg.add_edge(super_source, '{}_{}'.format(start_idx, 0))

        for goal_idx in [6, 7]:
            teg.add_edge('{}_{}'.format(goal_idx, timestep),super_goal)

        for t in range(timestep):

            # adding wait edges
            for u in range(1, 8):
                teg.add_edge('{}_{}'.format(u, t+1), '{}_{}'.format(u, t))

            # adding graph edges
            for u in self.roadmap_graph:
                for v, capacity in self.roadmap_graph[u]:
                    teg.add_edge( '{}_{}'.format(v, t), '{}_{}'.format(u, t+1), capacity=capacity)
                    teg.add_edge( '{}_{}'.format(u, t), '{}_{}'.format(v, t+1), capacity=capacity)

        return super_source, super_goal, teg, restricted_edges

    def build_roadmap_graph(self, method="MIN_CAPACITY"):
        """
            Find the earliest timestep that reaches the max flow
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

    def find_earliest_timestep(self):
        """
            Find earliest timestep such that the graph reaches target flow
        """
        timestep = 0
        max_flow = 0
        flow_dict = {}
        while timestep < self.max_timestep:
            super_source, super_goal, teg, restricted_edges = self.build_teg(timestep)
            max_flow, flow_dict = nx.maximum_flow(teg, super_source, super_goal)
            if max_flow > self.target_flow:
                return max_flow, flow_dict, timestep, teg, restricted_edges
            else:
                timestep += 1

        return None, None, None, None, None

    def visualize_teg(self, teg, restricted_edges):
        """
            Visualize TEG 
        """
        node_labels = {node: node for node in teg.nodes()}
        edge_labels = nx.get_edge_attributes(teg, 'capacity')

        # Draw the graph
        plt.figure(figsize=(10, 8))  # Set the size of the figure
        teg = teg.to_undirected()
        pos = nx.bfs_layout(teg, "VS")  # Compute positions using the spring layout
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