"""
    Max flow on Time Expanded Graph. Result from previous timestep is preserved
    to speed up the lookup.
"""

from collections import defaultdict
from matplotlib import pyplot as plt
import networkx as nx
from swarm_prm.solvers.macro.swarm_max_flow.gaussian_prm import GaussianPRM
from swarm_prm.solvers.macro.swarm_max_flow.max_flow import MaxFlowSolver

class TEGGraph:
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
        teg = defaultdict(list)
        super_source = "SS"
        super_sink = "SG"

        node_idx = [i for i in range(len(self.gaussian_prm.samples))]
        # Adding timestep -1 node for visualization purpose
        # Visualization source and restricted_edges are hidden
        # duirng TEG visualization

        restricted_edges  = []

        # Adding super source and super goal to the graph

        for i, start_idx in enumerate(self.gaussian_prm.starts_idx):
            teg[super_source].append((
                        '{}_{}'.format(start_idx, 0), 
                        int(self.gaussian_prm.starts_weight[i]*self.target_flow)
                        ))

        for i, goal_idx in enumerate(self.gaussian_prm.goals_idx):
            teg['{}_{}'.format(goal_idx, timestep)].append((
                        super_sink, 
                        int(self.gaussian_prm.goals_weight[i]*self.target_flow)
                        ))

        for t in range(timestep):
            # adding wait edges
            for u in node_idx:
                teg['{}_{}'.format(u, t)].append(('{}_{}'.format(u, t+1), float("inf")))

            # adding graph edges
            for u in self.roadmap_graph:
                for v, capacity in self.roadmap_graph[u]:
                    teg['{}_{}'.format(v, t)].append(('{}_{}'.format(u, t+1), capacity))
        return super_source, super_sink, teg, restricted_edges

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