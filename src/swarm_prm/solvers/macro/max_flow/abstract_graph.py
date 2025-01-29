"""
    Abstract Graph with capacity for path planning
"""
from collections import defaultdict
import heapq

import numpy as np

class AbstractGraph:
    """
        Abstract Graph for A star search
    """
    def __init__(self, gaussian_prm, agent_radius, constraints={}):
        self.gaussian_prm = gaussian_prm
        self.grid_size = self.gaussian_prm.hex_radius
        self.agent_radius = agent_radius
        self.nodes = self.gaussian_prm.samples 
        self.nodes_idx = [i for i in range(len(self.nodes))]

        self.nodes = self.gaussian_prm.samples 
        self.starts_idx = np.array(self.gaussian_prm.starts_idx)
        self.goals_idx = np.array(self.gaussian_prm.goals_idx)
        self.graph = self._build_graph()
        self.heuristic = self._compute_heuristic()
        self.constraints = constraints
        self.node_capacity = self._set_node_capacity()
        self.flow_dict = defaultdict(lambda:defaultdict(lambda:0))

    def _bfs(self, node_idx):
        """
            Breadth First Search from goals for heuristic computation.
            Return dictionary of heuristics
        """
        open_list = []
        h = {node_idx: 0}
        heapq.heappush(open_list, (0, node_idx))
        while open_list:
            v, node = heapq.heappop(open_list)
            for neighbor in self.get_neighbors(node):
                if neighbor not in h:
                    h[neighbor] = v+1
                    heapq.heappush(open_list, (v+1, neighbor))
        return h

    def _build_graph(self):
        """
            Build Graph with capacity constraints
        """
        graph = defaultdict(list)

        # no capacity limit at starts and goals
        for start_idx in self.starts_idx:
            graph[start_idx].append((start_idx, float("inf"))) 

        # capacity for wait at intermediate nodes
        for idx in self.nodes_idx:
            if idx not in self.starts_idx and idx not in self.goals_idx:
                graph[idx].append((np.int32(idx), self.gaussian_prm.gaussian_nodes[idx].get_capacity(self.agent_radius)))

        # use the capacity of the smaller node as the edge capacity
        for edge in self.gaussian_prm.roadmap:
            u, v = edge
            capacity = min(self.gaussian_prm.gaussian_nodes[u].get_capacity(self.agent_radius),
                           self.gaussian_prm.gaussian_nodes[v].get_capacity(self.agent_radius))
            graph[u].append((v, capacity))
            graph[v].append((u, capacity))

        return graph
    
    def get_starts(self):
        """
            Return start nodes
        """
        return self.starts_idx

    def _set_node_capacity(self):
        """
            Set node capacity
        """
        capacity_dict = {}
        for idx in self.nodes_idx:
            if idx in self.starts_idx:
                capacity_dict[idx] = float("inf")
            else:
                capacity_dict[idx] = self.gaussian_prm.gaussian_nodes[idx].get_capacity(self.agent_radius)
        return capacity_dict

    def update_flow(self, path, flow):
        """
            Update flow dict on the graph
        """
        for t, idx in enumerate(path):
            if idx not in self.flow_dict[t]:
                self.flow_dict[t][idx] = 0
            self.flow_dict[t][idx] += flow

    def get_path_flow(self, path):
        """
            Get path flow based on graph
        """
        max_flow = float("inf")
        for t, node in enumerate(path):
            max_flow = min(max_flow, self.get_node_capacity(node, t))
        return max_flow
    
    def get_neighbors(self, node_idx):
        """
            Get neighbors of a node
        """
        return [node[0] for node in self.graph[node_idx]]
        
        
    def _compute_heuristic(self):
        """
            Multi-source single goal from the goal locations
        """
        heuristic = defaultdict(lambda:float("inf"))
        for goal_idx in self.goals_idx:
            h = self._bfs(goal_idx)
            for node_idx in h:
                heuristic[node_idx] = min(heuristic[node_idx], h[node_idx])
        return heuristic

    def get_heuristic(self, node_idx):
        """
            Get heuristic of the next node
            Use minimum Eucledian distance normalized by grid size to the closest goals 
        """
        return self.heuristic[node_idx]
    
    def get_node_capacity(self, node_idx, t):
        """
            Get node capacity at node_idx at timestep t.
        """
        return self.node_capacity[node_idx] - self.flow_dict[t][node_idx]
