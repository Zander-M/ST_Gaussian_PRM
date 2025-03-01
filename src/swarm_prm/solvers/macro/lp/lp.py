"""
    Linear Programming solver for finding shortest paths
"""
from collections import defaultdict

class LPSolver:
    def __init__(self, gaussian_prm, num_agents, agent_radius):
        self.gaussian_prm = gaussian_prm
        self.starts = self.gaussian_prm.starts_idx
        self.goals = self.gaussian_prm.goals_idx
        self.num_agents = num_agents
        self.agent_radius = agent_radius
        self.roadmap, self.cost_dict = self.build_roadmap_graph()
        self.node_capacity = [node.get_capacity(agent_radius) for node in self.gaussian_prm.gaussian_nodes]

    def build_roadmap_graph(self):
        """
            Build graph with edge cost
        """
        graph = defaultdict(list)
        cost = defaultdict(defaultdict)
        for i, edge in enumerate(self.gaussian_prm.roadmap):
            u, v = edge
            graph[u].append(v)
            graph[v].append(u)
            cost[u][v] = self.gaussian_prm.roadmap_cost[i]
            cost[v][u] = self.gaussian_prm.roadmap_cost[i]
        return graph, cost

    def get_pairwise_shortest_paths(self):
        """
            Find shortest paths between starts and goals
        """
        shortest_paths = defaultdict(defaultdict)
        
    
    def get_solution(self):
        """
            Get solution paths
        """
    

    