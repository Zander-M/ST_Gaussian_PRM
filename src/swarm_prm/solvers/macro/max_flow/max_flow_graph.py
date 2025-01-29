"""
    Max flow on Time Expanded Graph. Result from previous timestep is reused
    to speed up the algorithm.
"""

from collections import defaultdict, deque
from swarm_prm.solvers.utils.gaussian_prm import GaussianPRM
from swarm_prm.solvers.macro.teg.max_flow import MaxFlowSolver

class MaxFlowGraph:
    def __init__(self, gaussian_prm:GaussianPRM, agent_radius, target_flow, max_timestep=100) -> None:
        self.gaussian_prm = gaussian_prm
        self.agent_radius = agent_radius
        self.target_flow = target_flow
        self.max_timestep = max_timestep
        self.roadmap_graph = self.build_roadmap_graph()
        self.nodes = [i for i in range(len(self.gaussian_prm.samples))]

    def build_roadmap_graph(self, method="MIN_CAPACITY"):
        graph = defaultdict(list)
        for edge in self.gaussian_prm.roadmap:
            u, v = edge
            capacity = min(self.gaussian_prm.gaussian_nodes[u].get_capacity(self.agent_radius),
                           self.gaussian_prm.gaussian_nodes[v].get_capacity(self.agent_radius))
            graph[u].append((v, capacity))
            graph[v].append((u, capacity))
        return graph

    def get_min_timestep(self):
        """
            Find the ealiest timestep for any agent to reach any goal using bfs
        """
        open_list = deque(zip([start for start in self.gaussian_prm.starts_idx], [0] * len(self.gaussian_prm.starts_idx)))
        goals = set(self.gaussian_prm.goals_idx)
        visited = set()
        while open_list:
            curr_node, time = open_list.popleft() 
            if curr_node in goals:
                return time
            visited.add(curr_node)
            for neighbor, _ in self.roadmap_graph[curr_node]:
                if neighbor not in visited:
                    open_list.append((neighbor, time+1))
        return 0

    def build_abstract_graph(self):
        """
            Build abstract graph based on roadmap
        """
        teg = defaultdict(lambda:dict())
        super_source = "SS"
        super_sink = "SG"

        node_idx = [i for i in range(len(self.gaussian_prm.samples))]
        # Adding super source and super goal to the graph

        for i, start_idx in enumerate(self.gaussian_prm.starts_idx):
            teg[super_source]['{}'.format(start_idx, 0)] = int(self.gaussian_prm.starts_weight[i]*self.target_flow)

        for i, goal_idx in enumerate(self.gaussian_prm.goals_idx):
            teg['{}'.format(goal_idx)][super_sink] = int(self.gaussian_prm.goals_weight[i]*self.target_flow)

        # adding graph edges
        for u in self.roadmap_graph:
            for v, capacity in self.roadmap_graph[u]:
                teg['{}'.format(u)]['{}'.format(v)] = capacity

        return super_source, super_sink, teg 

    def update_teg_flow_dict(self, teg, flow_dict, timestep):
        """
            Update Time Expanded Graph from previous timestep 
        """
        ### TEG
        super_sink = "SG"
        # update edges to super sink
        for i, goal_idx in enumerate(self.gaussian_prm.goals_idx):
            del teg['{}_{}'.format(goal_idx, timestep-1)][super_sink] 
            teg['{}_{}'.format(goal_idx, timestep)][super_sink] = int(self.gaussian_prm.goals_weight[i]*self.target_flow)

        # update edges
        for u in self.roadmap_graph:
            # adding wait edges
            teg['{}_{}'.format(u, timestep-1)]['{}_{}'.format(u, timestep)] = float("inf")
            
            # adding graph edges
            for v, capacity in self.roadmap_graph[u]:
                teg['{}_{}'.format(u, timestep-1)]['{}_{}'.format(v, timestep)] = capacity
    
        ### Flow Dict

        # update edges
        for u in self.roadmap_graph:
            # adding wait edges
            flow_dict['{}_{}'.format(u, timestep-1)]['{}_{}'.format(u, timestep)] = float("inf")
            flow_dict['{}_{}'.format(u, timestep)]['{}_{}'.format(u, timestep-1)] = 0

            # adding graph edges
            for v, capacity in self.roadmap_graph[u]:
                flow_dict['{}_{}'.format(u, timestep-1)]['{}_{}'.format(v, timestep)] = capacity
                flow_dict['{}_{}'.format(v, timestep)]['{}_{}'.format(u, timestep-1)] = 0

        # update goals
        for i, goal_idx in enumerate(self.gaussian_prm.goals_idx):

            flow = flow_dict[super_sink]['{}_{}'.format(goal_idx, timestep-1)]
            flow_dict['{}_{}'.format(goal_idx, timestep-1)]['{}_{}'.format(goal_idx, timestep)] = float("inf")
            flow_dict['{}_{}'.format(goal_idx, timestep)][super_sink] = int(self.gaussian_prm.goals_weight[i]*self.target_flow) - flow
            flow_dict['{}_{}'.format(goal_idx, timestep)]['{}_{}'.format(goal_idx, timestep-1)] = flow 
            flow_dict[super_sink]['{}_{}'.format(goal_idx, timestep)] = flow

            del flow_dict['{}_{}'.format(goal_idx, timestep-1)][super_sink]
            del flow_dict[super_sink]['{}_{}'.format(goal_idx, timestep-1)]

    def _residual_to_flow(self, teg, residual):
        """
            Construct forward flow graph from residual graph
        """
        flow_dict = defaultdict(lambda: dict())
        for u in teg:
            for v in teg[u]:
                if teg[u][v] == float("inf"):
                    flow = residual[v][u]
                    if flow > 0:
                        flow_dict[u][v] = flow 
                else:
                    flow = teg[u][v] - residual[u][v]
                    if flow > 0:
                        flow_dict[u][v] = flow
        return flow_dict
