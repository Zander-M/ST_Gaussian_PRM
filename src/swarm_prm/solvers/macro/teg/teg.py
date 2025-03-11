"""
    Max flow on Time Expanded Graph. Result from previous timestep is reused
    to speed up the algorithm.
"""

import time

from collections import defaultdict, deque
from swarm_prm.utils.gaussian_prm import GaussianPRM
from swarm_prm.solvers.macro.teg.max_flow import MaxFlowSolver

# distinguish nodes

IN_NODE = 0
OUT_NODE = 1 

def dd():
    return defaultdict()

class TEG:
    def __init__(self, gaussian_prm:GaussianPRM, agent_radius, 
                 num_agents, starts_agent_count, goals_agent_count, 
                 time_limit=100) -> None:
        self.gaussian_prm = gaussian_prm
        self.agent_radius = agent_radius
        self.num_agents = num_agents
        self.starts_agent_count = starts_agent_count
        self.goals_agent_count = goals_agent_count
        self.time_limit = time_limit 
        self.roadmap_graph = self.build_roadmap_graph()
        self.nodes = [i for i in range(len(self.gaussian_prm.samples))]
        self.node_capacity = [node.get_capacity(self.agent_radius) for node in self.gaussian_prm.gaussian_nodes]

        # Verify if instance is feasible
        for i, start in enumerate(self.gaussian_prm.starts_idx):
            assert self.node_capacity[start] >= self.starts_agent_count[i],\
                "Start capacity smaller than required."

        for i, goal in enumerate(self.gaussian_prm.goals_idx):
            assert self.node_capacity[goal] >= self.goals_agent_count[i], \
                "Goal capacity smaller than required."

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
            for neighbor in self.roadmap_graph[curr_node]:
                if neighbor not in visited:
                    open_list.append((neighbor, time+1))
        return 0

    def build_roadmap_graph(self):
        """
            Find the earliest timestep that reaches the max flow
        """
        graph = defaultdict(list)

        for edge in self.gaussian_prm.roadmap:
            u, v = edge
            graph[u].append(v)
            graph[v].append(u)

        # adding wait edges
        for i in range(len(self.gaussian_prm.samples)):
            graph[i].append(i)

        return graph

    def build_teg(self, timestep):
        """
            Build TEG based on timestep
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
            for u in self.roadmap_graph:
                for v in self.roadmap_graph[u]:
                    teg[(u, t, OUT_NODE)][(v, t+1, IN_NODE)] = float("inf")
                    teg[(v, t+1, IN_NODE)][(v, t+1, OUT_NODE)] = self.node_capacity[v]

        return super_source, super_sink, teg 
    
    def build_residual_graph(self, teg):
        """
            Build Residual Graph
        """
        residual_graph = defaultdict(lambda:dict())
        for u in teg:
            for v in teg[u]:
                residual_graph[u][v] = teg[u][v]
                residual_graph[v][u] = 0
        return residual_graph

    def update_teg_residual_dict(self, teg, residual_dict, timestep, super_sink):
        """
            Update TEG and Residual Dict for one timestep from previous timestep 
        """
        ### TEG
        # update edges to super sink
        for i, goal_idx in enumerate(self.gaussian_prm.goals_idx):
            teg[(goal_idx, timestep, OUT_NODE)][super_sink] = self.goals_agent_count[i]
            del teg[(goal_idx, timestep-1, OUT_NODE)][super_sink]                

        # update edges
        for u in self.roadmap_graph:
            for v in self.roadmap_graph[u]:
                teg[(u, timestep-1, OUT_NODE)][(v, timestep, IN_NODE)] = float("inf")
                teg[(v, timestep, IN_NODE)][(v, timestep, OUT_NODE)] = \
                    self.node_capacity[v]
    
        ### Residual Dict
        # update edges
        for u in self.roadmap_graph:
            for v in self.roadmap_graph[u]:
                residual_dict[(u, timestep-1, OUT_NODE)][(v, timestep, IN_NODE)] = float("inf")
                residual_dict[(v, timestep, IN_NODE)][(u, timestep-1, OUT_NODE)] = 0 
                residual_dict[(v, timestep, IN_NODE)][(v, timestep, OUT_NODE)] = self.node_capacity[v]
                residual_dict[(v, timestep, OUT_NODE)][(v, timestep, IN_NODE)] = 0

        # update goals. Preserve flow from previous residual flow
        for i, goal_idx in enumerate(self.gaussian_prm.goals_idx):
            flow = residual_dict[super_sink][(goal_idx, timestep-1, OUT_NODE)]

            residual_dict[(goal_idx, timestep-1, OUT_NODE)][(goal_idx, timestep, IN_NODE)] = float("inf")
            residual_dict[(goal_idx, timestep, IN_NODE)][(goal_idx, timestep-1, OUT_NODE)] = flow 

            residual_dict[(goal_idx, timestep, IN_NODE)][(goal_idx, timestep, OUT_NODE)] = self.node_capacity[goal_idx] - flow
            residual_dict[(goal_idx, timestep, OUT_NODE)][(goal_idx, timestep, IN_NODE)] = flow

            residual_dict[(goal_idx, timestep, OUT_NODE)][super_sink] = self.goals_agent_count[i] - flow
            residual_dict[super_sink][(goal_idx, timestep, OUT_NODE)] = flow

            del residual_dict[(goal_idx, timestep-1, OUT_NODE)][super_sink]
            del residual_dict[super_sink][(goal_idx, timestep-1, OUT_NODE)]

    def get_solution(self):
        """
            Find earliest timestep such that the graph reaches target flow
        """
        # start from minimum path lengh between start and goal
        timestep = self.get_min_timestep()
        max_flow = 0

        super_source, super_sink, teg = self.build_teg(timestep)
        residual_graph = self.build_residual_graph(teg)

        start_time = time.time()
        while time.time() - start_time < self.time_limit:
            max_flow, residual_graph = MaxFlowSolver(teg, super_source, super_sink, 
                                    residual_graph=residual_graph, initial_flow=max_flow).solve()
            print("Time step: ", timestep, "Max Flow: ", max_flow)

            # by construction the max flow will not exceed the target flow
            if max_flow == self.num_agents:
                flow_dict = self._residual_to_flow(teg, residual_graph) # remove residual graph edges
                return timestep, flow_dict, residual_graph
            timestep += 1
            self.update_teg_residual_dict(teg, residual_graph, timestep, super_sink)
        print("Timelimit Exceeded.")
        return None, None, None 
    
    def _residual_to_flow(self, teg, residual):
        """
            Construct forward flow graph from residual graph
        """
        flow_dict = defaultdict(dict)
        
        for u in teg:
            # we only look at out-node - in-node edges
            if u[-1] == OUT_NODE:
                for v in teg[u]:
                    flow = residual[v][u]
                    if flow > 0:
                            flow_dict[(u[0], u[1])][(v[0], v[1])] = flow
            
        return flow_dict
