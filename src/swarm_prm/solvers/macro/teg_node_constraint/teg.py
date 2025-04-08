"""
    Max flow on Time Expanded Graph. Result from previous timestep is reused
    to speed up the algorithm.
"""

import time

from collections import defaultdict, deque
from swarm_prm.utils import GaussianPRM
from swarm_prm.solvers.macro import MacroSolverBase, register_solver
from swarm_prm.solvers.macro.teg_node_constraint.max_flow import MaxFlow

# distinguish nodes

IN_NODE = 0
OUT_NODE = 1 

def dd():
    return defaultdict()

@register_solver("TEGNodeConstraintSolver")
class TEGNodeConstraintSolver(MacroSolverBase):
    def __init__(self, gaussian_prm:GaussianPRM, agent_radius, 
                 num_agents, starts_agent_count, goals_agent_count, 
                 flow_dicts=[], 
                 capacity_dicts = [],
                 max_timestep=0,
                 time_limit=100) -> None:

        # Problem instance
        self.gaussian_prm = gaussian_prm
        self.agent_radius = agent_radius
        self.num_agents = num_agents
        self.starts_agent_count = starts_agent_count
        self.goals_agent_count = goals_agent_count

        # Flow constraints
        self.flow_dicts = flow_dicts # existing flow on graph
        self.capacity_dicts = capacity_dicts
        self.max_timestep = max_timestep   # current solution time

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
            for neighbor in self.roadmap[curr_node]:
                if neighbor not in visited:
                    open_list.append((neighbor, time+1))
        return 0



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
            for u in self.roadmap:

                # Edge for capacity constraints. We have capacities occupied by previous agents. Capacity Constraint
                teg[(u, t+1, IN_NODE)][(u, t+1, OUT_NODE)] = self.node_capacity[u]  
                for capacity_dict in self.capacity_dicts:
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
        for u in self.roadmap:
            for v in self.roadmap[u]:
                teg[(u, timestep-1, OUT_NODE)][(v, timestep, IN_NODE)] = float("inf")
                teg[(v, timestep, IN_NODE)][(v, timestep, OUT_NODE)] = \
                    self.node_capacity[v]
    
        ### Residual Dict
        # update edges
        for u in self.roadmap:
            for v in self.roadmap[u]:
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
        timestep = max(self.get_min_timestep(), self.max_timestep)
        max_flow = 0

        super_source, super_sink, teg = self.build_teg(timestep)
        residual_graph = self.build_residual_graph(teg)

        start_time = time.time()
        while time.time() - start_time < self.time_limit:
            max_flow, residual_graph = MaxFlow(teg, super_source, super_sink, 
                                    residual_graph=residual_graph, initial_flow=max_flow).solve()
            # print("Time step: ", timestep, "Max Flow: ", max_flow) # TESTT

            # by construction the max flow will not exceed the target flow
            if max_flow == self.num_agents:
                flow_dict = self._residual_to_flow(teg, residual_graph) # remove residual graph edges
                capacity_dict = self._flow_to_capacity(flow_dict)
                return timestep, flow_dict, capacity_dict   
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
    
    def _flow_to_capacity(self, flow_dict):
        """
            Construct capacity dict indexed by (node, timestep), representing available flow
        """
        capacity_dict = defaultdict(lambda:0)
        for u in flow_dict:
            if u == ("SS", None):
                continue
            for v in flow_dict[u]:
                if v == ("SG", None):
                    continue
                capacity_dict[v] += flow_dict[u][v]
        return capacity_dict
