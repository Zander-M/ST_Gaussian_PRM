"""
    Min cost flow on Time Expanded Graph. Result from previous timestep is reused
    to speed up the algorithm.
"""

from collections import defaultdict, deque
from swarm_prm.utils.gaussian_prm import GaussianPRM
from swarm_prm.solvers.macro.mcf import MinCostFlowSolver 

# distinguish nodes

IN_NODE = 0
OUT_NODE = 1 

def dd():
    return defaultdict()

class TEG_MCF:
    def __init__(self, gaussian_prm:GaussianPRM, agent_radius, num_agents, max_timestep=100) -> None:
        self.gaussian_prm = gaussian_prm
        self.agent_radius = agent_radius
        self.num_agents = num_agents
        self.max_timestep = max_timestep
        self.roadmap_graph, self.cost_dict = self.build_roadmap_graph()
        self.nodes = [i for i in range(len(self.gaussian_prm.samples))]
        self.node_capacity = [node.get_capacity(self.agent_radius) for node in self.gaussian_prm.gaussian_nodes]

        # Verify if instance is feasible
        for i, start in enumerate(self.gaussian_prm.starts_idx):
            assert self.node_capacity[start] >= int(self.num_agents*self.gaussian_prm.starts_weight[i]),\
                "Start capacity smaller than required."

        for i, goal in enumerate(self.gaussian_prm.goals_idx):
            assert self.node_capacity[goal] >= int(self.num_agents*self.gaussian_prm.goals_weight[i]), \
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

        # adding wait edges
        for i in range(len(self.gaussian_prm.samples)):
            graph[i].append(i) # waiting at node has 0 transport cost
            cost[i][i] = 0
        return graph, cost

    def build_teg(self, timestep):
        """
            Build TEG based on timestep
        """
        teg = defaultdict(dict)
        super_source = ("SS", None, OUT_NODE)
        super_sink = ("SG", None, IN_NODE)

        # Adding super source and super goal to the graph
        for i, start_idx in enumerate(self.gaussian_prm.starts_idx):
            teg[super_source][(start_idx, 0, IN_NODE)] = \
                int(self.num_agents*self.gaussian_prm.starts_weight[i])
            teg[(start_idx, 0, IN_NODE)][(start_idx, 0, OUT_NODE)] = self.node_capacity[start_idx]

        for i, goal_idx in enumerate(self.gaussian_prm.goals_idx):
            teg[(goal_idx, timestep, OUT_NODE)][super_sink] = \
                int(self.num_agents*self.gaussian_prm.goals_weight[i])

        # adding graph edges
        for t in range(timestep):
            for u in self.roadmap_graph:
                for v in self.roadmap_graph[u]:
                    teg[(u, t, OUT_NODE)][(v, t+1, IN_NODE)] = float("inf")
                    teg[(v, t+1, IN_NODE)][(v, t+1, OUT_NODE)] = self.node_capacity[v]
        return super_source, super_sink, teg 
    
    def build_residual_graph_cost_graph(self, teg):
        """
            Build Residual Graph and Cost Graph from Time Expanded Graph
        """
        residual_graph = defaultdict(lambda:dict())
        cost_graph = defaultdict(lambda:dict())
        for u in teg:
            for v in teg[u]:
                residual_graph[u][v] = teg[u][v]
                residual_graph[v][u] = 0
                if u == ("SS", None, OUT_NODE) or \
                    v ==("SG", None, IN_NODE) :
                    cost_graph[u][v] = 0
                    cost_graph[v][u] = 0
                elif u[2] == OUT_NODE:
                    cost_graph[u][v] = self.cost_dict[u[0]][v[0]]
                    cost_graph[v][u] = -self.cost_dict[v[0]][u[0]]
                else:
                    cost_graph[u][v] = 0
                    cost_graph[v][u] = 0
        return residual_graph, cost_graph

    def update_residual_graph_cost_graph(self, teg, residual_graph, cost_graph, timestep, super_sink):
        """
            Update Residual Graph and Cost Graph for one timestep from previous timestep 
        """
        ### TEG
        # update edges to super sink
        for i, goal_idx in enumerate(self.gaussian_prm.goals_idx):
            teg[(goal_idx, timestep, OUT_NODE)][super_sink] = \
                int(self.num_agents*self.gaussian_prm.goals_weight[i])
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
                residual_graph[(u, timestep-1, OUT_NODE)][(v, timestep, IN_NODE)] = float("inf")
                residual_graph[(v, timestep, IN_NODE)][(u, timestep-1, OUT_NODE)] = 0 
                residual_graph[(v, timestep, IN_NODE)][(v, timestep, OUT_NODE)] = self.node_capacity[v]
                residual_graph[(v, timestep, OUT_NODE)][(v, timestep, IN_NODE)] = 0

                cost_graph[(u, timestep-1, OUT_NODE)][(v, timestep, IN_NODE)] = self.cost_dict[u][v] 
                cost_graph[(v, timestep, IN_NODE)][(u, timestep-1, OUT_NODE)] = -self.cost_dict[u][v] 
                cost_graph[(v, timestep, IN_NODE)][(v, timestep, OUT_NODE)] = 0 
                cost_graph[(v, timestep, OUT_NODE)][(v, timestep, IN_NODE)] = 0

        # update goals. Preserve flow from previous residual flow
        for i, goal_idx in enumerate(self.gaussian_prm.goals_idx):
            flow = residual_graph[super_sink][(goal_idx, timestep-1, OUT_NODE)]

            residual_graph[(goal_idx, timestep-1, OUT_NODE)][(goal_idx, timestep, IN_NODE)] = float("inf")
            residual_graph[(goal_idx, timestep, IN_NODE)][(goal_idx, timestep-1, OUT_NODE)] = flow 

            residual_graph[(goal_idx, timestep, IN_NODE)][(goal_idx, timestep, OUT_NODE)] = self.node_capacity[goal_idx] - flow
            residual_graph[(goal_idx, timestep, OUT_NODE)][(goal_idx, timestep, IN_NODE)] = flow

            residual_graph[(goal_idx, timestep, OUT_NODE)][super_sink] = int(self.gaussian_prm.goals_weight[i]*self.num_agents) - flow
            residual_graph[super_sink][(goal_idx, timestep, OUT_NODE)] = flow

            cost_graph[(goal_idx, timestep, OUT_NODE)][super_sink] = 0
            cost_graph[super_sink][(goal_idx, timestep, OUT_NODE)] = 0

            del residual_graph[(goal_idx, timestep-1, OUT_NODE)][super_sink]
            del residual_graph[super_sink][(goal_idx, timestep-1, OUT_NODE)]
            del cost_graph[(goal_idx, timestep-1, OUT_NODE)][super_sink]
            del cost_graph[super_sink][(goal_idx, timestep-1, OUT_NODE)]

    def get_earliest_timestep(self):
        """
            Find earliest timestep such that the graph reaches target flow
        """
        # start from minimum path lengh between start and goal
        timestep = self.get_min_timestep()
        max_flow = 0
        cost = 0

        super_source, super_sink, teg = self.build_teg(timestep)
        residual_graph, cost_graph = self.build_residual_graph_cost_graph(teg)

        while timestep < self.max_timestep:
            max_flow, residual_graph, cost = MinCostFlowSolver(super_source, super_sink, 
                                    residual_graph, cost_graph, initial_flow=max_flow, initial_cost=cost).solve()
            print("Time step: ", timestep, "Max Flow: ", max_flow, "Cost: ", cost) 
            # by construction the max flow will not exceed the target flow
            if max_flow == self.num_agents:
                flow_dict = self._residual_to_flow(teg, residual_graph) # remove residual graph edges
                return timestep, flow_dict
            timestep += 1
            self.update_residual_graph_cost_graph(teg, residual_graph, cost_graph, timestep, super_sink)
        return 0, None
    
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
