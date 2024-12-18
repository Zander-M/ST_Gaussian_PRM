"""
    Max flow on Time Expanded Graph. Result from previous timestep is preserved
    to speed up the lookup.
"""

from collections import defaultdict, deque
from swarm_prm.solvers.macro.gaussian_prm.gaussian_prm import GaussianPRM
from swarm_prm.solvers.macro.teg.max_flow import MaxFlowSolver

class TEGGraph:
    def __init__(self, gaussian_prm:GaussianPRM, agent_radius, target_flow, max_timestep=100) -> None:
        self.gaussian_prm = gaussian_prm
        self.agent_radius = agent_radius
        self.target_flow = target_flow
        self.max_timestep = max_timestep
        self.roadmap_graph = self.build_roadmap_graph()
        self.nodes = [i for i in range(len(self.gaussian_prm.samples))]

    def get_min_timestep(self):
        """
            Find the ealiest timestep for any agent to reach any goal using bfs
            TODO: implement this 
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

    def build_teg(self, timestep):
        """
            Build TEG based on timestep
        """
        teg = defaultdict(lambda:dict())
        super_source = "SS"
        super_sink = "SG"

        node_idx = [i for i in range(len(self.gaussian_prm.samples))]
        # Adding super source and super goal to the graph

        for i, start_idx in enumerate(self.gaussian_prm.starts_idx):
            teg[super_source]['{}_{}'.format(start_idx, 0)] = int(self.gaussian_prm.starts_weight[i]*self.target_flow)

        for i, goal_idx in enumerate(self.gaussian_prm.goals_idx):
            teg['{}_{}'.format(goal_idx, timestep)][super_sink] = int(self.gaussian_prm.goals_weight[i]*self.target_flow)

        for t in range(timestep):
            # adding wait edges
            for u in node_idx:
                teg['{}_{}'.format(u, t)]['{}_{}'.format(u, t+1)] = float("inf")

            # adding graph edges
            for u in self.roadmap_graph:
                for v, capacity in self.roadmap_graph[u]:
                    teg['{}_{}'.format(u, t)]['{}_{}'.format(v, t+1)] = capacity

        return super_source, super_sink, teg 

    def update_teg_flow_dict(self, teg, flow_dict, prev_timestep, curr_timestep):
        """
            Update Time Expanded Graph from previous timestep 
        """

        ### TEG

        assert prev_timestep < curr_timestep, "Previous Timestep should be larger than curr timestep"
        super_sink = "SG"
        node_idx = [i for i in range(len(self.gaussian_prm.samples))]

        # update edges to super sink
        for _, goal_idx in enumerate(self.gaussian_prm.goals_idx):
            del teg['{}_{}'.format(goal_idx, prev_timestep)][super_sink] 

        for i, goal_idx in enumerate(self.gaussian_prm.goals_idx):
            teg['{}_{}'.format(goal_idx, curr_timestep)][super_sink] = int(self.gaussian_prm.goals_weight[i]*self.target_flow)

        # update edges
        for t in range(prev_timestep, curr_timestep):
            # adding wait edges
            for u in node_idx:
                teg['{}_{}'.format(u, t)]['{}_{}'.format(u, t+1)] = float("inf")

            # adding graph edges
            for u in self.roadmap_graph:
                for v, capacity in self.roadmap_graph[u]:
                    teg['{}_{}'.format(u, t)]['{}_{}'.format(v, t+1)] = capacity
        
        ### Flow Dict
        # Agent can wait at goals indefinitely. Adding wait 
        for t in range(prev_timestep, curr_timestep):
            for u in node_idx:
                # update wait flow
                flow_dict['{}_{}'.format(u, t)]['{}_{}'.format(u, t+1)] = float("inf")
                flow_dict['{}_{}'.format(u, t+1)]['{}_{}'.format(u, t)] = 0

                # update neighbor flow
                for v, capacity in self.roadmap_graph[goal_idx]:
                    flow_dict['{}_{}'.format(v, t)]['{}_{}'.format(goal_idx, t+1)] = capacity
                    flow_dict['{}_{}'.format(goal_idx, t+1)]['{}_{}'.format(v, t)] = 0

        # Update flow to super sink 
        for i, goal_idx in enumerate(self.gaussian_prm.goals_idx):
            flow_dict['{}_{}'.format(goal_idx, curr_timestep)][super_sink] = \
                flow_dict['{}_{}'.format(goal_idx, prev_timestep)][super_sink]

            flow_dict[super_sink]['{}_{}'.format(goal_idx, curr_timestep)] = \
                flow_dict[super_sink]['{}_{}'.format(goal_idx, prev_timestep)]

            d = flow_dict['{}_{}'.format(goal_idx, prev_timestep)]
            del d[super_sink]
            d = flow_dict[super_sink]
            del d['{}_{}'.format(goal_idx, prev_timestep)]

    def get_earliest_timestep(self):
        """
            Find earliest timestep such that the graph reaches target flow
        """
        timestep = self.get_min_timestep()
        max_flow = 0
        flow_dict = None

        super_source, super_sink, teg = self.build_teg(timestep)

        while timestep < self.max_timestep:
            max_flow, flow_dict = MaxFlowSolver(teg, super_source, super_sink, flow_dict=flow_dict).solve()
            print("timestep:", timestep, "max_flow:", max_flow)
            if max_flow == self.target_flow:
                return max_flow, flow_dict, timestep, teg 
            else:
                timestep += 1
            # import pprint
            # pprint.pprint(flow_dict["SG"])
            # pprint.pprint(flow_dict["240_22"])
            # pprint.pprint(flow_dict["241_22"])
            self.update_teg_flow_dict(teg, flow_dict, timestep-1, timestep)
            # pprint.pprint(flow_dict["SG"])
            # pprint.pprint(flow_dict["241_23"])
            # pprint.pprint(flow_dict["240_23"])
            # assert False

        return None, None, None, None 

    
    def flow_to_trajectory(self, flow_dict):
        """
            Convert Flow to Trajectories per agent
        """
        trajectory = []
        return trajectory
