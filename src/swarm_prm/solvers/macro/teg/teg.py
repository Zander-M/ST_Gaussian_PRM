"""
    Max Flow on Time Expanded Graph, followed by Min Cost Flow with the found 
    solution timestep. Result from previous timestep is reused to speed up 
    the algorithm.
"""

import time

from collections import defaultdict, deque
from swarm_prm.solvers.macro import MacroSolverBase, register_solver
from swarm_prm.solvers.macro.teg import MaxFlow, MinCostFlow

# node labels 

IN_NODE = 0
OUT_NODE = 1 

@register_solver("TEGSolver")
class TEGSolver(MacroSolverBase):
    def init_solver(self, **kwargs) -> None:
        pass

    def get_min_timestep(self):
        """
            Find the ealiest timestep for any agent to reach any goal using bfs
        """
        open_list = deque(zip([start for start in self.starts_idx], [0] * len(self.starts_idx)))
        goals = set(self.goals_idx)
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
    
    def get_path(self, flow_dict, timestep):
        """
            Get individual solution paths
        """
        paths = [[] for _ in range(self.num_agents)]

        # Build reverse lookup: where can I go from here?
        next_moves = defaultdict(list)
        for u in flow_dict:
            for v, flow in flow_dict[u].items():
                if flow > 0:
                    next_moves[u].extend([v] * flow)

        for agent_id in range(self.num_agents):
            u = ("SS", None)
            while u != ("SG", None):
                v = next_moves[u].pop()
                if v[0] is not None and v[0] != "SG":
                    paths[agent_id].append(v[0])
                u = v

        # padding paths to solution length, agent waits at goal
        for i in range(self.num_agents):
            if not paths[i]:
                raise RuntimeError(f"Agent {i} has no valid path.")
            goal = paths[i][-1]
            pad_len = timestep - len(paths[i])
            paths[i].extend([goal] * pad_len)
        return paths

    def get_cost(self, paths):
        """
            Get average cost per agent. We use Wasserstein distance between states
            as an estimator.
        """
        cost = 0
        for path in paths:
            for u, v in zip(path[:-1], path[1:]):
                cost += self.cost_dict[u][v]
        return cost / len(paths)

    def build_teg(self, timestep):
        """
            Build TEG based on timestep
        """
        teg = defaultdict(dict)
        super_source = ("SS", None, OUT_NODE)
        super_sink = ("SG", None, IN_NODE)

        # Adding super source and super goal to the graph
        for i, start_idx in enumerate(self.starts_idx):
            teg[super_source][(start_idx, 0, IN_NODE)] = self.starts_agent_count[i]
            teg[(start_idx, 0, IN_NODE)][(start_idx, 0, OUT_NODE)] = self.node_capacity[start_idx]

        for i, goal_idx in enumerate(self.goals_idx):
            teg[(goal_idx, timestep, OUT_NODE)][super_sink] = self.goals_agent_count[i]

        # adding graph edges
        for t in range(timestep):
            for u in self.roadmap:

                # Edge for capacity constraints. We have capacities occupied by previous agents. Capacity Constraint
                teg[(u, t+1, IN_NODE)][(u, t+1, OUT_NODE)] = self.node_capacity[u]  

                # Avoid sharing node between swarms
                for capacity_dict in self.capacity_dicts: 
                    if (u, t+1) in capacity_dict:
                        teg[(u, t+1, IN_NODE)][(u, t+1, OUT_NODE)] = 0
                        break

                # Avoid dynamic obstacle/swarm goal states
                for obstacle_goal_dict in self.obstacle_goal_dicts: 
                    if u in obstacle_goal_dict and t+1 > obstacle_goal_dict[u]:
                        teg[(u, t+1, IN_NODE)][(u, t+1, OUT_NODE)] = 0
                        break

                # Edge between states
                for v in self.roadmap[u]:
                    # check if inverse edge between different nodes exists. Edge Constraint
                    if (u != v):
                        edge_exists = False
                        for flow_dict in self.flow_dicts:
                            if (v, t) in flow_dict and (u, t+1) in flow_dict[(v, t)]:
                                edge_exists = True
                                break
                        if not edge_exists:
                            teg[(u, t, OUT_NODE)][(v, t+1, IN_NODE)] = float("inf")
                    else: # wait edges
                        teg[(u, t, OUT_NODE)][(v, t+1, IN_NODE)] = float("inf")

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
        for i, goal_idx in enumerate(self.goals_idx):
            teg[(goal_idx, timestep, OUT_NODE)][super_sink] = self.goals_agent_count[i] 
            del teg[(goal_idx, timestep-1, OUT_NODE)][super_sink]                

        # update edges
        for u in self.roadmap:
            for v in self.roadmap[u]:
                teg[(u, timestep-1, OUT_NODE)][(v, timestep, IN_NODE)] = float("inf")
                
                is_obstacle_goal = False
                for obstacle_goal_dict in self.obstacle_goal_dicts:
                    if v in obstacle_goal_dict and obstacle_goal_dict[v] <= timestep:
                        is_obstacle_goal = True
                        break
                teg[(v, timestep, IN_NODE)][(v, timestep, OUT_NODE)] = 0 if is_obstacle_goal else self.node_capacity[v]

        ### Residual Dict
        # update edges
        for u in self.roadmap:
            for v in self.roadmap[u]:
                residual_graph[(u, timestep-1, OUT_NODE)][(v, timestep, IN_NODE)] = float("inf")
                residual_graph[(v, timestep, IN_NODE)][(u, timestep-1, OUT_NODE)] = 0 
                is_obstacle_goal = False
                for obstacle_goal_dict in self.obstacle_goal_dicts:
                    if v in obstacle_goal_dict and obstacle_goal_dict[v] <= timestep:
                        is_obstacle_goal = True
                        break
                residual_graph[(v, timestep, IN_NODE)][(v, timestep, OUT_NODE)] = 0 if is_obstacle_goal else self.node_capacity[v]
                residual_graph[(v, timestep, OUT_NODE)][(v, timestep, IN_NODE)] = 0

                cost_graph[(u, timestep-1, OUT_NODE)][(v, timestep, IN_NODE)] = self.cost_dict[u][v] 
                cost_graph[(v, timestep, IN_NODE)][(u, timestep-1, OUT_NODE)] = -self.cost_dict[u][v] 
                cost_graph[(v, timestep, IN_NODE)][(v, timestep, OUT_NODE)] = 0 
                cost_graph[(v, timestep, OUT_NODE)][(v, timestep, IN_NODE)] = 0

        # update goals. Preserve flow from previous residual flow
        # BE CAREFUL, WE ASSUME GOAL STATES ARE DISJOINT FOR DIFFERNT SWARM
        for i, goal_idx in enumerate(self.goals_idx):
            flow = residual_graph[super_sink][(goal_idx, timestep-1, OUT_NODE)]

            residual_graph[(goal_idx, timestep-1, OUT_NODE)][(goal_idx, timestep, IN_NODE)] = float("inf")
            residual_graph[(goal_idx, timestep, IN_NODE)][(goal_idx, timestep-1, OUT_NODE)] = flow 

            residual_graph[(goal_idx, timestep, IN_NODE)][(goal_idx, timestep, OUT_NODE)] = self.node_capacity[goal_idx] - flow
            residual_graph[(goal_idx, timestep, OUT_NODE)][(goal_idx, timestep, IN_NODE)] = flow

            residual_graph[(goal_idx, timestep, OUT_NODE)][super_sink] = self.goals_agent_count[i] - flow
            residual_graph[super_sink][(goal_idx, timestep, OUT_NODE)] = flow

            cost_graph[(goal_idx, timestep, OUT_NODE)][super_sink] = 0
            cost_graph[super_sink][(goal_idx, timestep, OUT_NODE)] = 0

            del residual_graph[(goal_idx, timestep-1, OUT_NODE)][super_sink]
            del residual_graph[super_sink][(goal_idx, timestep-1, OUT_NODE)]
            del cost_graph[(goal_idx, timestep-1, OUT_NODE)][super_sink]
            del cost_graph[super_sink][(goal_idx, timestep-1, OUT_NODE)]

    def solve(self, **constraint_dicts):
        """
            Find earliest timestep such that the graph reaches target flow
        """
        self.flow_dicts = constraint_dicts.get("flow_dicts", [])
        self.obstacle_goal_dicts = constraint_dicts.get("obstacle_goal_dicts", [])
        self.capacity_dicts = constraint_dicts.get("capacity_dicts", [])
        self.max_timestep = constraint_dicts.get("max_timestep", 0)


        # Construct solution that is at least as long as the existing solutions. 
        timestep = max(self.get_min_timestep(), self.max_timestep)
        max_flow = 0

        super_source, super_sink, teg = self.build_teg(timestep)
        residual_graph, cost_graph = self.build_residual_graph_cost_graph(teg)

        start_time = time.time()
        while time.time() - start_time < self.time_limit:
            max_flow, residual_graph = MaxFlow(super_source, super_sink, 
                                    residual_graph=residual_graph, initial_flow=max_flow).solve()

            # Solution Found
            if max_flow == self.num_agents:

                # Reduce solution flow cost
                ss, sg, teg = self.build_teg(timestep)
                _, cost_graph = self.build_residual_graph_cost_graph(teg)
                flow_dict = MinCostFlow(teg, 
                                        cost_graph,
                                        ss,
                                        sg,
                                        self.num_agents
                                        ).solve()

                capacity_dict = self._flow_to_capacity(flow_dict)
                goal_state_dict = {goal_idx: timestep for goal_idx in self.goals_idx}
                paths = self.get_path(flow_dict, timestep)
                cost = self.get_cost(paths)
                # TODO: compute suboptimality

                return {
                    "timestep": timestep, 
                    "g_nodes": self.gaussian_prm.gaussian_nodes,
                    "starts_idx": self.starts_idx,
                    "goals_idx": self.goals_idx,
                    "paths": paths,
                    "cost" : cost,
                    "flow_dict": flow_dict, 
                    "capacity_dict": capacity_dict, 
                    "goal_state_dict": goal_state_dict, 
                    "success": True
                    }

            timestep += 1
            self.update_residual_graph_cost_graph(teg, residual_graph, cost_graph, timestep, super_sink)
        print("Timelimit Exceeded.")
        return {"success": False} 
    
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
