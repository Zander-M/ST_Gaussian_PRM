"""
    Prioritized Planning for Gaussian PRM
"""
import numpy as np

from collections import defaultdict
from pprint import pprint
import heapq

from swarm_prm.solvers.macro.swarm_prm.gaussian_prm import GaussianPRM

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
                graph[idx].append((idx, self.gaussian_prm.gaussian_nodes[idx].get_capacity(self.agent_radius)))

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

class STAStar:
    """
        Single agent spatio-temporal A star search on PRM for max flow w.r.t.
        the constraints

        We assume agents are always planning from super source to super sink

    """

    def __init__(self, nodes, graph:AbstractGraph, constraints=defaultdict(dict)):
        """
            nodes:
                Node indexes
            
            graph: 
                Abstract Graph

            constraints:
                Constraints indexed by timestep.
                Format:
                {timestep:[(v1, v2), ...], ...}
        """
        self.nodes = nodes
        self.graph = graph
        self.constraints = constraints 

    def search(self, start):
        """
            A Star search
        """
        open_heap = []
        visited = {}
        e_count = 0 # tie-breaking for heapq

        # Add start
        start_idx = self.graph.starts_idx[start]

        curr_state_dict = {
            "t": 0,
            "parent": None,
            "node_idx": start_idx
        }
        f_value = self.graph.get_heuristic(start_idx)
        heapq.heappush(open_heap, (f_value, e_count, curr_state_dict))
        e_count += 1

        while open_heap:
            # pop from open list, take one step and add new node to open list
            _, _, curr_state_dict =heapq.heappop(open_heap)
            curr_node_idx = curr_state_dict["node_idx"]
            visited[(curr_state_dict["node_idx"], curr_state_dict["t"])] = 0 

            # if reaching one of the goals, 
            if self.goal_test(curr_node_idx):
                break

            # all neighboring nodes + wait
            next_nodes = self.graph.get_neighbors(curr_node_idx) + [curr_node_idx]
            actions = [(curr_node_idx, next_node) for next_node in next_nodes \
                if not self.is_constrained((curr_node_idx, next_node), curr_state_dict["t"]) \
                    and (next_node, curr_state_dict["t"]) not in visited]
            
            for action in actions:
                state_dict = {
                    "t" : curr_state_dict["t"] + 1,
                    "parent" : curr_state_dict,
                    "node_idx" : action[1]
                }

                f_value = curr_state_dict["t"] + self.graph.get_heuristic(action[1])
                heapq.heappush(open_heap, (f_value, e_count, state_dict))
                e_count += 1
        
        # Construct path based on trajectory
        path = []
        while curr_state_dict["parent"] is not None:
            path.append(curr_state_dict["node_idx"])
            curr_state_dict = curr_state_dict["parent"]
        path.append(curr_state_dict["node_idx"])
        return path[::-1]

    
    def is_constrained(self, action, timestep):
        """
            return valid actions that respects the constraints

            actions: [(v1, v2), ...] 
                Travel from v1 to v2
            
            timestep:
                Timestep of of the action
        """
        return action in self.constraints[timestep]

    def goal_test(self, node_idx):
        """
            Test if goal is reached
        """
        return node_idx in self.graph.goals_idx

class PrioritizedPlanningMaxFlow:
    """
        Prioritized Planning for finding earliest timestep to reach target flow
    """
    def __init__(self, gaussian_prm:GaussianPRM, agent_radius, target_flow) -> None:
        self.gaussian_prm = gaussian_prm
        self.agent_radius = agent_radius
        self.target_flow = target_flow
        self.graph= AbstractGraph(gaussian_prm, agent_radius)
        self.nodes = [i for i in range(len(self.gaussian_prm.samples))]

    def solve(self):
        """
            Solve for agent trajectories
        """
        curr_flow = 0
        constraints = defaultdict(dict)
        paths = []

        # compute number of agents at starts
        starts_flow = []
        for i in range(len(self.gaussian_prm.starts_idx)):
           starts_flow.append(int(self.target_flow * self.gaussian_prm.starts_weight[i])) 
        
        while curr_flow < self.target_flow:

            # Choose start nodes with positive weight
            start = -1
            for i, source_flow in enumerate(starts_flow):
                if source_flow > 0:
                    start = i
                
            path = STAStar(self.nodes, self.graph, constraints).search(start)
            flow = self.graph.get_path_flow(path)

            # update flow dict
            self.graph.update_flow(path, flow)
            curr_flow += flow

            paths.append((path, flow))
            constraints = self.update_constraints(constraints, path)

    def update_constraints(self, constraints, path):
        """
            Convert Path to constraints

            Edge Constraint: Agents move in the same direction across the same edge
            Capacity Constraint: Cannot move into node with no available capacity

            Constraints are negative actions. The agents cannot take the actions
            listed in the constraints

            Constriant format:
            {t:[(v1, v2), ...]...}
        """
        # Edge Constraints
        print("Edge Constraints")
        for t, (u, v) in enumerate(zip(path[:-1], path[1:])):
            if (v, u) not in constraints[t]:
                constraints[t][(v, u)] = ""

        # Capacity Constraints
        print("Capacity Constraints")
        # Forbid agents to enter node that is full
        for t, node in enumerate(path):
            if self.graph.get_node_capacity(node, t) == 0:
                neighbors = self.graph.get_neighbors(node)
                for neighbor in neighbors:
                    constraints[t][(neighbor, node)] = ""
        return constraints


