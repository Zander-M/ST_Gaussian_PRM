"""
    Spatial Temporal A Star on abstract graph
"""
from collections import defaultdict
import heapq

from swarm_prm.solvers.macro.teg.abstract_graph import AbstractGraph


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

            # all neighboring nodes 
            next_nodes = self.graph.get_neighbors(curr_node_idx) 
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
