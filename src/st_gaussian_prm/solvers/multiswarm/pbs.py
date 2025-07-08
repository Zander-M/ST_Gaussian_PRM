"""
    Priority Based Search for Swarm Planning
"""
from __future__ import annotations

from collections import defaultdict, deque
import copy
import time

class PBSNode:
    """
        PBS Tree nodes
    """
    def __init__(self):
        self.parents = defaultdict(set)  # parent nodes
        self.children = defaultdict(set) # chlid nodes
        self.solutions = {} # solutions
    
    def add_priority(self, i, j):
        """
            Add priority i -> j
        """
        self.parents[j].add(i)
        self.children[i].add(j)
    
    def has_cycle(self, start):
        """
            Detect cycle in priority ordering. Return true if detected cycle.
        """
        visited = set()
        stack = [start]
        while stack:
            node = stack.pop()
            if node in visited:
                return True
            visited.add(node)
            stack.extend(self.children.get(node, []))
        return False

    def copy(self):
        """
            Copy current node for updates
        """
        node = PBSNode()
        node.parents = copy.deepcopy(self.parents)
        node.children = copy.deepcopy(self.children)
        node.solutions = copy.deepcopy(self.solutions)
        return node
        
# Utils 

def get_predecessor_solutions(agent_id, parents, solutions):
    """
        Get predecessor solutions 
    """
    visited = set()
    stack =list(parents.get(agent_id, []))
    parents_solutions = []

    while stack:
        parent = stack.pop()
        if parent in visited:
            continue
        visited.add(parent)
        parents_solutions.append(solutions[parent])
        stack.extend(parents.get(parent, []))
    return parents_solutions

def get_replan_agents(agent_id, children):
    """
        Get agents to replan. Current agent + all descendants 
    """
    visited = {agent_id}
    stack = [agent_id]
    while stack:
        node = stack.pop()
        for child in children.get(node, []):
            if child not in visited:
                visited.add(child)
                stack.append(child)
    return visited

def get_constraint_dicts(solutions):
    """
        Get various dictionaries 

    """
    constraint_dicts = {
        "flow_dicts": [],
        "occupancy_sets": [],
        "obstacle_goal_dicts": [],
        "max_timestep" : 0
    }
    for solution in solutions:
        constraint_dicts["flow_dicts"].append(solution["flow_dict"])
        constraint_dicts["occupancy_sets"].append(solution["occupancy_set"])
        constraint_dicts["obstacle_goal_dicts"].append(solution["goal_state_dict"])
        constraint_dicts["max_timestep"] = max(constraint_dicts["max_timestep"], solution["timestep"])
    return constraint_dicts

def get_replan_order(agents, parents):
    """
        Get order to replan using topological sort. Kahn's algorithm.
    """
    # count in-degree
    in_degree = defaultdict(int)
    for agent in agents:
        for parent in parents.get(agent, set()):
            if parent in agents:
                in_degree[agent] += 1
    
    queue = deque([agent for agent in agents if in_degree[agent] == 0])
    result = []

    while queue:
        agent = queue.popleft()
        result.append(agent)
        for child in agents:
            if agent in parents.get(child, set()):
                in_degree[child] -= 1
                if in_degree[child] == 0:
                    queue.append(child)
    return result

def detect_conflict(solutions):
    """
        Detect inter-swarm collisions in solutions
    """
    n = len(solutions)

    occupancy = [s["occupancy_set"] for s in solutions]
    goals = [s["goal_state_dict"] for s in solutions]
    flows = [s["flow_dict"] for s in solutions]

    for i in range(n):
        for j in range(i+1, n):
            if i == j:
                continue

            # --- Vertex Conflict ---
            if occupancy[i] & occupancy[j]:
                return True, (i, j)

            # --- Goal Conflict ---
            # Check if j enters i's goal after i reaches it
            for goal_node, goal_time in goals[i].items():
                if any(node == goal_node and t > goal_time for node, t in occupancy[j]):
                    return True, (i, j)
            # Check if i enters j's goal after j reaches it
            for goal_node, goal_time in goals[j].items():
                if any(node == goal_node and t > goal_time for node, t in occupancy[i]):
                    return True, (i, j)

            # --- Edge Conflict ---
            # Check i's flow vs j's reversed flow
            flow_i = flows[i]
            flow_j = flows[j]
            for (u1, t1), out_edges_i in flow_i.items():
                if u1 == "SS":
                    continue
                for (v1, t2), f1 in out_edges_i.items():
                    if v1 == "SG" or t2 != t1 + 1 or f1 == 0 or u1 is None or v1 is None:
                        continue
                    if (v1, t1) in flow_j:
                        if (u1, t2) in flow_j[(v1, t1)]:
                            return True, (i, j)
    return False, (None, None)

# PBS

class PriorityBasedSearch:
    def __init__(self, instances, time_limit=180):
        self.instances = instances
        self.time_limit = time_limit

    def solve(self):
        start_time = time.time()
        num_agents = len(self.instances)

        # Initial PBS root node
        root = PBSNode()

        # Plan all agents independently
        for agent_id in range(num_agents):
            constraints = get_constraint_dicts([])  # no constraints initially
            solution = self.instances[agent_id].solve(**constraints)
            if not solution["success"]:
                return {"success": False}
            root.solutions[agent_id] = solution

        # Initialize open list with root
        open_list = [root]

        while open_list and (time.time() - start_time) < self.time_limit:
            node = open_list.pop()  # DFS

            # Check for conflicts
            sorted_solutions = [node.solutions[i] for i in sorted(node.solutions)]
            conflict_found, conflict = detect_conflict(sorted_solutions)

            if not conflict_found:
                # Success!
                sorted_paths = [node.solutions[i]["paths"] for i in range(num_agents)]
                agent_count = [len(path) for path in sorted_paths]

                # Compute max path length
                max_len = max(len(path[0]) for path in sorted_paths)

                # Flatten and pad each individual path
                padded_paths = []
                for agent_paths in sorted_paths:
                    for path in agent_paths:
                        padded = path + [path[-1]] * (max_len - len(path))
                        padded_paths.append(padded)
                return {
                    "success": True,
                    "paths": padded_paths,
                    "agent_count": agent_count
                }

            i, j = conflict

            for higher, lower in [(i, j), (j, i)]:
                child = node.copy()
                child.add_priority(higher, lower)

                # Check for cycle
                if child.has_cycle(lower):
                    continue

                # Replan affected subtree
                replan_set = get_replan_agents(lower, child.children)
                replan_order = get_replan_order(replan_set, child.parents)

                success = True
                for agent_id in replan_order:
                    predecessors = get_predecessor_solutions(agent_id, child.parents, child.solutions)
                    constraint_dicts = get_constraint_dicts(predecessors)
                    solution = self.instances[agent_id].solve(**constraint_dicts)
                    if not solution["success"]:
                        success = False
                        break
                    child.solutions[agent_id] = solution

                if success:
                    open_list.append(child)

        # Timed out or exhausted search
        return {"success": False}