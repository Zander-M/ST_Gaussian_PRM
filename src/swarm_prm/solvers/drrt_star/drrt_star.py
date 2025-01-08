"""
    DRRT Star for continuous space motion planning
"""

import numpy as np
import matplotlib.pyplot as plt
import heapq

class Node:
    def __init__(self, position):
        self.position = np.array(position)
        self.cost = 0
        self.parent = None

class DRRTStar:
    def __init__(self, start_positions, goal_positions, obstacle_list, map_size, max_samples=1000, r=15):
        self.start_positions = [Node(pos) for pos in start_positions]
        self.goal_positions = goal_positions
        self.obstacle_list = obstacle_list
        self.map_size = map_size
        self.max_samples = max_samples
        self.r = r
        self.tree = [[] for _ in start_positions]

    def plan(self):
        for agent_idx, start in enumerate(self.start_positions):
            self.tree[agent_idx].append(start)
            for _ in range(self.max_samples):
                rand_node = self.get_random_node()
                nearest_node = self.get_nearest_node(agent_idx, rand_node)
                new_node = self.steer(nearest_node, rand_node)
                if not self.collision_check(nearest_node.position, new_node.position):
                    neighbors = self.find_neighbors(agent_idx, new_node)
                    min_cost_node = self.choose_parent(neighbors, nearest_node, new_node)
                    if min_cost_node:
                        new_node.parent = min_cost_node
                        new_node.cost = min_cost_node.cost + np.linalg.norm(new_node.position - min_cost_node.position)
                    self.tree[agent_idx].append(new_node)
                    self.rewire(agent_idx, new_node, neighbors)
            
    def get_random_node(self):
        return Node(np.random.rand(2) * self.map_size)

    def get_nearest_node(self, agent_idx, rand_node):
        return min(self.tree[agent_idx], key=lambda node: np.linalg.norm(node.position - rand_node.position))

    def steer(self, from_node, to_node):
        direction = to_node.position - from_node.position
        length = np.linalg.norm(direction)
        direction = direction / length if length > 0 else direction
        new_position = from_node.position + direction * min(self.r, length)
        return Node(new_position)

    def collision_check(self, start, end):
        for obs in self.obstacle_list:
            if np.linalg.norm(obs - start) + np.linalg.norm(obs - end) <= np.linalg.norm(start - end) + 2:
                return True
        return False

    def find_neighbors(self, agent_idx, node):
        return [n for n in self.tree[agent_idx] if np.linalg.norm(n.position - node.position) <= self.r]

    def choose_parent(self, neighbors, nearest_node, new_node):
        min_cost = nearest_node.cost + np.linalg.norm(new_node.position - nearest_node.position)
        min_cost_node = nearest_node
        for neighbor in neighbors:
            cost = neighbor.cost + np.linalg.norm(neighbor.position - new_node.position)
            if cost < min_cost and not self.collision_check(neighbor.position, new_node.position):
                min_cost = cost
                min_cost_node = neighbor
        return min_cost_node

    def rewire(self, agent_idx, new_node, neighbors):
        for neighbor in neighbors:
            new_cost = new_node.cost + np.linalg.norm(new_node.position - neighbor.position)
            if new_cost < neighbor.cost and not self.collision_check(new_node.position, neighbor.position):
                neighbor.parent = new_node
                neighbor.cost = new_cost

# Example usage
start_positions = [[10, 10], [20, 20]]
goal_positions = [[90, 90], [80, 80]]
obstacle_list = np.array([[50, 50], [30, 40]])
map_size = 100
planner = DRRTStar(start_positions, goal_positions, obstacle_list, map_size)
planner.plan()
