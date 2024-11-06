"""
    Spatial Hash for quick neighbour lookup
"""
from collections import defaultdict
import math

import numpy as np

class SpatialHash:
    """
        Spatial Hash function for quick neighbour lookup
    """
    def __init__(self, grid_size) -> None:
        self.grid_size = grid_size
        self.hash_table = defaultdict(list)
        self.agent_positions = {}

    def _hash(self, position):
        """
            Get grid index
        """
        assert position[0] is not np.nan and position[1] is not np.nan, print(position)
        cell_x = math.floor(position[0] / self.grid_size)
        cell_y = math.floor(position[1] / self.grid_size)
        return (cell_x, cell_y)

    def insert(self, agent_id, position):
        """
            Insert agent position into hash table
        """
        cell = self._hash(position)
        self.hash_table[cell].append((agent_id, position))
        self.agent_positions[agent_id] = position

    def remove(self, agent_id, position):
        """
            Remove agent from hash
        """
        cell = self._hash(position)
        if cell in self.hash_table:
            self.hash_table[cell] = [
                (id, pos) for id, pos in self.hash_table[cell] if id != agent_id
            ]
            if not self.hash_table[cell]:
                del self.hash_table[cell]

    def update_position(self, agent_id, new_position):
        """
            Update agent position
        """
        old_position = self.agent_positions.get(agent_id)
        if old_position is None:
            self.insert(agent_id, new_position)
            return

        old_cell = self._hash(old_position)
        new_cell = self._hash(new_position)

        if old_cell != new_cell:
            self.remove(agent_id, old_cell)
            self.insert(agent_id, new_cell)
        
        self.agent_positions[agent_id] = new_position

    def query_radius(self, position, radius):
        """
            return neighboring agent indices within certain radius
        """
        cell_x, cell_y = self._hash(position)
        nearby_agents= []

        search_radius = math.ceil(radius / self.grid_size)
        
        for dx in range(-search_radius, search_radius+1):
            for dy in range(-search_radius, search_radius+1):
                cell = (cell_x + dx, cell_y + dy)
                for agent_id, agent_pos in self.hash_table.get(cell, []):
                    distance = np.linalg.norm(position-agent_pos)
                    if distance <= radius:
                        nearby_agents.append(agent_id)

        return nearby_agents