"""
    Steering behaviour for swarm path following
"""

from typing import TYPE_CHECKING 

import numpy as np

from st_gaussian_prm.utils import spatial_hash

if TYPE_CHECKING:
    from typing import Tuple, List
    from st_gaussian_prm.utils import GaussianNode


class SteeringBehaviour:
    """
        Steering Behaviour
    """
    def __init__(self, gaussian_paths, gaussian_nodes, **steering_behaviour_config):

        self.gaussian_paths = gaussian_paths
        self.gaussian_nodes = gaussian_nodes

        self.anchor_dist = steering_behaviour_config.get("anchor_dist", 10)
        self.max_force = steering_behaviour_config.get("max_force", 0.1)
        self.max_vel = steering_behaviour_config.get("max_vel", 0.5)
        self.path_radius = steering_behaviour_config.get("path_radius", 5)
        self.update_time_interval = steering_behaviour_config.get("update_time_interval", 1)

    def get_timestep_segment(self, t):
        """
            Given the timestep, get current line segment
        """
        pass

    def seek(self, positions:np.ndarray, targets:List[GaussianNode])-> np.ndarray:
        """
            Let agents follow targets in Gaussian APF style
        """

        return np.zeros(0)

    def solve(self):
        """
            Find single agent trajectories across the map
        """
        pass

    def update(self, line:np.ndarray, positions:np.ndarray, velocities:np.ndarray) -> Tuple[np.ndarray, bool]:
        """
            Compute steering vector based on current path segment 
        """

        pos = np.copy(positions)

        # Compute directional vector
        future_pos = pos + velocities * self.update_time_interval

        future_vec = (future_pos - line[0])
        line_norm = (line[1] - line[0]) / np.linalg.norm((line[1]-line[0]))
        norm_point = line[0] + line_norm * np.dot(future_vec, line_norm)
        dist_vec = future_vec - norm_point
        mask = dist_vec[np.linalg.norm(dist_vec) > self.path_radius]


        # Compute repelling force

        return np.zeros(0), False 