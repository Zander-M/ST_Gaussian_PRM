"""
    Given single agent solution trajectories, use APF
    to derive actual solution paths
"""

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.distance import cdist


class APF:
    """
        APF planner
    """
    def __init__(self, solution_paths, **apf_config):
        
        # APF parameters
        self.solution_paths = solution_paths
        self.num_agents = len(solution_paths)
        self.goal_tolerance = apf_config.get("goal_tolerance", 0.001)
        self.step_size = apf_config.get("step_size", 0.05)
        self.k_attr = apf_config.get("k_attr", 1.0)
        self.k_rep = apf_config.get("k_rep", 0.0002)
        self.repulsion_radius = apf_config.get("repulsion_radius", 0.55)
        self.max_rep_force = apf_config.get("max_rep_force", 1.0)
        self.min_dist = apf_config.get("min_dist", 0.05)
        self.max_step = apf_config.get("max_step", 0.07)
        self.damping = apf_config.get("damping", 0.7)

    def update(self, t):
        """
            Update trajectories for one uniform timestep
            Return 
            TODO: implement this
        """
        positions = np.array([path[t] for path in self.solution_paths])
        goals = np.array([path[t+1] for path in self.solution_paths])
        trajectories = np.empty_like(positions) 
        reached_goal = False
        while not reached_goal:
            dists_to_goals = np.linalg.norm(positions - goals, axis=1)
            if np.any(dists_to_goals < self.goal_tolerance):
                reached_goal = True
            else: 
                forces = self.compute_apf_forces(positions, goals)
                step = self.step_size * forces
                norm = np.linalg.norm(step)
                step = np.min(self.max_step * step / norm)
                trajectories = np.vstack((trajectories, positions))
                positions += self.damping*step
        return trajectories

    def compute_apf_forces(self, pos, goals):
        forces = np.zeros_like(pos)
        for i in range(len(pos)):
            # Attractive force
            goal_diff = goals[i] - pos[i]
            f_attr = self.k_attr * goal_diff

            # Repulsice force from other agents
            f_rep = np.zeros(2)
            agent_diff = pos[i] - pos
            dist = cdist([pos[i]], pos).flatten()
            mask = (dist < self.repulsion_radius) & (dist > 1e-6)
            dists_clamped = np.clip(dist, self.min_dist, np.inf)
            rep_mag = self.k_rep * (1.0 / dists_clamped - 1.0 / self.repulsion_radius) / (dists_clamped**2)
            rep_mag = np.clip(rep_mag, 0, self.max_rep_force)
            rep_mag[~mask] = 0 # masking

            agent_diff_normed = np.divide(agent_diff, dists_clamped[..., np.newaxis], 
                out=np.zeros_like(agent_diff), where=dists_clamped[..., np.newaxis] > 0)

            f_rep += np.sum(rep_mag[..., np.newaxis] * agent_diff_normed , axis=0)
            forces[i] = f_attr + f_rep
        return forces

    def solve(self):
        """
            Find trajectories across the map
            TODO: implement this
        """
        segments = []
        for t in range(len(self.solution_paths[0])-1):
            trajectories = self.update(t)
            segments.append(trajectories)
        return segments

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import numpy as np

    from swarm_prm.solvers.micro.apf import APF

    num_agents = 100

    # generate per agent trajectory

    means = np.array([(0., 0.), (1., 0.), (1.5, 1), (2., 2.)])
    covs = [np.eye(2) for _ in means]

    samples = np.stack([np.random.multivariate_normal(mean, cov, size=num_agents) 
               for mean, cov in zip(means, covs)])

    paths = np.einsum("ijk->jik", samples)

    print("paths", paths.shape)

    trajectories = APF(paths).solve()

    apf_config = {

    }