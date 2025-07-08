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
    def __init__(self, gaussian_paths, gaussian_nodes, **apf_config):
        
        # APF parameters. Emperically chosen for experiment
        self.gaussian_paths = gaussian_paths  
        self.num_agents = len(gaussian_paths)
        self.gaussian_nodes = gaussian_nodes
        self.goal_chisq_threshold= apf_config.get("goal_chisq_threshold", 5.991) # for 95% CI
        self.step_size = apf_config.get("step_size", 0.05)
        self.k_attr = apf_config.get("k_attr", 1.0)
        self.k_rep = apf_config.get("k_rep", 0.0002)
        self.repulsion_radius = apf_config.get("repulsion_radius", 0.55)
        self.max_rep_force = apf_config.get("max_rep_force", 1.0)
        self.min_dist = apf_config.get("min_dist", 0.05)
        self.max_step = apf_config.get("max_step", 0.07)
        self.damping = apf_config.get("damping", 0.7)
        self.max_apf_iterations = apf_config.get("max_apf_iterations", 5000)
    
    def update(self, t, positions, fixed_num_steps=1000):
        """
            Update trajectories for one uniform timestep using soft Gaussian
            attractive force.
            Return APF waypoints.
        """
        path_len = len(self.gaussian_paths[0])
        pad_path = False
        if t+1 < path_len:
            goals_gaussians = [self.gaussian_nodes[path[t+1]] for path in self.gaussian_paths]
        else:
            # Extra padding step
            goals_gaussians = [self.gaussian_nodes[path[t]] for path in self.gaussian_paths]
            pad_path = True
        goals_mean = np.array([goals_gaussian.mean for goals_gaussian in goals_gaussians])
        goals_cov = np.array([goals_gaussian.covariance for goals_gaussian in goals_gaussians])
        inv_cov = np.linalg.inv(goals_cov)

        trajectories = [] 
        iter_count = 0

        final_positions = np.copy(positions)
        while True:
            diff = positions - goals_mean
            mahalanobis_sq = np.einsum("ni,nij,nj->n", diff, inv_cov, diff)
            if not pad_path and np.all(mahalanobis_sq < self.goal_chisq_threshold):
                final_positions = np.copy(positions)
                break
            elif pad_path and iter_count == fixed_num_steps:
                final_positions = np.copy(positions)
                break
            else: 
                forces = self.compute_apf_forces(positions, goals_gaussians)
                step = self.step_size * forces
                norm = np.linalg.norm(step, axis=1, keepdims=True)
                scaling = np.minimum(1.0, self.max_step / norm)
                step = step * scaling
                positions += self.damping*step
                trajectories.append(np.copy(positions))
                iter_count += 1

            if iter_count > self.max_apf_iterations:
                raise RuntimeError(f"APF failed to converge: max_iter_num exceeded at timestep {t}") 

        # Ensure trajectories always has at least one frame
        if len(trajectories) == 0:
            dummy = np.expand_dims(final_positions, axis=1)
            return dummy, final_positions
        trajectories = np.stack(trajectories, axis=1)
        if trajectories.shape[1] > 1:
            return trajectories[:, 1:, :], final_positions
        else:
            return trajectories, final_positions

    def compute_apf_forces(self, pos, gaussian_goals):
        forces = np.zeros_like(pos)
        for i in range(len(pos)):
            # Attractive force
            mean_diff = gaussian_goals[i].mean - pos[i]
            Sigma_inv = np.linalg.inv(gaussian_goals[i].covariance)
            f_attr = self.k_attr * (Sigma_inv @ mean_diff)

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
        """
        segments = []

        # initialize randomly
        start_gaussians = [self.gaussian_nodes[gaussian_path[0]] for gaussian_path in self.gaussian_paths]
        positions = np.vstack([np.random.multivariate_normal(mean=g_node.mean, cov=g_node.covariance)
                     for g_node in start_gaussians])
        
        for t in range(len(self.gaussian_paths[0])):
            trajectories, final_positions = self.update(t, positions)
            positions = final_positions 
            segments.append(trajectories)

        return np.concatenate(segments, axis=1)

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import numpy as np

    from st_gaussian_prm.solvers.micro.apf import APF

    num_agents = 100

    # generate per agent trajectory

    means = np.array([(0., 0.), (1., 0.), (1.5, 1), (2., 2.)])
    covs = [np.eye(2) for _ in means]

    samples = np.stack([np.random.multivariate_normal(mean, cov, size=num_agents) 
               for mean, cov in zip(means, covs)])

    paths = np.einsum("ijk->jik", samples)

    print("paths", paths.shape)
    apf_config = {
    }
    trajectories = APF(paths, **apf_config).solve()
