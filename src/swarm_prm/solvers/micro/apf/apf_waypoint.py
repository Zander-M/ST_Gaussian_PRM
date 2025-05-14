"""
    APF planner with waypoints. Takes per-agent time-indexed paths as input,
    output updated trajectories using APF.
"""

from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Circle

import numpy as np
from shapely.geometry import Point
from shapely.ops import nearest_points

from swarm_prm.utils.spatial_hash import SpatialHash

class APFWaypointSolver:
    def __init__(self, roadmap, macro_solution, agent_radius, 
                 velocity_cap=1.0, obs_thresh=1, max_dist=10, 
                 attract_coeff=0.2, repel_coeff=0.5, agent_repel_coeff=0.5,  
                 max_timestep_iter=100, reach_dist=3,
                 hash_grid_size = 3):

        self.roadmap = roadmap
        self.macro_solution = macro_solution
        self.agent_radius = agent_radius
        self.num_agent = len(self.macro_solution)
        self.max_timestep_iter = max_timestep_iter
        self.reach_dist = reach_dist
        self.hash_grid_size = hash_grid_size
        
        # APF parameters 
        self.velocity_cap = velocity_cap
        self.attract_coeff = attract_coeff
        self.repel_coeff = repel_coeff
        self.agent_repel_coeff = agent_repel_coeff
        self.obs_thresh = obs_thresh
        self.max_dist = max_dist

        self.solution_trajectory = []
        self.solution_length = 0
        self.spatial_hash = SpatialHash(self.hash_grid_size)

    def update(self, timestep):
        """
            Update trajectory per timestep
        """

        # parallelized agent update
        pos = np.array([self.solution_trajectory[i][-1] for i in range(self.num_agent)])
        goal = np.array([self.macro_solution[i][timestep] for i in range(self.num_agent)])
        reach_timestep_goal = np.zeros(self.num_agent, dtype=bool)
        timestep_iter = 0

        while not all(reach_timestep_goal) \
            and timestep_iter < self.max_timestep_iter * self.num_agent:

            # Attractive Force
            f_att = (goal-pos) * self.attract_coeff

            # Repelling Force
            f_rep = np.zeros_like(pos)
            self.spatial_hash.build(pos)

            for i in range(self.num_agent):
                p_i = pos[i]
                rep = np.zeros(2)

                # Obstacle repulsion
                for obs in self.roadmap.obstacles:
                    dist = obs.get_dist(p_i)
                    if dist < self.obs_thresh:
                        dist = max(dist, 1e-6) # lowerbound
                        obs_point = nearest_points(obs.geom, Point(p_i))[1]
                        f = (p_i - obs_point.coords[0]) / np.linalg.norm(p_i - obs_point.coords[0])
                        rep += self.repel_coeff * f * (1/dist - 1/self.obs_thresh) ** 2
                    
                # Agent repulsion
                neighbors = self.spatial_hash.query_radius(p_i, self.obs_thresh)
                for j in neighbors:
                    if j == i:
                        continue
                    p_j = pos[j]
                    dist = np.linalg.norm(p_i - p_j) - self.agent_radius
                    if dist < self.obs_thresh:
                        dist = max(dist, 1e-6)
                        f = (p_i - p_j) / np.linalg.norm(p_i - p_j)
                        rep += self.agent_repel_coeff * f * (1/dist - 1/self.obs_thresh) ** 2
                
                f_rep[i] = rep

                # Update all agents with velocity cap
                f_total = f_att + f_rep 
                f_total = f_total / np.linalg.norm(f_total) # normalize?
                norms = np.linalg.norm(f_total, axis=1)

                new_pos = self.solution_trajectory[agent_idx][-1] + f_total * self.step_size 
                if self.roadmap.is_radius_collision(new_pos, self.agent_radius):
                    new_pos = self.solution_trajectory[agent_idx][-1] # wait if new position collide with obstacle

                # Test reach condition
                if np.linalg.norm(self.macro_solution[agent_idx][timestep] - new_pos) < self.reach_dist:
                    reach_timestep_goal[agent_idx] = True
                
                self.solution_trajectory[agent_idx].append(new_pos)
                self.spatial_hash.update_position(agent_idx, new_pos)

            timestep_iter += 1
            # verify reaching condition

        if timestep_iter == self.max_timestep_iter * self.num_agent:
            return False 
        return True
        
        # assert timestep_iter < self.max_timestep_iter, "Cannot find path to next goal within {self.max_timestep_iter} timesteps."

    def get_f_att(self, agent_idx, timestep):
        """
            Compute attractive force for the agent at a timestep
        """
        if self.attract_strategy == "UNIFORM":
            """
                Attract agents to next waypoint.
            """
            pos = self.solution_trajectory[agent_idx][-1]
            return (self.macro_solution[agent_idx][timestep]-pos) * self.attract_coeff

        elif self.attract_strategy == "GAUSSIAN":
            """
                Attract agents based on 2D Gaussian distribution.
                Use Gaussian path.
                TODO: implement this
            """
            pass
        return np.array([0, 0])

    def get_f_rep(self, agent_idx):
        """
            Compute repelling force for the agent at a timestep.
        """
        f_rep = np.array([0, 0], dtype=np.float64)
        pos = self.solution_trajectory[agent_idx][-1]

        # repulsive force from obstacles
        for obs in self.roadmap.obstacles:
            dist = obs.get_dist(pos)
            if dist < self.obs_thresh:
                dist = max(1e-6, dist) # threshold
                obs_point =  nearest_points(obs.geom, Point(pos))[1]
                f = (pos - obs_point.coords[0]) / np.linalg.norm((pos - obs_point.coords[0])) # normalized direction vector
                f_rep += self.repel_coeff * f * (1/dist - 1/self.obs_thresh) ** 2

        # repulsive force from agents
        neighboring_agents = self.spatial_hash.query_radius(pos, self.obs_thresh)
        for i in neighboring_agents:
            if i == agent_idx:
                continue
            agent_pos = self.solution_trajectory[i][-1]
            dist = np.linalg.norm(pos-agent_pos)-self.agent_radius
            if dist < self.obs_thresh:
                dist = max(1e-6, dist) # threshold
                f = (pos - agent_pos) / np.linalg.norm(pos - agent_pos)
                f_rep += self.agent_repel_coeff * f * (1/dist - 1/self.obs_thresh) ** 2
        return f_rep
    
    def get_solution(self):
        """
            Get total solution
        """
        for t in range(len(self.macro_solution[0])):
            if not self.update(t):
                print("Early Termination at macro timestep {}".format(t))
                break
        print("Found solution")

        # padding solutions
        self.solution_length = 0 
        for traj in self.solution_trajectory:
            self.solution_length = max(len(traj), self.solution_length)
        
        padded_solution = []

        for traj in self.solution_trajectory:
            wait_len = self.solution_length - len(traj)
            traj += [traj[-1]] * wait_len
            padded_solution.append(traj)

        self.solution_trajectory = padded_solution
        return self.solution_trajectory

## Visualization

    def animate_solution(self, fig, ax):
        """
            Visualize solution trajectory provided instance
        """
        
        agents = []

        def init():
            for agent in agents:
                agent.remove()
            agents.clear()
            return []

        def update(frame):
            for agent in agents:
                agent.remove()
            agents.clear()
            cmap = plt.get_cmap("tab10")
            locs = [traj[frame] for traj in self.solution_trajectory]
            for i, loc in enumerate(locs):
                agent, = ax.add_patch(Circle((loc[0, 0], loc[0, 1]), radius=self.agent_radius, color=cmap(i%10)))
                agents.append(agent)
            return agents

        anim = FuncAnimation(fig, update, frames=self.solution_length, 
                             init_func=init, blit=True, interval=100)
        anim.save("apf_solution.gif", writer='pillow', fps=6)

