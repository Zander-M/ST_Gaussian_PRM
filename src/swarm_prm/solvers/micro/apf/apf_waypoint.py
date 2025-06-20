"""
    APF planner with waypoints. Takes per-agent time-indexed paths as input,
    output updated trajectories using APF.
"""

from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Circle

import numpy as np
from scipy.spatial import KDTree
from shapely.geometry import Point
from shapely.ops import nearest_points

from swarm_prm.utils.spatial_hash import SpatialHash

class APFWaypointSolver:
    def __init__(self, roadmap, macro_solution, agent_radius, **apf_config):

        self.roadmap = roadmap # for getting obstacles only
        self.macro_solution = macro_solution
        self.agent_radius = agent_radius
        self.num_agent = len(self.macro_solution)
        self.max_timestep_iter = apf_config.get("max_timestep_iter", 100)# number of steps to make per state transition
        self.reach_dist = apf_config.get("reach_dist", 3)# reach threshold
        
        # APF parameters 
        self.agent_repel_coeff = apf_config.get("agent_repel_coeff", 0.5) 
        self.attract_coeff = apf_config.get("attract_coeff", 0.2)
        self.obs_thresh = apf_config.get("obs_thresh", 1) 
        self.repel_coeff = apf_config.get("repel_coeff", 0.5)
        self.step_size = apf_config.get("step_size", 1.0) 
        self.velocity_cap = apf_config.get("velocity_cap", 1.0)

        self.solution_trajectory = [[] for _ in range(self.num_agent)]
        self.solution_length = 0

    def update(self, timestep):
        """
            Update trajectory per timestep
        """

        # parallelized agent update
        pos = np.array([self.solution_trajectory[i][-1] for i in range(self.num_agent)])
        goals = np.array([self.macro_solution[i][timestep] for i in range(self.num_agent)])
        reach_timestep_goal = np.zeros(self.num_agent, dtype=bool)
        timestep_iter = 0

        while not all(reach_timestep_goal) \
            and timestep_iter < self.max_timestep_iter * self.num_agent:

            # Attractive Force
            f_att = (goals-pos) * self.attract_coeff

            # Repelling Force
            f_rep = np.zeros_like(pos)
            tree = KDTree(pos)
            neighbors_list = tree.query_ball_point(pos, r=self.obs_thresh)

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
                        rep += self.repel_coeff * f * (1/dist - 1/(self.obs_thresh+self.agent_radius*2)) ** 2
                    
                # Agent repulsion
                for j in neighbors_list[i]:
                    if j == i:
                        continue
                    p_j = pos[j]
                    dist = np.linalg.norm(p_i - p_j) - self.agent_radius*2
                    if dist < (self.obs_thresh + self.agent_radius*2):
                        dist = max(dist, 1e-6)
                        f = (p_i - p_j) / np.linalg.norm(p_i - p_j)
                        rep += self.agent_repel_coeff * f * (1/dist - 1/(self.obs_thresh+self.agent_radius*2)) ** 2
                
                f_rep[i] = rep

            # Update all agents with velocity cap
            f_total = f_att + f_rep
            norms = np.linalg.norm(f_total, axis=1, keepdims=True)
            norms = np.maximum(norms, 1e-6)  # prevent divide-by-zero
            scaling = np.minimum(1.0, self.velocity_cap / norms)
            f_total = f_total * scaling

            new_pos = pos + f_total * self.step_size

            new_pos = pos + f_total*self.step_size

            # Handle obstacle collisions
            for i in range(self.num_agent):
                if self.roadmap.is_radius_collision(new_pos[i], self.agent_radius):
                    new_pos[i] = pos[i]  # wait if new position collide with obstacle

                # Test reach condition
                if np.linalg.norm(goals[i]-new_pos[i]) < self.reach_dist:
                    reach_timestep_goal[i] = True
            
                # Verify reaching condition
                self.solution_trajectory[i].append(new_pos[i])

            pos = new_pos
            timestep_iter += 1

        if timestep_iter == self.max_timestep_iter * self.num_agent:
            return False 
        return True
    
    def reach_state_check(self):
        """
            Check if state is reached.
            TODO: implement this
        """
        
    def get_solution(self):
        """
            Get total solution
        """

        # initialize starts
        for i in range(self.num_agent):
            self.solution_trajectory[i].append(self.macro_solution[i][0])

        for t in range(1, len(self.macro_solution[0])):
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

    def animate_solution(self, paths, fig, ax, fig_path="."):
        """
            Visualize solution trajectory provided instance
        """
        agents = []
        cmap = plt.get_cmap("tab10")
        
        for i in range(len(paths)):
            loc = paths[i][0]
            circle = Circle((loc[0], loc[1]), radius=self.agent_radius, color=cmap(i % 10))
            agents.append(circle)
            ax.add_patch(circle)

        def init():
            return agents

        def update(frame):
            for agent, traj in zip(agents, paths):
                agent.set_center(traj[frame])
            return agents

        anim = FuncAnimation(fig, update, frames=self.solution_length, 
                             init_func=init, blit=True, interval=100)
        anim.save(f"{fig_path}/apf_solution.mp4", writer='ffmpeg', fps=6)

