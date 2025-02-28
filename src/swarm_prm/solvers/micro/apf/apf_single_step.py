"""
    Artificial potential field solvers for finding solution paths given Gaussian trajectory
    Consider roadmaps with different resolutions
"""
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Circle

import numpy as np
from shapely.geometry import Point
from shapely.ops import nearest_points

from swarm_prm.utils.gaussian_utils import GaussianNode
from swarm_prm.utils.spatial_hash import SpatialHash

class APFSingleStepSolver:
    """
        Update one timestep for all agents and repeat
    """
    def __init__(self, roadmap, macro_trajectory, agent_radius, macro_timestep, 
                 step_size=0.1, obs_thresh=1, max_dist=8, max_init_attempt=100,
                 attract_coeff=0.2, repel_coeff=0.5, agent_repel_coeff=0.5,  
                 max_timestep_iter=100, reach_dist=3,
                 hash_grid_size = 10,
                 attract_strategy="UNIFORM", ordering_strategy="RANDOM"):
        self.roadmap = roadmap
        self.macro_trajectory = macro_trajectory
        self.agent_radius = agent_radius
        self.num_agent = len(self.macro_trajectory)
        self.macro_timestep = macro_timestep
        self.max_timestep_iter = max_timestep_iter
        self.reach_dist = reach_dist
        self.hash_grid_size = hash_grid_size
        self.attract_strategy = attract_strategy
        self.ordering_strategy= ordering_strategy 
        
        # APF parameters 

        self.max_init_attempt = max_init_attempt
        self.step_size = step_size # step size
        self.attract_coeff = attract_coeff
        self.repel_coeff = repel_coeff
        self.agent_repel_coeff = agent_repel_coeff
        self.obs_thresh = obs_thresh
        self.max_dist = max_dist

        self.solution_trajectory = []
        self.solution_length = 0
        self.spatial_hash = SpatialHash(self.hash_grid_size)
        
        self.initailize_starts()

    def initailize_starts(self):
        """
            Initialize agent locations based on start locations
        """
        # adding starting positions with noise
        points = []
        for agent_idx in range(self.num_agent):
                r = np.random.uniform(self.obs_thresh, self.max_dist)
                theta = np.random.uniform(0, 2 * np.pi)

                pt = self.macro_trajectory[agent_idx][0] \
                    + np.array([np.cos(theta) * r, np.sin(theta) *r])

                self.solution_trajectory.append([pt])
                points.append(pt)
                self.spatial_hash.insert(agent_idx, pt)


    def update(self, timestep):
        """
            Update trajectory per timestep
        """
        order = [i for i in range(len(self.macro_trajectory))]
        if self.ordering_strategy == "SEQUENTIAL": # Nothing needed to do
            pass
        elif self.ordering_strategy == "RANDOM": # Update agent in random order
            np.random.shuffle(order)
        elif self.ordering_strategy == "GEOMETRY": # Plan agent closest to its goal first
            dist = [np.linalg.norm(self.macro_trajectory[agent_idx][timestep] - self.solution_trajectory[agent_idx][-1])
                    for agent_idx in range(self.num_agent)]
            order = np.argsort(dist)
        else:
            assert False, "Unimplemented ordering sequence"

        reach_timestep_goal = [False] * self.num_agent
        timestep_iter = 0
        while not all (reach_timestep_goal) \
            and timestep_iter < self.max_timestep_iter * self.num_agent:
            for agent_idx in order:
                f_att = self.get_f_att(agent_idx, timestep)
                f_rep = self.get_f_rep(agent_idx)

                f_total = f_att + f_rep 
                f_total = f_total / np.linalg.norm(f_total) # normalize?
                new_pos = self.solution_trajectory[agent_idx][-1] + f_total * self.step_size 

                if self.roadmap.is_radius_collision(new_pos, self.agent_radius):
                    old_pos = self.solution_trajectory[agent_idx][-1]
                    obs = self.roadmap.get_closest_obstacle(new_pos)
                    closest_point = nearest_points(obs.geom, Point(old_pos))[0]
                    v = closest_point.coords[0] - old_pos
                    assert np.linalg.norm(v) > 0, "Distance to obstacle less or equal to 0"
                    norm = (v)/np.linalg.norm(v)
                    new_pos = old_pos + (f_total - norm.dot(f_total) * norm) * self.step_size
                    # new_pos = self.solution_trajectory[agent_idx][-1] # wait if new position collide with obstacle
                                                                    #   FIXIT: keep velocity componet perpendicular to the obstacle norm

                # Test reach condition
                if np.linalg.norm(self.macro_trajectory[agent_idx][timestep] - new_pos) < self.reach_dist:
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
                Attract agents to mean of nodes. Use simple path.
            """
            pos = self.solution_trajectory[agent_idx][-1]
            return (self.macro_trajectory[agent_idx][timestep]-pos) * self.attract_coeff

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
                dist = max(1, dist) # threshold
                obs_point = nearest_points(obs.geom, Point(pos))[0]
                f = (pos - obs_point.coords[0]) / dist # normalized direction vector
                f_rep += self.repel_coeff * f * (1/dist - 1/self.obs_thresh) ** 2

        # repulsive force from agents
        neighbouring_agents = self.spatial_hash.query_radius(pos, self.obs_thresh)
        for i in neighbouring_agents:
            if i == agent_idx:
                continue
            agent_pos = self.solution_trajectory[i][-1]
            dist = np.linalg.norm(pos-agent_pos)-self.agent_radius
            if dist < self.obs_thresh:
                dist = max(1, dist) # threshold
                f = (pos - agent_pos) / np.linalg.norm(pos - agent_pos)
                f_rep += self.agent_repel_coeff * f * (1/dist - 1/self.obs_thresh) ** 2
        return f_rep
    
    def get_solution(self):
        """
            Get total solution
        """
        for t in range(self.macro_timestep):
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

