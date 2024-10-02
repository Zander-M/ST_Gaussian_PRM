"""
    Artificial potential field solvers for finding solution paths given Gaussian trajectory
    Consider roadmaps with different resolutions
"""
import numpy as np
from shapely.geometry import Point
from shapely.ops import nearest_points

from swarm_prm.solvers.swarm_prm.macro.gaussian_utils import GaussianNode

class APFSingleStepSolver:
    """
        Update one timestep for all agents and repeat
    """
    def __init__(self, roadmap, macro_trajectory, agent_radius, solution_timestep, 
                 step_size=0.1, att_coeff=0.2, rep_coeff=0.5, obs_thresh=1, 
                 max_timestep_iter=100, reach_thresh=0.1,
                 attract_strategy="UNIFORM", ordering_strategy="RANDOM"):
        self.roadmap = roadmap
        self.macro_trajectory = macro_trajectory
        self.agent_radius = agent_radius
        self.num_agent = len(self.macro_trajectory)
        self.solution_timestep = solution_timestep
        self.max_timestep_iter = max_timestep_iter
        self.goal_thresh = reach_thresh
        self.attract_strategy = attract_strategy
        self.ordering_strategy= ordering_strategy 
        
        # Tune parameters here
        self.step_size = step_size # step size
        self.att_coeff = att_coeff
        self.rep_coeff = rep_coeff
        self.obs_thresh = obs_thresh

        self.solution_trajectory = []
        
        # adding starting positions with noise
        for agent_idx in range(self.num_agent):
            self.solution_trajectory.append([self.macro_trajectory[agent_idx][0] + np.random.rand(1, 2)])

    def update(self, timestep):
        """
            Update trajectory per timestep
        """
        order = [i for i in range(len(self.macro_trajectory))]
        if self.ordering_strategy == "SEQUENTIAL": # Nothing needed to do
            pass
        elif self.ordering_strategy == "RANDOM": # Update agent in random order
            np.random.shuffle(order)
        else:
            assert False, "Unimplemented ordering sequence"

        reach_timestep_goal = [False] * self.num_agent
        timestep_iter = 0
        while False in reach_timestep_goal \
            and timestep_iter < self.max_timestep_iter * self.num_agent:
            for agent_idx in order:
                f_att = self.get_f_att(agent_idx, timestep)
                f_rep = self.get_f_rep(agent_idx)
                f_total = f_att + f_rep 
                f_total = f_total / np.linalg.norm(f_total) # normalize?
                new_pos = self.solution_trajectory[agent_idx][-1] + f_total * self.step_size 
                if self.roadmap.is_radius_collision(new_pos[0], self.agent_radius):
                    new_pos = self.solution_trajectory[agent_idx][-1] # wait if new position collide with obstacle
                self.solution_trajectory[agent_idx].append(new_pos)
            timestep_iter += 1

        if timestep_iter < self.max_timestep_iter:
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
            return (self.macro_trajectory[agent_idx][timestep]-pos) * self.att_coeff

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
        f_rep = np.array([[0, 0]], dtype=np.float64)
        pos = self.solution_trajectory[agent_idx][-1]

        # repulsive force from obstacles
        for obs in self.roadmap.obstacles:
            dist = obs.get_dist(pos)
            if dist < self.obs_thresh:
                dist = max(1e-6, dist) # threshold
                obs_point =  nearest_points(obs.geom, Point(pos))[1]
                f_rep += self.rep_coeff * (pos - obs_point.coords) * (1/dist**2)

        # repulsive force from agents
        for i in range(self.num_agent):
            agent_pos = self.solution_trajectory[i][-1]
            dist = np.linalg.norm(pos-agent_pos)-self.agent_radius
            if dist < self.obs_thresh:
                dist = max(1e-6, dist) # threshold
                f_rep += self.rep_coeff * (pos - agent_pos)  * (1/dist**2)
        return f_rep
    
    def get_solution(self):
        """
            Get total solution
        """
        for t in range(self.solution_timestep):
            if not self.update(t):
                print("Early Termination")
                return self.solution_trajectory
        print("Found solution")
        return self.solution_trajectory

class APFPPSolver:
    """
        Plan one agent at a time (Prioritized Planning style)
    """
    def __init__(self, roadmap, macro_trajectory, agent_radius, solution_timestep, 
                 step_size=0.1, att_coeff=0.2, rep_coeff=0.5, obs_thresh=1, 
                 max_timestep_iter=100, reach_thresh=0.1,
                 attract_strategy="UNIFORM", ordering_strategy="RANDOM"):
        self.roadmap = roadmap
        self.macro_trajectory = macro_trajectory
        self.agent_radius = agent_radius
        self.num_agent = len(self.macro_trajectory)
        self.solution_timestep = solution_timestep
        self.max_timestep_iter = max_timestep_iter
        self.goal_thresh = reach_thresh
        self.attract_strategy = attract_strategy
        self.ordering_strategy= ordering_strategy 
        
        # Tune parameters here
        self.step_size = step_size # step size
        self.att_coeff = att_coeff
        self.rep_coeff = rep_coeff
        self.obs_thresh = obs_thresh

        self.solution_trajectory = []
        
        # adding starting positions with noise
        for agent_idx in range(self.num_agent):
            self.solution_trajectory.append([self.macro_trajectory[agent_idx][0] + np.random.rand(1, 2)])

    def plan(self, agent_idx):
        """
            Find agent plan based on agent index
        """

        reach_timestep_goal = [False] * self.num_agent
        timestep_iter = 0
        while False in reach_timestep_goal \
            and timestep_iter < self.max_timestep_iter * self.num_agent:
            for agent_idx in order:
                f_att = self.get_f_att(agent_idx, timestep)
                f_rep = self.get_f_rep(agent_idx)
                f_total = f_att + f_rep 
                f_total = f_total / np.linalg.norm(f_total) # normalize?
                new_pos = self.solution_trajectory[agent_idx][-1] + f_total * self.step_size 
                if self.roadmap.is_radius_collision(new_pos[0], self.agent_radius):
                    new_pos = self.solution_trajectory[agent_idx][-1] # wait if new position collide with obstacle
                self.solution_trajectory[agent_idx].append(new_pos)
            timestep_iter += 1

        if timestep_iter < self.max_timestep_iter:
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
            return (self.macro_trajectory[agent_idx][timestep]-pos) * self.att_coeff

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
        f_rep = np.array([[0, 0]], dtype=np.float64)
        pos = self.solution_trajectory[agent_idx][-1]

        # repulsive force from obstacles
        for obs in self.roadmap.obstacles:
            dist = obs.get_dist(pos)
            if dist < self.obs_thresh:
                dist = max(1e-6, dist)
                obs_point =  nearest_points(obs.geom, Point(pos))[1]
                f_rep += self.rep_coeff * (pos - obs_point.coords) * (1/dist**2)

        # repulsive force from agents
        for i in range(self.num_agent):
            agent_pos = self.solution_trajectory[i][-1]
            dist = np.linalg.norm(pos-agent_pos)-self.agent_radius
            if dist < self.obs_thresh:
                dist = max(1e-6, dist)
                f_rep += self.rep_coeff * (pos - agent_pos)  * (1/dist**2)
        return f_rep
    
    def get_solution(self):
        """
            Get total solution
        """
        agent_idx = [i for i in range(self.num_agent)]
        for idx in agent_idx:
            if not self.plan(idx):
                print("Early Termination")
                return self.solution_trajectory
        return self.solution_trajectory