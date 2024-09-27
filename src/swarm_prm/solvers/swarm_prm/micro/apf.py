"""
    Artificial potential field solvers for finding solution paths given Gaussian trajectory
    Consider roadmaps with different resolutions
"""
import numpy as np

from swarm_prm.macro.gaussian_utils import GaussianNode

class APFSingleStepSolver:
    """
        Update one timestep for all agents and repeat
    """
    def __init__(self, map, macro_trajectory, agent_radius, solution_timestep, 
                 step_size=0.1, att_coeff=0.2, rep_coeff=0.5, 
                 max_timestep_iter=100, reach_thresh=0.1):
        self.map = map
        self.macro_trajectory = macro_trajectory
        self.agent_radius = agent_radius
        self.num_agent = len(self.macro_trajectory)
        self.solution_timestep = solution_timestep
        self.max_timestep_iter = max_timestep_iter
        self.goal_thresh = reach_thresh
        
        # Tune parameters here
        self.step_size = step_size # step size
        self.att_coeff = att_coeff
        self.rep_coeff = rep_coeff

        self.solution_trajectory = []
        
        # adding starting 
        for agent_idx in range(self.num_agent):
            self.solution_trajectory.append([self.macro_trajectory[agent_idx][0]])

    def update(self, timestep, strategy="RANDOM"):
        """
            Update trajectory per timestep
        """
        order = [i for i in range(len(self.macro_trajectory))]
        if strategy == "SEQUENTIAL": # Nothing needed to do
            pass
        elif strategy == "RANDOM": # Update agent in random order
            np.random.shuffle(order)
        else:
            assert False, "Unimplemented planning sequence"

        reach_timestep_goal = [False] * self.num_agent
        timestep_iter = 0
        while False in reach_timestep_goal and timestep_iter < self.max_timestep_iter:
            for agent_idx in order:
                f_att = self.get_f_att(agent_idx, timestep)
                f_rep = self.get_f_rep(agent_idx, timestep)
                total_f = f_att * self.att_coeff + f_rep * self.rep_coeff
                new_pos = self.solution_trajectory[agent_idx][-1] + total_f * self.step_size 
                self.solution_trajectory[agent_idx].append(new_pos)
            timestep_iter += 1
        assert timestep_iter < self.max_timestep_iter, "Cannot find path to next goal within {self.max_timestep_iter} timesteps."

    def get_f_att(self, agent_idx, timestep):
        """
            Compute attractive force for the agent at a timestep
            TODO: compute attractive force
        """
        return np.array([0, 0])

    def get_f_rep(self, agent_idx, timestep):
        """
            Compute repelling force for the agent at a timestep
            TODO: compute repelling force
        """
        return np.array([0, 0])

class APFPPSolver:
    """
        Plan one agent at a time (Prioritized Planning style)
    """
    def __init__(self, map, trajectory, step_size=0.1):
        self.map = map
        self.trajectory = trajectory
        self.step_size = step_size