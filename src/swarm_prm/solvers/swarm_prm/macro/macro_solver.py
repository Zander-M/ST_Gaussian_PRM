"""
    Macro solver for the high level search. Given the start GMM and goal GMM,
    find a spatial-temporal solution to the instance. The plan should observe 
    the max flow capacity of the PRM.
"""

import ortools

from gaussian_prm import GaussianPRM, GaussianInstance

class MacroSolver:
    def __init__(self, gaussian_prm:GaussianPRM, gaussian_instance:GaussianInstance) -> None:
        self.gaussian_prm = gaussian_prm
        self.gaussian_instance = gaussian_instance
        self.solution_paths = [] # st solution pair on prm map

    def a_star_solver(self):
        """
            A star solver for path on the map. 
        """
        pass

    def gaussian_solver(self):
        """
            Solving Gaussian trajectories based on paths
        """
        pass

    def lp_solver(self):
        """
            Finding the path weight assignment for each of the solution paths
            based on transition cost.
        """
        pass

if __name__ == "__main__":
    pass