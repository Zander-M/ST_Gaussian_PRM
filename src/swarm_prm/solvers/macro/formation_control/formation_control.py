"""
    Formation Control solver.
    We assume the formation of the agents follows a 2D Gaussian distribution.
    The intermediate formations are the inscribed ellipse of the intersecting
    polytopes.
"""
from swarm_prm.solvers.macro import MacroSolverBase, register_solver
from swarm_prm.solvers.macro.formation_control import iris

@register_solver("FormationControlSolver")
class FormationControlSovler(MacroSolverBase):
    def init_solver(self, **kwargs):
        """
            Solver specific initialization
        """
        pass

    def solve(self):
        """
            Find solution paths
        """
        return {"success": False}