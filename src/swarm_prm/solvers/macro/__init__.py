"""
    Macro Solver base class. Solvers can be added to register for
    
"""

from .macro_solver_base import MacroSolverBase

SOLVER_REGISTRY = {}

def register_solver(name):
    def decorator(cls):
        SOLVER_REGISTRY[name] = cls
        return cls
    return decorator

# Initialize Solver Registry

from .teg import TEGSolver
from .teg_mcf import TEGMCFSolver
from .teg_node_constraint import TEGNodeConstraintSolver
from .drrt import DRRTSolver
from .drrt_star import DRRTStarSolver
from .lp import LPSolver
from .formation_control import FormationControlSovler