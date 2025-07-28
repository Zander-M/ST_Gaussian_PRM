"""
    Macro Solver base class. Solvers can be added to register for
    
"""

from .macro_solver_base import MacroSolverBase

# Initialize Solver Registry

SOLVER_REGISTRY = {}

def register_solver(name):
    def decorator(cls):
        SOLVER_REGISTRY[name] = cls
        return cls
    return decorator

from .teg import TEGSolver 
from .drrt import DRRTSolver
from .drrt_star import DRRTStarSolver
from .lp import LPSolver
from .formation_control import FormationControlSovler

# Get solvers

def get_solver_class(name):
    try:
        return SOLVER_REGISTRY[name]
    except KeyError:
        raise ValueError(f"Solver {name} not found. Available: {list(SOLVER_REGISTRY)}")