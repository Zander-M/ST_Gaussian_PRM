
import pickle
from swarm_prm.solvers.macro.drrt import DRRT


with open("notebooks/solutions/drrt_multi_agent_roadmap.pkl", "rb") as f:
    gaussian_prm = pickle.load(f)

num_agents = 2
agent_radius = 1

drrt_solver = DRRT(gaussian_prm, num_agents, agent_radius)
path, cost= drrt_solver.get_solution()

