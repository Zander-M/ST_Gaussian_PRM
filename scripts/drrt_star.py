
import pickle
from swarm_prm.solvers.macro.drrt_star import DRRT_Star

# Single Agent
with open("notebooks/solutions/drrt_single_agent_roadmap.pkl", "rb") as f:
    gaussian_prm = pickle.load(f)

num_agents = 1
agent_radius = 1

drrt_star_solver = DRRT_Star(gaussian_prm, num_agents, agent_radius)
path, cost = drrt_star_solver.get_solution()

# # Multi Agent
# with open("notebooks/solutions/drrt_multi_agent_roadmap.pkl", "rb") as f:
#     gaussian_prm = pickle.load(f)

# num_agents = 2
# agent_radius = 1

# drrt_solver = DRRT_Star(gaussian_prm, num_agents, agent_radius)
# path, cost= drrt_solver.get_solution()

