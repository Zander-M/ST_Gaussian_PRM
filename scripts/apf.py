import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# === PARAMETERS ===
NUM_AGENTS = 3
GOAL_TOL = 0.1
STEP_SIZE = 0.05
K_ATTR = 1.0      # Attractive potential gain
K_REP = 0.0002       # Repulsive potential gain
REPULSION_RADIUS = 0.15

# === INITIAL CONDITIONS ===
np.random.seed(0)
positions = np.array([[1, 0.1], [2, 0], [1.5, 0]], dtype=np.float64)
goals = np.array([[1, 0], [1, 0], [1, 0]], dtype=np.float64)

# === APF PLANNING FUNCTION ===
def compute_apf_forces(pos, goals):
    forces = np.zeros_like(pos)
    for i in range(len(pos)):
        # Attractive force
        diff = goals[i] - pos[i]
        f_attr = K_ATTR * diff

        # Repulsive force from other agents
        f_rep = np.zeros(2)
        for j in range(len(pos)):
            if i == j:
                continue
            diff_ij = pos[i] - pos[j]
            dist = np.linalg.norm(diff_ij)
            if dist < REPULSION_RADIUS and dist > 1e-4:
                f_rep += K_REP * (1.0 / dist - 1.0 / REPULSION_RADIUS) * (diff_ij / (dist**3))

        forces[i] = f_attr + f_rep
    return forces

# === VISUALIZATION SETUP ===
fig, ax = plt.subplots()
scat = ax.scatter([], [], c='blue')
goal_plot = ax.scatter(goals[:, 0], goals[:, 1], c='green', marker='x')
ax.set_xlim(-1, 3)
ax.set_ylim(-1, 3)
ax.set_aspect('equal')

# === SIMULATION LOOP ===
def update(frame):
    global positions
    forces = compute_apf_forces(positions, goals)
    positions += STEP_SIZE * forces
    scat.set_offsets(positions)
    return scat,

ani = animation.FuncAnimation(fig, update, frames=200, interval=100, blit=True)
plt.title("3-Agent APF Navigation")
plt.show()
