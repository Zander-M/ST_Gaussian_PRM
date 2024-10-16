from matplotlib import pyplot as plt
import numpy as np
import triangle as tr

from swarm_prm.envs.map import Obstacle

# Circle obstacle
obs = Obstacle([0, 0], "POLYGON", [[0, 0], [0, 5], [4, 8], [9, 8], [9, 6], [8, 5],
                                   [7, 3], [6, 0]])

hole = Obstacle([0, 0], "POLYGON", [[3, 3], [3, 6], [6, 6], [6, 3]])

# obs = Obstacle([0, 0], "CIRCLE", 30)
pts1, seg1, _ = obs.get_edge_segments(2)
pts2, seg2, hole_pos = hole.get_edge_segments(2)

pts = np.vstack([pts1, pts2])
seg = np.vstack([seg1, seg2+seg1.shape[0]])



A = dict(vertices=pts, segments=seg, holes=[hole_pos])
B = tr.triangulate(A, "qpa3")

pts, edges, ray_origin, ray_direct = tr.voronoi(B['vertices'].tolist())
C = dict(vertices=pts, edges=edges,
         ray_origins=ray_origin, ray_directions=ray_direct)

tr.compare(plt, B, C)

plt.show()
# Polygon obstacle

