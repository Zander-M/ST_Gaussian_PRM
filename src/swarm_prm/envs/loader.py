"""
    Load map and test instances
"""

from matplotlib import pyplot as plt
import yaml

from swarm_prm.envs.map_objects import *


class MapLoader:
    """
        Return map object given yaml file
    """
    def __init__(self, fname) -> None:
        self.fname = fname
        with open(fname, "r") as f:
            data = f.read()
        map_dict = yaml.load(data, Loader=yaml.SafeLoader)
        map_info = Map(map_dict["width"], map_dict["height"])
        for obs in map_dict["obstacles"]:
            map_info.add_obstacle(CircleObstacle([obs["x"], obs["y"]], obs["radius"]))
        self.map_info = map_info
    
    def get_map(self):
        """
            Return Map
        """
        return self.map_info, self.fname

    def visualize(self, fname):
        """
            Visualize instance
        """
        fig, ax = plt.subplots()
        ax.set_xlim([0, self.map_info.width])
        ax.set_ylim([0, self.map_info.height])
        for obs in self.map_info.get_obstacles():
            x, y = obs.get_pos()
            # ax.plot(x, y, 'ro', markersize=3)
            ax.add_patch(plt.Circle((x, y), radius=obs.radius, color="gray"))
        
        ax.set_aspect('equal')
        plt.savefig("{}.png".format(fname))
        
        
class InstanceLoader:
    """
        Return instance object given yaml file
    """
    def __init__(self, fname) -> None:
        pass

    def visualize(self):
        """
            Visualize instance
        """
        pass

if __name__ == "__main__":
    fname = "../../data/envs/map_3.yaml"
    map_info = MapLoader(fname)
    map_info.visualize("test_map")