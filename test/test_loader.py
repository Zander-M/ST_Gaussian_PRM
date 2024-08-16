from envs.loader import MapLoader

if __name__ == "__main__":
    fname = "../data/envs/map_2.yaml"
    loader = MapLoader(fname)
    loader.visualize("test_map")
    map_instance = loader.get_map()
