import numpy as np
import itertools
import networkx as nx
import pandas as pd
import argparse
from grid_utils.grid import get_grid, draw_grid

### PARSE ARGUMENTS ###
parser = argparse.ArgumentParser(description='Construct grid for solving TSP and SP problems')
parser.add_argument('--n-samples', type=int, default=5000, help='Number of samples to take for clustering')
parser.add_argument('--min-dist', type=float, default=0.003, help='Distance threshold for clustering')
args = parser.parse_args()
#######################

# Construct, draw and save grid.
grid_data = get_grid(args.n_samples, args.min_dist)
grid_data = grid_data[np.all(grid_data != 0, axis=1), :]
draw_grid(grid_data)

# Create network representation of grid.
network = nx.Graph()
network.add_nodes_from(range(grid_data.shape[0]))
    
# Go over grid and add nodes with corresponding longitude and latitude data.
for idx in range(grid_data.shape[0]):
    network.add_node(idx, latlon=tuple(grid_data[idx, :]))

# Save constructed network.
nx.write_gpickle(network, './data/grid_data/grid_network.gpickle')

