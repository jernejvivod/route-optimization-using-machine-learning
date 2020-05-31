import numpy as np
import networkx as nx
import argparse
import random
from models.distance import get_dist_func

# NOTE: g-score is the path cost.
# NOTE: f-score is the path cost + heuristic.

def heuristic(network, node, goal, dist_func):
    """
    Heuristic function for estimating distance from specified node to goal node.

    Args:
        network (object): Networkx representation of the network
        node (int): Node for which to compute the heuristic
        goal (int): Goal node
        dist_func (function): Function used to compute distance between two nodes.

    Returns:
        (float): Computed heuristic
    """
    
    # Compute distance from node to goal.
    return dist_func(node, goal)


def reconstruct_path(current, came_from):
    """
    Reconstruct path using last node and dictionary that maps each node on path
    to its predecessor.

    Args:
        current (int): Last node in discovered path
        came_from (dict): Dictionary mapping nodes on path to their predecessors

    Retuns:
        (tuple): Path in the form of a list and the same path encoded in an edge list
    """

    # Initialize path and add last found node.
    path = [current]

    # Reconstruct.
    while current in came_from:
        current = came_from[current]
        path.insert(0, current)
    
    # Construct edgelist.
    edgelist = [(path[idx], path[idx+1]) for idx in range(len(path)-1)]

    # Return path and edge list.
    return path, edgelist



def a_star(network, start, goal, dist_func):
    """
    Perform A* search algorithm to find path from starting node to goal node.

    Args:
        network (object): Networkx representation of the network
        start (int): Starting node for the search
        goal (int): Goal node for the search
        dist_func (function): Distance function mapping two nodes to the
        distance between them.

    Returns:
        (tuple) : list representing the found path, edgelist representation of the found path, 
        list of edge lists that can be used for animating the process.
    """
    
    # Initialize list for storing edge lists (for animation).
    edgelists = []
    
    # Partially apply heuristic with network and goal node.
    h = lambda node: heuristic(network, node, goal, dist_func)
    
    # Initialize array of node IDs.
    node_list = np.array(list(network.nodes()))

    # Initialize set of unvisited nodes
    # with starting node.
    open_set = {start}

    # Initialize dictionary mapping nodes to nodes immediately
    # preceding them on the cheapest path.
    came_from = dict()
    
    # Initialize dictionary mapping nodes to cost of cheapest path from start
    # to the node currently known.
    g_score = dict.fromkeys(node_list, np.inf)
    g_score[start] = 0.0

    # Initialize dictionary mapping nodes to the current best guess as to
    # how short a path from start to finish can be if it goes through n.
    f_score = dict.fromkeys(node_list, np.inf)
    f_score[start] = h(start)
    
    # While set of open nodes is not empty.
    while len(open_set) > 0:

        # Set node in open set with lowest f-score as current node and remove
        # from set of open nodes.
        current = min([(el, f_score[el]) for el in open_set], key=lambda x: x[1])[0]
        open_set.remove(current)
        
        # Reconstruct path from current node and append to list of edge lists.
        _, edgelist = reconstruct_path(current, came_from)
        edgelists.append(edgelist)
        
        # Check if goal.
        if current == goal:
            path, edgelist = reconstruct_path(current, came_from)
            return path, edgelist, edgelists
        else:

            # Go over neighbors of current node.
            for neighbor in network.neighbors(current):

                # Compute tentative g-score and check if better than g-score of node.
                g_score_found = g_score[current] + dist_func(current, neighbor)
                if g_score_found < g_score[neighbor]:

                    # If g-score better, set new g-score and set predecessor to current.
                    g_score[neighbor] = g_score_found
                    came_from[neighbor] = current

                    # Compute f-score of neighbor (cost path + heuristic).
                    f_score[neighbor] = g_score[neighbor] + h(neighbor)

                    # If neighbor not yet explored, add to open set.
                    if neighbor not in open_set:
                        open_set.add(neighbor)
    
    # If goal node not found, return signal values.
    return [], [], []


if __name__ == '__main__':

    ### PARSE ARGUMENTS ###
    parser = argparse.ArgumentParser(description='Approximate solution to TSP using particle swarm optimization.')
    parser.add_argument('--new-network', action='store_true', 
            help='create new network and select random nodes as start and goal nodes.')
    parser.add_argument('--num-nodes', type=int, default=150, help='Number of nodes to use')
    parser.add_argument('--dist-func', type=str, default='geodesic', choices=['geodesic', 'learned'], 
            help='Distance function to use')
    parser.add_argument('--prediction-model', type=str, default='gboosting', choices=['gboosting', 'rf'], 
            help='Prediction model to use for learned distance function')
    parser.add_argument('--n-nearest', type=int, default=3, 
            help='Number of nearest nodes with which to connect a node when constructing the network.')
    args = parser.parse_args()
    #######################

    # Parse problem network.
    if args.new_network:
        network = nx.read_gpickle('./data/grid_data/grid_network.gpickle')
    else:
        network = nx.read_gpickle('./data/grid_data/grid_network_sp.gpickle')

    # Get distance function.
    dist_func = get_dist_func(network, which=args.dist_func, prediction_model=args.prediction_model)
    if args.dist_func == 'learned':
        dist_func.type = 'learned'
    else:
        dist_func.type = 'geodesic'
    
    if args.new_network:
        # If creating a new network.

        # Number of nodes to remove from network.
        to_remove = network.number_of_nodes() - args.num_nodes
        
        # Remove randomly sampled nodes to get specified number of nodes.
        network.remove_nodes_from(random.sample(list(network.nodes), to_remove))
        
        # Connect each node with specified number of its nearest unconnected neighbors.
        for node in network.nodes():

            # Get list of nodes with which the node can connect.
            free_nodes = [n for n in network.nodes() if n != node and not network.has_edge(n, node)]

            # Sort free nodes by distance can choose nodes with which to connect.
            closest_nodes = sorted([(n, dist_func(node, n)) for n in free_nodes], key=lambda x: x[1])
            connect_to = list(map(lambda x: x[0], closest_nodes[:args.n_nearest]))

            # Connect to specified number of closest neighbors.
            for idx in range(len(connect_to)):
                network.add_edge(node, connect_to[idx])
        
        # Get connected components in network.
        connected_components = list(map(list, nx.connected_components(network)))

        # Connect connected components.
        for idx in range(len(connected_components)-1):
            n1 = random.choice(connected_components[idx])
            n2 = random.choice(connected_components[idx+1])
            network.add_edge(n1, n2)

        # Set start and end nodes.
        START_NODE = random.choice(list(network.nodes()))
        dists = sorted([(n, dist_func(START_NODE, n)) for n in network.nodes()], key=lambda x: x[1])
        GOAL_NODE = dists[-1][0]

    else:
        # If using a pre-set network.
        
        # Set pre-set nodes as start and end nodes.
        START_NODE = 450
        GOAL_NODE = 4
        
    # Get solution using A* search.
    path, edgelist, edgelists = a_star(network, START_NODE, GOAL_NODE, dist_func)

    if len(path) == 0 and len(edgelist) == 0 and len(edgelists) == 0:
        print("Goal node unreachable from starting node!")
    else:
        # Save list of edge lists and network for animation.
        np.save('./results/edgelists/edgelist_sp_astar.npy', list(map(np.vstack, filter(lambda x: len(x) > 0, edgelists))))
        nx.write_gpickle(network, './results/networks/network_sp_astar.gpickle')

