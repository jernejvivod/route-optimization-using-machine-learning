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
    
    # TODO: add learned heuristic.
    # Compute geodesic distance from node to goal.
    return dist_func(node, goal)


def reconstruct_path(current, came_from):
    path = [current]
    while current in came_from:
        current = came_from[current]
        path.insert(0, current)
    return path



def a_star(network, start, goal, dist_func):
    
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
        
        # Check if goal.
        if current == goal:
            return reconstruct_path(current, came_from)
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
    
    # If goal node not found, return None.
    return None


if __name__ == '__main__':

    ### PARSE ARGUMENTS ###
    parser = argparse.ArgumentParser(description='Approximate solution to TSP using particle swarm optimization.')
    parser.add_argument('--num-nodes', type=int, default=30, help='Number of nodes to use')
    parser.add_argument('--dist-func', type=str, default='geodesic', choices=['geodesic', 'learned'], 
            help='Distance function to use')
    parser.add_argument('--prediction-model', type=str, default='gboosting', choices=['gboosting', 'rf'], 
            help='Prediction model to use for learned distance function')
    args = parser.parse_args()
    #######################

    # Parse problem network.
    network = nx.read_gpickle('./data/grid_data/grid_network.gpickle')
    
    # Number of nodes to remove from network.
    to_remove = network.number_of_nodes() - args.num_nodes
    
    # Remove randomly sampled nodes to get specified number of nodes.
    network.remove_nodes_from(random.sample(list(network.nodes), to_remove))
    
    num_edges = 50
    # Add edges to network.
    while network.number_of_edges() < num_edges:
        network.add_edge(*random.sample(list(network.nodes()), 2))
    
    connected_components = list(map(list, nx.connected_components(network)))
    if len(connected_components) > 1:
        for idx in range(len(connected_components)-1):
            n1 = random.choice(connected_components[idx])
            n2 = random.choice(connected_components[idx+1])
            network.add_edge(n1, n2)


    # Get distance function.
    dist_func = get_dist_func(network, which=args.dist_func, prediction_model=args.prediction_model)
    

    start = random.choice(list(network.nodes()))
    goal = random.choice(list(network.nodes()))
    
    # Get solution using partice swarm.
    path = a_star(network, start, goal, dist_func)
    import pdb
    pdb.set_trace()


    # Save list of edge lists for animation.
    np.save('./results/edgelists/edgelist_tsp_ps.npy', list(map(np.vstack, edgelists)))
    nx.write_gpickle(network, './results/networks/network_tsp_ps.gpickle')
   
    # Print best solution fitness.
    print('Fitness of best found solution: {0:.3f}'.format(solution_fitness))
    
    # Print initial best fitness.
    print('Fitness of initial solution: {0:.3f}'.format(initial_fitness))

    # Print increase in fitness.
    print('Fitness value improved by: {0:.3f}%'.format(100*initial_fitness/solution_fitness))
