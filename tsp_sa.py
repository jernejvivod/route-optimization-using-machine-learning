import numpy as np
import random
import networkx as nx
import more_itertools
import matplotlib.pyplot as plt
import geopy.distance as geodesic_dist
from models.distance import get_dist_func
import argparse


def get_fitness(solution_edgelist):
    """
    Compute fitness value for solution given by edge list.

    Args:
        solution_edgelist (list): List of edges representing the
        solution

    Returns:
        (float): Computed fitness value.
    """

    return sum(map(lambda el : dist_func(*el), solution_edgelist))


def initial_solution(network, strategy):
    """
    Compute initial solution for the TSP problem using specified strategy.

    Args:
        network (object): Networkx representation of the network.
        strategy (str): Method used to construct the initial solution.
        Valid values are 'greedy' and 'random'.

    Returns:
        (tuple): Edge list representing the initial solution and the fitness value
        of the initial solution.
    """
    
    # Start with random node.
    starting_node = random.choice(list(network.nodes()))
    
    # Initialize list for the solution edge list.
    solution_edgelist = []

    # Initialize set for nodes not yet part of path.
    nodes_free = set(np.array(network.nodes()))

    # Set starting node as first node and remove from free nodes.
    src = starting_node
    nodes_free.remove(src)
    
    # While there are nodes not yet part of path, create path.
    while len(nodes_free) > 0:

        # If doing greedy search, connect to closest node. Else connect to
        # random node.
        if strategy == 'greedy':
            dst = min(nodes_free, key=lambda x: dist_func(src, x))
        elif strategy == 'random':
            dst = random.choice(list(nodes_free))

        # Append to solution edge list, remove node from set of free nodes
        # and make curent last node in path the next source node.
        solution_edgelist.append([src, dst])
        nodes_free.remove(dst)
        src = dst
    
    # Append edge for path from last node back to starting node.
    solution_edgelist.append([src, starting_node])

    # Compute fitness for initial configuration.
    fitness = get_fitness(solution_edgelist)
    
    # Return initial solution as edge list and the fitness of the current
    # solution.
    return solution_edgelist, fitness


def anneal(network, max_it=1e4, temp=-1.0, temp_min=1.0e-8, alpha=0.995):
    """
    Approximate solution to TSP problem on given network using simulated annealing.

    Args:
        network (object): Networkx representation of the network
        max_it (int): Maximum iterations to perform
        temp (float): Initial temperature
        temp_min (float): Minimum temperature. The procedure is stopped when the
        temperature falls below this value
        alpha (float): The cooling rate

    Returns:
        (tuple): The edgelist encoding the solution, the current fitness value, the best found fitness value,
        the initial fitness value, the list of accepted edgelists (for animations), the list of temperature values
        for each iteration, the list of fitness values for each iteration
    """

    # Set starting temperature, stopping temperature, alpha, maximum iterations
    # and initialize iterations counter.
    curr_temp = np.sqrt(network.number_of_nodes()) if temp == -1.0 else temp
    it_count = 0

    # Get initial solution using greedy search.
    solution_edgelist, initial_fitness = initial_solution(network, strategy='random')
    solution_perm = np.array([el[0] for el in solution_edgelist])
    solution_length = len(solution_perm)
    
    # Set current fitness and best fitness.
    current_fitness = initial_fitness
    best_fitness = initial_fitness
    
    # List for storing all accepted states (for animating).
    accepted_edgelists = []

    # Lists for storing next temperature and next fitness value.
    temp_vals = []
    fitness_vals = []
    
    # Perform annealing while temperature above minimum and
    # maximum number of iterations not achieved.
    while curr_temp > temp_min and it_count < max_it:
        
        # Append temperature and fitness value
        temp_vals.append(curr_temp)
        fitness_vals.append(current_fitness)

        # Generate neighbor state.
        rnd1 = random.randint(2, solution_length)
        rnd2 = random.randint(1, solution_length - rnd1 + 1)
        solution_perm[rnd2 : (rnd2 + rnd1)] = np.array(list(reversed(solution_perm[rnd2 : (rnd2 + rnd1)])))

        # Evaluate neighbor.
        neighbor_edgelist = list(more_itertools.pairwise(np.hstack((solution_perm, solution_perm[0]))))
        neighbor_fitness = get_fitness(neighbor_edgelist)
        
        # If neighbor fitness is better, accept. If neighbor fitness worse,
        # accept with probability dependent on fitness difference and current temperature.
        if neighbor_fitness <= current_fitness:
            current_fitness = neighbor_fitness
            solution_edgelist = neighbor_edgelist
            accepted_edgelists.append(solution_edgelist)
            if neighbor_fitness < best_fitness:
                best_fitness = neighbor_fitness
        else:
            
            # Compute probability of accepting worse state.
            p_accept = np.exp(-np.abs(neighbor_fitness - current_fitness)/curr_temp)
            if random.random() < p_accept:
                current_fitness = neighbor_fitness
                solution_edgelist = neighbor_edgelist
                accepted_edgelists.append(solution_edgelist)
            else:
                # Undo permutation if not accepted.
                solution_perm[rnd2 : (rnd2 + rnd1)] = np.array(list(reversed(solution_perm[rnd2 : (rnd2 + rnd1)])))

        
        # Increment iteration counter and decrease temperature.
        it_count += 1
        curr_temp *= alpha
        print("current fitness: {0}".format(current_fitness))
        print("current iteration: {0}/{1}".format(it_count, max_it))


    return solution_edgelist, current_fitness, best_fitness, initial_fitness, accepted_edgelists, temp_vals, fitness_vals


if __name__ == '__main__':


    ### PARSE ARGUMENTS ###
    parser = argparse.ArgumentParser(description='Approximate solution to TSP using simulated annealing.')
    parser.add_argument('--num-nodes', type=int, default=30, help='Number of nodes to use')
    parser.add_argument('--dist-func', type=str, default='geodesic', choices=['geodesic', 'learned'], 
            help='Distance function to use')
    parser.add_argument('--prediction-model', type=str, default='gboosting', choices=['gboosting', 'rf'], 
            help='Prediction model to use for learned distance function')
    parser.add_argument('--max-it', type=int, default=1e4, help='Maximum iterations to perform')
    parser.add_argument('--temp', type=float, default=-1.0, help='Initial temperature')
    parser.add_argument('--temp-min', type=float, default=1.0e-8, help='Minimal temperature')
    parser.add_argument('--alpha', type=float, default=0.995, help='Cooling rate')
    args = parser.parse_args()
    #######################

    
    # Parse problem network.
    network = nx.read_gpickle('./data/grid_data/grid_network.gpickle')
    
    # Number of nodes to remove from network.
    to_remove = network.number_of_nodes() - args.num_nodes
    
    # Remove randomly sampled nodes to get specified number of nodes.
    network.remove_nodes_from(random.sample(list(network.nodes), to_remove))

    # Get distance function.
    dist_func = get_dist_func(network, which=args.dist_func, prediction_model=args.prediction_model)
    
    # Get solution using simulated annealing.
    solution_edgelist, current_fitness, best_fitness, initial_fitness, \
            accepted_edgelists, temp_vals, fitness_vals = anneal(network, max_it=args.max_it, 
                    temp=args.temp, temp_min=args.temp_min, alpha=args.alpha)

    # Save list of edge lists for animation.
    np.save('./results/edgelists/edgelist_tsp_sa.npy', list(map(np.vstack, accepted_edgelists)))
    nx.write_gpickle(network, './results/networks/network_tsp_sa.gpickle')
    
    # Plot temperature and fitness with respect to iteration.
    plt.plot(temp_vals)
    plt.xlabel('Iteration')
    plt.ylabel('Temperature')
    plt.savefig('./results/plots/temperature_tsp_sa.png')
    plt.clf()
    plt.plot(fitness_vals)
    plt.xlabel('Iteration')
    plt.ylabel('Fitness')
    plt.savefig('./results/plots/fitness_tsp_sa.png')

    # Print best solution fitness.
    print('Fitness of best found solution: {0:.3f}'.format(best_fitness))
    
    # Print initial best fitness.
    print('Fitness of initial solution: {0:.3f}'.format(initial_fitness))

    # Print increase in fitness.
    print('Fitness value improved by: {0:.3f}%'.format(100*initial_fitness/best_fitness))

