import numpy as np
import random
import networkx as nx
from models.distance import get_dist_func
import argparse


def get_fitness(solution, initial_node, chromosome_order):
    """
    Compute fitness value for tsp solution given by specified solution.

    Args:
        solution: Solution to evaluate

    Returns:
        (float): Computed fitness value.
    """

    solution_perm = [initial_node] + [el[1] for el in sorted([(p, idx) for p, idx in zip(solution, chromosome_order)], key=lambda x: x[0])] + [initial_node]
    return sum(map(lambda el: dist_func(*el), [(solution_perm[idx], solution_perm[idx+1]) for idx in range(len(solution_perm)-1)]))


def pso(network, initial_node, n_particles=100, c1=0.5, c2=0.5, w=0.9, max_it=1000):
    
    node_list = list(network.nodes())
    chromosome_order = node_list[1:]
    initial_node = node_list[0]

    # Initialize list of particles.
    particles = []
    for _ in range(n_particles):
        particle = dict()
        particle['solution'] = np.random.dirichlet(np.ones(network.number_of_nodes()-1))
        particle['fitness_solution'] = get_fitness(particle['solution'], initial_node, chromosome_order)
        particle['p_best_solution'] = particle['solution'].copy()
        particle['fitness_p_best'] = particle['fitness_solution']
        particle['velocity'] = np.zeros(len(particle['solution']), dtype=float)
        particles.append(particle) 
    
    
    g_best_particle = min(particles, key=lambda x: x['fitness_p_best'])
    g_best_solution = g_best_particle['solution']
    g_best_fitness = g_best_particle['fitness_solution']


    # Perform particle swarm optimization.
    it_idx = 0
    while it_idx < max_it:
        it_idx += 1
        print(it_idx)
        
        # Go over particles.
        for particle in particles:
            r1 = np.random.rand()
            r2 = np.random.rand()
            particle['velocity'] = w*particle['velocity'] + \
                    c1*r1*(particle['p_best_solution'] - particle['solution']) + \
                    c2*r2*(g_best_solution - particle['solution'])
            
            particle['solution'] += particle['velocity']
            particle['solution'] /= np.sum(particle['solution'])
            particle['fitness_solution'] = get_fitness(particle['solution'], initial_node, chromosome_order)

            if particle['fitness_solution'] < particle['fitness_p_best']:
                particle['p_best_solution'] = particle['solution']
                particle['fitness_p_best'] = particle['fitness_solution']
            if particle['fitness_solution'] < g_best_fitness:
                g_best_fitness = particle['fitness_solution']
                g_best_solution = particle['solution']
                print(g_best_fitness)
           

    return g_best_solution, g_best_fitness, edgelists, fitness_vals




if __name__ == '__main__':

    ### PARSE ARGUMENTS ###
    parser = argparse.ArgumentParser(description='Approximate solution to TSP using particle swarm optimization.')
    parser.add_argument('--num-nodes', type=int, default=10, help='number of nodes to use')
    parser.add_argument('--dist-func', type=str, default='geodesic', choices=['geodesic', 'learned'],
            help='distance function to use')
    parser.add_argument('--prediction-model', type=str, default='xgboost', choices=['gboosting', 'rf'],
            help='prediction model to use for learned distance function')
    parser.add_argument('--max-it', type=int, default=100, help='maximum iterations to perform')
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

    # Approximate solution using particle swarm optimization.
    solution_edgelist, best_fitness, edgelists, fitness_vals = pso(network, initial_node=0, n_particles=100, c1=0.5, c2=0.5, max_it=args.max_it)

    # Save list of edge lists for animation.
    # np.save('./results/edgelists/edgelist_tsp.gpickle', list(map(np.vstack, accepted_edgelists)))
    # nx.write_gpickle(network, './results/networks/network_tsp.gpickle')

