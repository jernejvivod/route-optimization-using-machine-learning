import numpy as np
import random
import networkx as nx
from models.distance import get_dist_func
import argparse

## TODO - implementation with probabilities.

def get_fitness(solution):
    """
    Compute fitness value for tsp solution given by specified solution.

    Args:
        solution: Solution to evaluate

    Returns:
        (float): Computed fitness value.
    """

    return sum(map(lambda el: dist_func(*el), [(solution[idx], solution[idx+1]) for idx in range(len(solution)-1)] + [(solution[-1], solution[0])]))


def pso(network, initial_node, n_particles=100000, c1=0.9, c2=1.0, max_it=1000):
    
    edgelists = []
    fitness_vals = []

    # Initialize list of particles.
    particles = []
    for _ in range(n_particles):
        particle = dict()
        particle['solution'] = np.random.permutation(np.array(list(network.nodes())))
        particle['fitness_solution'] = get_fitness(particle['solution'])
        particle['p_best_solution'] = particle['solution'].copy()
        particle['fitness_p_best'] = particle['fitness_solution']
        particles.append(particle) 
    
    g_best_fitness_prev = 1.0e6

    # Perform particle swarm optimization.
    it_idx = 0
    while it_idx < max_it:
        it_idx += 1
        print(it_idx)

        # Get current global best solution and its score.
        g_best_particle = min(particles, key=lambda x: x['fitness_p_best'])
        g_best_solution = g_best_particle['solution']
        g_best_fitness = g_best_particle['fitness_solution']
        if g_best_fitness < g_best_fitness_prev:
            edgelists.append([(g_best_solution[idx], g_best_solution[idx+1]) 
                for idx in range(len(g_best_solution)-1)] + [(g_best_solution[-1], g_best_solution[0])])
        g_best_fitness_prev = g_best_fitness
        print(g_best_fitness)
        fitness_vals.append(g_best_fitness)
        
        # Go over particles.
        for particle in particles:
            
            # Initialize lists for current velocities.
            velocity1 = []
            velocity2 = []
            
            p_best_solution_copy = particle['p_best_solution'].copy()
            g_best_solution_copy = g_best_solution.copy()


            # Compute directions towards personal best and global best.
            for idx in range(network.number_of_nodes()):
                if particle['solution'][idx] != p_best_solution_copy[idx]:
                    swap = (idx, np.where(p_best_solution_copy == particle['solution'][idx])[0][0])
                    velocity1.append(swap)
                    p_best_solution_copy[[swap[0], swap[1]]] = p_best_solution_copy[[swap[1], swap[0]]]
                if particle['solution'][idx] != g_best_solution_copy[idx]:
                    swap = (idx, np.where(g_best_solution_copy == particle['solution'][idx])[0][0])
                    velocity2.append(swap)
                    g_best_solution_copy[[swap[0], swap[1]]] = g_best_solution_copy[[swap[1], swap[0]]]
            

            # Compute new solution of current particle.
            for swap1 in velocity1:
                if np.random.rand() <= c1:
                    particle['solution'][[swap1[0], swap1[1]]] = particle['solution'][[swap1[1], swap1[0]]]
            for swap2 in velocity2:
                if np.random.rand() <= c2:
                    particle['solution'][[swap2[0], swap2[1]]] = particle['solution'][[swap2[1], swap2[0]]]
            
            
            particle['fitness_solution'] = get_fitness(particle['solution'])
            if particle['fitness_solution'] < particle['fitness_p_best']:
                particle['fitness_p_best'] = particle['fitness_solution']
                particle['p_best_solution'] = particle['solution']


    return g_best_solution, g_best_fitness, edgelists, fitness_vals




if __name__ == '__main__':

    ### PARSE ARGUMENTS ###
    parser = argparse.ArgumentParser(description='Approximate solution to TSP using particle swarm optimization.')
    parser.add_argument('--num-nodes', type=int, default=40, help='number of nodes to use')
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
    solution_edgelist, best_fitness, edgelists, fitness_vals = pso(network, initial_node=0, n_particles=2000, c1=0.9, c2=1.0, max_it=args.max_it)

    # Save list of edge lists for animation.
    # np.save('./results/edgelists/edgelist_tsp.gpickle', list(map(np.vstack, accepted_edgelists)))
    # nx.write_gpickle(network, './results/networks/network_tsp.gpickle')

