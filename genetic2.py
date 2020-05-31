import numpy as np
import networkx as nx
import argparse
import random
from models.distance import get_dist_func


def get_fitness(solution, node_list):
    """
    Get fitness of solution encoded by permutation.
    
    Args:
        solution (numpy.ndarray): Solution encoded as a permutation
        node_list (list): List of node IDs in network
    
    Returns:
        (float): Fitness of specified solution
    """
    
    # Append path back to initial node.
    solution_aux = np.hstack((solution, solution[0]))

    # Compute fitness.
    return np.sum([dist_func(node_list[el[0]], node_list[el[1]]) 
        for el in [(solution_aux[idx], solution_aux[idx+1]) 
            for idx in range(len(solution_aux)-1)]])


def crossover(inst1, inst2, p_mut):
    """
    Perform crossover operation between two solutions encoded in permutation
    lists.

    Args:
        inst1 (list): First solution
        inst2 (list): Second solution
    
    Returns:
        (tuple): Offspring (results) of crossover

    """
    
    # Cut solutions at random points.
    c1 = inst1[:np.random.randint(1, len(inst1))]
    c2 = inst2[:np.random.randint(1, len(inst2))]

    # Append elements in second solution in order found.
    offspring1 = np.hstack((c1, inst2[~np.in1d(inst2, c1)]))
    offspring2 = np.hstack((c2, inst1[~np.in1d(inst1, c2)]))

    # Apply mutations with specified probability.
    if np.random.rand() < p_mut:
        p1 = np.random.randint(0, len(offspring1))
        p2 = np.random.randint(0, len(offspring1))
        offspring1[[p1, p2]] = offspring1[[p2, p1]]
    if np.random.rand() < p_mut:
        p1 = np.random.randint(0, len(offspring2))
        p2 = np.random.randint(0, len(offspring2))
        offspring2[[p1, p2]] = offspring2[[p2, p1]]
    
    # Return the offspring.
    return offspring1, offspring2


def tournament(population, num_in_group, p_mut):
    """
    Perform tournament selection and replace worst two members of
    each tournament with results of performing crossover between two
    winning members.

    Args:
        population (list): Population of solutions sorted by fitness
        num_in_group (int): Number of solutions in each tournament
        p_mut (float): Mutation probability

    Returns:
        population (list): Population after performing tournament selection
        and crossover.

    """
    
    # Go over groups.
    for idx in range(0, len(population), num_in_group):

        # Get next group.
        group = population[idx:idx+num_in_group]

        # Compute probabilities of selecting each solution.
        p = 0.9*np.ones(len(group), dtype=float)
        p_opp = 1 - p
        p_sel = p*(p_opp**np.arange(len(group)))

        # Get winners.
        breeders_idx = np.random.choice(range(len(group)), size=2, replace=False, p=p_sel/np.sum(p_sel))

        # Perform crossover of winners, replace worst solutions in group and add back to population.
        offspring1, offspring2 = crossover(*[group[idx] for idx in breeders_idx], p_mut)
        group[-2:] = [offspring1, offspring2]
        population[idx:idx+num_in_group] = group

    # Return population after tournament selection and crossover operations.
    return population


def genetic(network, max_it=300, population_size=30, method='ranked', breeding_coeff=0.5, num_in_group=4, p_mut=0.08):
    """
    Approximate solution to travelling salesman problem using genetic algorithms.

    Args:
        network (object): Networkx representation of the network
        max_it (int): Maximum iterations to perform
        population_size (int): Population size
        method (str): Method used to update population. Valid values are 'ranked' and 'tournament'
        breeding_coeff (float): Fraction of best ants to use in crossover and fraction of worst ants to
        replace with offspring (ranked method)
        num_in_group (int): Number of solutions in each tournament (tournament method)
        p_mut (float): Probability of mutation
    """
    
    # Check if method parameter value is valid.
    if method not in {'ranked', 'tournament'}:
        raise(ValueError('unknown method parameter value'))
    
    # Initialize list for storing edge lists (for animations).
    edgelists = []

    # Get list of node IDs.
    node_list = list(network.nodes())

    # Initialize population using random solutions.
    population = [np.random.permutation(np.arange(len(node_list))) 
            for _ in range(population_size)]
    
    # Get initial best solution and fitness.
    initial_best = sorted([(inst, get_fitness(inst, node_list)) 
        for inst in population], key=lambda x: x[1])
    
    # Initialize dictionary for storing global best solution and its fitness.
    global_best = {
            'fitness' : initial_best[0][1],
            'solution' : initial_best[0][0]
            }
   
    # Main iteration loop
    for it_idx in range(max_it):


        print(it_idx)
        print(global_best['fitness'])

        # Evaluate and sort population
        evaluated = sorted([(inst, get_fitness(inst, node_list)) 
            for inst in population], key=lambda x: x[1])

        # Check best in population agains global best.
        if evaluated[0][1] < global_best['fitness']:
            global_best['fitness'] = evaluated[0][1]
            global_best['solution'] = evaluated[0][0]
            solution = np.hstack((global_best['solution'], global_best['solution'][0]))
            edgelists.append([(node_list[solution[idx]], node_list[solution[idx+1]]) for idx in range(len(solution)-1)])

        if method == 'ranked':
            # if performing ranked selection.

            # Get sorted population.
            sorted_population = list(map(lambda x: x[0], evaluated))
            
            # Get number of breeders and make sure number is even.
            n_breeders = int(np.ceil(breeding_coeff*len(evaluated)))
            n_breeders += n_breeders % 2

            # Get breeders.
            breeders = sorted_population[:n_breeders]

            # Initialize list for storing offspring.
            offspring = []

            # Perform crossover among pairs of breeders.
            for idx in range(0, len(breeders) - 1, 2):
                offspring1, offspring2 = crossover(breeders[idx], breeders[idx+1], p_mut)
                
                # Apply mutations with specified probability.
                if np.random.rand() < p_mut:
                    p1 = np.random.randint(0, len(offspring1))
                    p2 = np.random.randint(0, len(offspring1))
                    offspring1[[p1, p2]] = offspring1[[p2, p1]]
                if np.random.rand() < p_mut:
                    p1 = np.random.randint(0, len(offspring2))
                    p2 = np.random.randint(0, len(offspring2))
                    offspring2[[p1, p2]] = offspring2[[p2, p1]]
                
                # Add offspring to list.
                offspring.extend([offspring1, offspring2])

            # Replace worst solution in population with offspring of selected breeders.
            sorted_population[-n_breeders:] = offspring
            population = sorted_population

        elif method == 'tournament':
            # If performing tournament selection.

            # Perform tournament selection and crossover to get updated population.
            population = tournament(list(map(lambda x: x[0], evaluated)), num_in_group, p_mut)

    # Return best found solution, fitness value of best found solution, initial best fitness value and 
    # edgelist of network states corresponding to global best position updates.
    return global_best['solution'], global_best['fitness'], initial_best[0][1], edgelists


if __name__ == '__main__':

    ### PARSE ARGUMENTS ###
    parser = argparse.ArgumentParser(description='Approximate solution to TSP using particle swarm optimization.')
    parser.add_argument('--num-nodes', type=int, default=30, help='Number of nodes to use')
    parser.add_argument('--dist-func', type=str, default='geodesic', choices=['geodesic', 'learned'], 
            help='Distance function to use')
    parser.add_argument('--prediction-model', type=str, default='gboosting', choices=['gboosting', 'rf'], 
            help='Prediction model to use for learned distance function')
    parser.add_argument('--max-it', type=int, default=500, help='Number of iterations to perform')
    parser.add_argument('--population-size', type=int, default=50, help='Population size to use')
    parser.add_argument('--method', type=str, choices=['ranked', 'tournament'], default='ranked', help='Population update method to use')
    parser.add_argument('--breeding-coeff', type=float, default=0.5, 
            help='Fraction of best solution for which to perform crossover and fraction of worst solution to replace by offspring (ranked method)')
    parser.add_argument('--num-in-group', type=int, default=10, help='Tournament size (tournament method)')
    parser.add_argument('--p-mut', type=float, default=0.08, help='Mutation probability')
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
    
    # Get solution using genetic algorithm. 
    solution, solution_fitness, initial_fitness, edgelists = genetic(network, max_it=args.max_it, population_size=args.population_size, 
            method=args.method, breeding_coeff=args.breeding_coeff, num_in_group=args.num_in_group, p_mut=args.p_mut)

    # Save list of edge lists for animation.
    np.save('./results/edgelists/edgelist_tsp_genetic2.npy', list(map(np.vstack, edgelists)))
    nx.write_gpickle(network, './results/networks/network_tsp_genetic2.gpickle')
   
    # Print best solution fitness.
    print('Fitness of best found solution: {0:.3f}'.format(solution_fitness))
    
    # Print initial best fitness.
    print('Fitness of initial solution: {0:.3f}'.format(initial_fitness))

    # Print increase in fitness.
    print('Fitness value improved by: {0:.3f}%'.format(100*initial_fitness/solution_fitness))

