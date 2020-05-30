import numpy as np
import networkx as nx
import argparse
import random
from models.distance import get_dist_func

def get_fitness(solution, initial_node, node_list):
    """
    Get fitness of solution encoded by permutation.
    
    Args:
        solution (numpy.ndarray): Solution encoded as a permutation
        initial_node (int): Initial node in the permutation (equal to the first element - redundant)
        node_list (list): List of node IDs in network
    
    Returns:
        (float): Fitness of specified solution
    """
    
    # Append path back to initial node.
    solution_aux = np.hstack((solution, initial_node))

    # Compute fitness.
    return np.sum([dist_func(node_list[el[0]], node_list[el[1]]) 
        for el in [(solution_aux[idx], solution_aux[idx+1]) 
            for idx in range(len(solution_aux)-1)]])


def get_inv_dist_mat(node_list):
    """
    Get pairwise distance matrix for specified nodes in node list.

    Args:
        node_list (list): Nodes for which to compute the pairwise distances

    Returns:
        (numpy.ndarray): Matrix of pairwise distances
    """
    
    # Initialize array.
    dist_mat = np.zeros((len(node_list), len(node_list)), dtype=float)

    # Compute pairwise distances
    for idx1 in range(len(node_list)-1):
        for idx2 in range(idx1+1, len(node_list)):
            dist_mat[idx1, idx2] = dist_mat[idx2, idx1] = 1/dist_func(node_list[idx1], node_list[idx2])

    # Return computed distance matrix.
    return dist_mat


def aco(network, n_ants=100, max_it=500, rho=0.1, alpha=1.0, beta=1.0, q=1.0, 
        aug='relinking', p_mut=0.08, p_accept_worse=0.1, breeding_coeff=0.5):
    """
    Perform ant colony optimization to estimate solution for travelling salesman problem.

    Args:
        network (object): Networkx representation of the graph
        n_ants (int): Number of ants to use
        max_it (int): Maximum number of iterations to perform
        rho (float): Evaporation rate
        alpha (float): Pheromone matrix power in transition probability matrix construction
        beta (float): Inverse distance matrix power in transition probability matrix construction
        q (float): Pheromone trail coefficient
        aug (str): Algorithm augmentation to use. If None, use no augmentation. If equal to 'relinking' use path
        relinking method. If equal to 'genetic' use replacement of worst ants with crossovers of best ants.
        p_mut (float): Mutation probability
        p_accept_worse (float): Probability of accepting a relinked solution that is worse than original.
        breeding_coeff (float): Fraction of best ants to use in crossover and fraction of worst ants to
        replace with offspring (genetic augmentation)

    Returns:
        (tuple): Best found solution, fitness of best solution, edgelists corresponding to solutions representing
        the new global best solution.
    """
    
    # Check aug parameter.
    if aug is not None:
        if aug not in {'relinking', 'genetic'}:
            raise(ValueError('unknown value specified for aug parameter'))

    # Initialize list for storing edge lists (for animating).
    edgelists = []
    
    # Initialize list of nodes (for converting enumerations to actual node IDs).
    node_list = list(network.nodes())
    
    # Set initial node.
    initial_node = 0
    
    # Initilize best found solution.
    best_solution = {
            'fitness' : np.inf,
            'solution' : None
            }

    # Compute distance matrix for locations.
    inv_dist_mat = get_inv_dist_mat(node_list)

    # Initialize pheromone matrix.
    pher_mat = 0.01*np.ones_like(inv_dist_mat, dtype=float)

    # Initialize iteration index.
    it_idx = 0

    # Main iteration loop.
    while it_idx < max_it:
        
        # Increment iteration counter.
        it_idx += 1

        # Print iteration index and best fitness.
        print('iteration: {0}'.format(it_idx))
        print('best fitness: {0}'.format(best_solution['fitness']))
    
        # Initialize array for storing ant solutions.
        ant_solutions = np.empty((n_ants, len(node_list)), dtype=int)

        # Initialize array for storing ant fitness values.
        ant_fitness_vals = np.empty(n_ants, dtype=float)
    
        # Build transition probability matrix.
        p_mat = (pher_mat**alpha) * (inv_dist_mat**beta)
        
        # Run ACO step.
        for ant_idx in range(n_ants):

            # Set initial node.
            current_node = initial_node

            # Get set of unvisited nodes.
            unvisited = set(range(len(node_list)))
            unvisited.remove(initial_node)
        
            # Build ant's solution.
            solution_nxt = np.empty(len(node_list), dtype=int)
            solution_nxt[0] = initial_node
            for step_idx in range(len(node_list) - 1):
                unvisited_list = list(unvisited)
                probs = p_mat[current_node, unvisited_list] / np.sum(p_mat[current_node, unvisited_list])
                node_nxt = np.random.choice(unvisited_list, size=1, p=probs)[0]
                unvisited.remove(node_nxt)
                solution_nxt[step_idx+1] = node_nxt
                current_node = node_nxt
             
            # Compute fitness of solution and compare to global best.
            fitness_solution = get_fitness(solution_nxt, initial_node, node_list)
            ant_fitness_vals[ant_idx] = fitness_solution
            if fitness_solution < best_solution['fitness']:
                best_solution['fitness'] = fitness_solution
                best_solution['solution'] = solution_nxt
                solution_nxt_aug = np.hstack((solution_nxt, initial_node))

                # Store edge list (for animating).
                edgelists.append([(node_list[solution_nxt_aug[idx]], node_list[solution_nxt_aug[idx+1]]) 
                    for idx in range(len(solution_nxt_aug) - 1)])
            
            # Store ant's solution.
            ant_solutions[ant_idx, :] = solution_nxt
            
        
        # Initialize matrix for accumulating pheromones (for pheromone update).
        pher_add_mat = np.zeros_like(pher_mat, dtype=float)
        
        if aug == 'relinking':
            # If using relinking augmentation.

            # Go over solutions.
            for idx_solution in range(ant_solutions.shape[0]):

                # Split solution at random point.
                sec1, sec2 = np.split(ant_solutions[idx_solution], \
                        indices_or_sections=[np.random.randint(1, len(ant_solutions[idx_solution]))])

                # Relink.
                solution_mod = np.hstack((sec1, list(reversed(sec2)))) 

                # Apply mutation with probability.
                if np.random.rand() < p_mut:
                    p1 = np.random.randint(0, len(solution_mod))
                    p2 = np.random.randint(0, len(solution_mod))
                    solution_mod[[p1, p2]] = solution_mod[[p2, p1]]
                
                # Compute fitness value of relinked solution.
                fitness_mod = get_fitness(solution_mod, initial_node, node_list)

                # If fitness better accept. Also accept with specified probability.
                if (fitness_mod < ant_fitness_vals[idx_solution]) or (np.random.rand() < p_accept_worse):
                    ant_solutions[idx_solution, :] = solution_mod
                    ant_fitness_vals[idx_solution] = fitness_mod

        if aug == 'genetic': 
            # If using genetic augmentation.
            
            # Sort ants ant fitness values from best to worst.
            p = ant_fitness_vals.argsort()
            ant_fitness_vals = ant_fitness_vals[p]
            ant_solutions = ant_solutions[p, :]
            
            # Get number of new ants and initialize array for crossovers.
            n_new_ants = int(np.ceil(breeding_coeff*ant_solutions.shape[0]))
            ant_solutions_new = np.empty((n_new_ants, ant_solutions.shape[1]), dtype=int)
            ant_fitness_vals_new = np.empty(ant_solutions_new.shape[0], dtype=float)

            # Go over solutions for which to perform crossover.
            for idx in range(0, ant_solutions_new.shape[0], 2):

                # Get solutions and cut at random point.
                ant_sol_1 = ant_solutions[idx, :]
                ant_sol_2 = ant_solutions[idx+1, :]
                c1 = ant_sol_1[:np.random.randint(1, len(ant_sol_1))]
                c2 = ant_sol_2[:np.random.randint(1, len(ant_sol_2))]
                
                # Append elements in second solution in order found.
                offspring1 = np.hstack((c1, ant_sol_2[~np.in1d(ant_sol_2, c1)]))
                offspring2 = np.hstack((c2, ant_sol_1[~np.in1d(ant_sol_1, c2)]))

                # Apply mutations with specified probability.
                if np.random.rand() < p_mut:
                    p1 = np.random.randint(0, len(offspring1))
                    p2 = np.random.randint(0, len(offspring1))
                    offspring1[[p1, p2]] = offspring1[[p2, p1]]
                if np.random.rand() < p_mut:
                    p1 = np.random.randint(0, len(offspring2))
                    p2 = np.random.randint(0, len(offspring2))
                    offspring2[[p1, p2]] = offspring2[[p2, p1]]
                
                # Set offspring and fitness values.
                ant_solutions_new[idx, :] = offspring1
                ant_solutions_new[idx+1, :] = offspring2
                ant_fitness_vals_new[idx] = get_fitness(offspring1, initial_node, node_list)
                ant_fitness_vals_new[idx+1] = get_fitness(offspring2, initial_node, node_list)
            
            # Replace worst ants with offspring of best.
            ant_solutions[-ant_solutions_new.shape[0]:] = ant_solutions_new
            ant_fitness_vals[-len(ant_fitness_vals_new):] = ant_fitness_vals_new

        
        # Compute and print diversity of solutions.
        diversity = (np.mean(ant_fitness_vals) - np.min(ant_fitness_vals))/(np.max(ant_fitness_vals) - np.min(ant_fitness_vals))
        print(diversity)
        
        # Add pheromones to pheromone accumulation matrix (for next iteration).
        for idx_sol, solution in enumerate(ant_solutions):
            for idx in range(len(solution)-1):
                pher_add_mat[solution[idx], solution[idx+1]] += q*(1/ant_fitness_vals[idx_sol])
                pher_add_mat[solution[idx+1], solution[idx]] += q*(1/ant_fitness_vals[idx_sol])

        # Update pheromone matrix.
        pher_mat = (1-rho)*pher_mat + pher_add_mat


    # Return best found solution, fitness value of best found solution and edgelist of network states 
    # corresponding to global best position updates.
    return best_solution['solution'], best_solution['fitness'], edgelists
    

if __name__ == '__main__':

    ### PARSE ARGUMENTS ###
    parser = argparse.ArgumentParser(description='Approximate solution to TSP using ant colony optimization.')
    parser.add_argument('--num-nodes', type=int, default=50, help='Number of nodes to use')
    parser.add_argument('--dist-func', type=str, default='geodesic', choices=['geodesic', 'learned'], 
            help='Distance function to use')
    parser.add_argument('--prediction-model', type=str, default='gboosting', choices=['gboosting', 'rf'], 
            help='Prediction model to use for learned distance function')
    parser.add_argument('--max-it', type=int, default=100, help='Maximum iterations to perform')
    parser.add_argument('--n-ants', type=int, default=100, help='Number of ants to use')
    parser.add_argument('--rho', type=float, default=0.1, help='Evaporation rate parameter')
    parser.add_argument('--alpha', type=float, default=1.0, help='Alpha parameter in transition probability matrix update')
    parser.add_argument('--beta', type=float, default=1.0, help='Beta parameter in transition probability matrix update')
    parser.add_argument('--q', type=float, default=1.0, help='Pheromone update coefficient')
    parser.add_argument('--aug', type=str, default=None, choices=['relinking', 'genetic'], help='Augmentation to use')
    parser.add_argument('--p-mut', type=float, default=0.08, help='Mutation rate (augmentation)')
    parser.add_argument('--p-accept-worse', type=float, default=0.08, 
            help='Probability of accepting a worse result of relinking (relinking augmentation)')
    parser.add_argument('--breeding-coeff', type=float, default=0.5, 
            help='Fraction of best solution for which to perform crossover and fraction of worst solution to replace by offspring (genetic augmentation)')
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

    # Get solution using ant colony optimization.
    solution_position, solution_fitness, edgelists = aco(network, n_ants=args.n_ants, max_it=args.max_it, rho=args.rho, 
            alpha=args.alpha, beta=args.beta, q=args.q, aug=args.aug, p_mut=args.p_mut, 
            p_accept_worse=args.p_accept_worse, breeding_coeff=args.breeding_coeff)

    # Save list of edge lists for animation.
    np.save('./results/edgelists/edgelist_tsp_ac.npy', list(map(np.vstack, edgelists)))
    nx.write_gpickle(network, './results/networks/network_tsp_ac.gpickle')
   
    # Print best solution fitness.
    print('Fitness of best found solution: {0:.3f}'.format(solution_fitness))
    
