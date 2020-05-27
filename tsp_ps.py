import numpy as np
import networkx as nx
import argparse
import random
from models.distance import get_dist_func


def get_fitness(perm_mat, node_list):
    """
    Compute fitness of position given by permutation matrix.

    Args:
        perm_mat (np.ndarray): Permutation matrix representing the
        position
        node_list (list): Node list (n-th element corresponds to 
        n-th row/column in permutation matrix)
    """
    
    return np.sum([dist_func(node_list[el[0]], node_list[el[1]]) for el in zip(*np.where(perm_mat))])


def get_prob_mat(perm_mat):
    """
    Get probability matrix from given permutation matrix.

    Args:
        perm_mat (numpy.ndarray): Permutation matrix

    Returns:
        (numpy.ndarray): Resulting probability matrix
    """

    # Get random matrix.
    prob_mat = np.random.rand(*perm_mat.shape)

    # Divide elements not equal to 1 in permutation matrix.
    prob_mat[~perm_mat.astype(bool)] /= 1000

    # Set diagonal values to 0.
    prob_mat[np.diag_indices_from(prob_mat)] = 0

    # Return computed probability matrix.
    return prob_mat


def get_selection_prob_coeff_mat(prob_mat, vel):
    """
    Get selection probability coefficients matrix from
    given probability matrix and velocity matrix.

    Args:
        prob_mat (numpy.ndarray): Probability matrix
        vel (numpy.ndarray): Velocity matrix

    Returns:
        (numpy.ndarray): Selection probability coefficients matrix
    """
    return np.clip(prob_mat + vel, a_min=None, a_max=1.0)


def merge_perm(perm_list):
    """
    Merge subpermutations given as tuples in list.

    Args:
        perm_list (list): List of subpermutations represented as tuples
    Returns:
        (list): List of merged subpermutations
    """

    # Copy list of permutations.
    perm_list_copy = perm_list.copy()

    # Set found flag to true.
    found = True
    while found:

        # Go over subpermutations.
        for idx in range(len(perm_list_copy)):

            # Get subpermutation to try to merge and remove from list.
            subperm = perm_list_copy[idx]
            del(perm_list_copy[idx])

            # Try to merge subpermutation.
            perm_list_copy, flg = merge_aux(subperm, perm_list_copy)

            # If merge successful, repeat by trying to merge next subpermutation.
            if flg:
                break
            else:
                # If merge not sucessful, put subpermutation back into list.
                perm_list_copy.insert(idx, subperm)

                # If at last subpermutation, return
                if idx == len(perm_list_copy)-1:
                    found = False
                else:
                    pass
    
    return perm_list_copy


def merge_aux(subperm, perm_list):
    """
    Auxiliary function for merging subpermutations.

    Args:
        subperm (tuple):
        perm_list (list)

    Returns:
        (tuple) list with subpermutation merged (if successful) and a flag
        that is set to True if merge was successful and False otherwise.
    """
    
    # Copy list of subpermutations.
    perm_list_aux = perm_list.copy()

    # Go over subpermutations and try to merge.
    for idx in range(len(perm_list_aux)):

        if subperm[-1] == perm_list_aux[idx][0]:
            # If end of subpermutation matches beginning of next.
            res = subperm + perm_list_aux[idx][1:]
            del(perm_list_aux[idx])
            perm_list_aux.append(res)
            return perm_list_aux, True
        if subperm[0] == perm_list_aux[idx][-1]:
            # If end of the subpermutation matches end of next.
            res = perm_list_aux[idx][:-1] + subperm
            del(perm_list_aux[idx])
            perm_list_aux.append(res)
            return perm_list_aux, True

    # If match unsuccessful, set flag to False.
    return perm_list, False



def update_pos(sel_prob_coeff_mat):
    """
    Update position of particle using computed selection probability
    coefficients matrix.
    """
    
    # Initialize empty list for constructed subpermutations.
    perms = []

    # Initialize array for updated position.
    updated_pos = np.zeros_like(sel_prob_coeff_mat, dtype=int) 

    # Create a copy of the selection probability coefficients matrix.
    sel_prob_coeff_mat_copy = sel_prob_coeff_mat.copy()

    # Iterate as many times as rows in selection probability coefficients matrix.
    for idx in range(sel_prob_coeff_mat.shape[0] - 1):

        # Select row.
        row_sel = np.argmax(np.max(sel_prob_coeff_mat_copy, axis=1))

        # Get column probabilities.
        col_probs = sel_prob_coeff_mat_copy[row_sel, :]/np.sum(sel_prob_coeff_mat_copy[row_sel, :])

        # Select column.
        col_sel = np.random.choice(range(len(col_probs)), size=1, p=col_probs)[0]

        # Append constructed subpermutation to list.
        perms.append((row_sel, col_sel))

        # Merge permutations in list.
        perms = merge_perm(perms)

        # Update position matrix.
        updated_pos[row_sel, col_sel] = 1 #

        # Remove contradictory elements from probability
        # coefficient matrix.
        sel_prob_coeff_mat_copy[row_sel, :] = 0 #
        sel_prob_coeff_mat_copy[:, col_sel] = 0 #

        #if idx < sel_prob_coeff_mat.shape[0]:
        for perm in perms:
            sel_prob_coeff_mat_copy[perm[-1], perm[0]] = 0
    
    # Return updated position.
    return updated_pos


def perm_to_mat(perm):
    """
    Transform permutation specified as a list to same permutation
    specified as a permutation matrix.

    Args:
        perm (list): Permutation specified as a list

    Returns:
        (numpy.ndarray): Matrix representation of the specified permutation
    """
    
    # Extend permutation to complete cycle.
    perm_aug = np.hstack((perm, perm[0]))

    # Initialize permutation matrix.
    perm_mat = np.zeros((len(perm), len(perm)), dtype=int)

    # Build permutation matrix.
    for idx in range(len(perm_aug)-1):
        perm_mat[perm_aug[idx], perm_aug[idx+1]] = 1
    
    # Return permutation matrix.
    return perm_mat


### SPECIAL OPERATORS USED IN VELOCITY UPDATE ###

def clip_plus(a, b):
    return np.clip(a + b, a_min=None, a_max=1.0)

def clip_minus(a, b):
    return np.clip(a - b, a_min=0, a_max=None)

def clip_mult(a, b):
    return np.clip(a * b, a_min=None, a_max=1.0)

#################################################


def get_edge_list(perm_mat, node_list):
    """
    Get edge list from permutation matrix. Decode actual node IDs using list of nodes.

    Args:
        perm_mat (numpy.ndarray): Permutation matrix to decode
        node_list (list): List of nodes in network

    Returns:
        (list): Decoded edgelist
    """
    return [(node_list[row], node_list[col]) for row, col in enumerate(np.argmax(perm_mat, axis=1))]


def pso(network, n_particles=100, w=0.9, c1=1.0, c2=1.0, max_it=10000):
    """
    Approximate solution to travelling salesman problem using particle swarm optimization.

    Args:
        network (object): Networkx representation of the graph.
        n_particles (int): Number of particles to use.
        w (float): The inertia weight.
        c1 (float): First velocity component weight.
        c2 (float): Second velocity component weight.
        max_it (int): Maximum iterations to perform.
    """
    
    # Initialize list for storing edge lists (for animating).
    edgelists = []

    # Initialize list of nodes (for converting enumerations to actual node IDs).
    node_list = list(network.nodes())
    
    # Initialize list of particles.
    particles = []
    for _ in range(n_particles):
        particle = dict()
        particle['position'] = perm_to_mat(np.random.permutation(np.arange(len(node_list))))
        particle['fitness_position'] = get_fitness(particle['position'], node_list)
        particle['p_best_position'] = particle['position'].copy()
        particle['fitness_p_best'] = particle['fitness_position']
        particle['velocity'] = np.zeros((len(node_list), len(node_list)), dtype=float)
        particles.append(particle) 
    
    
    # Set global best particle.
    best_particle = min(particles, key=lambda x: x['fitness_p_best'])
    global_best_solution = {
            'position' : best_particle['p_best_position'],
            'fitness' : best_particle['fitness_p_best']
            }
    initial_fitness = global_best_solution['fitness']
    
    # Initialize iteration counter.
    it_idx = 0

    # Main iteration loop.
    while it_idx < max_it:
        
        # Increment iteration counter.
        it_idx += 1

        # Process particles.
        for particle in particles:

            # Get velocity of particle.
            r1 = np.random.rand(len(node_list), len(node_list))
            r2 = np.random.rand(len(node_list), len(node_list))
            vel = clip_plus(w*particle['velocity'], clip_mult(c1, r1)) * \
                    clip_plus(clip_minus(particle['p_best_position'], particle['position']), clip_mult(c2, r2)) * \
                    clip_minus(global_best_solution['position'], particle['position'])
            
            # Get probability matrix.
            prob_mat = get_prob_mat(particle['position'])

            # Get selection probability coefficient matrix.
            sel_prob_coeff_mat = get_selection_prob_coeff_mat(prob_mat, vel)

            # Update position of particle.
            particle['position'] = update_pos(sel_prob_coeff_mat)
            particle['fitness_position'] = get_fitness(particle['position'], node_list)

            # Compare to personal best and global best position.
            if particle['fitness_position'] < particle['fitness_p_best']:
                particle['fitness_p_best'] = particle['fitness_position']
                particle['p_best_position'] = particle['position']
            if particle['fitness_position'] < global_best_solution['fitness']:
                global_best_solution['fitness'] = particle['fitness_position']
                global_best_solution['position'] = particle['position']
                edgelists.append(get_edge_list(global_best_solution['position'], node_list))
    
    # Return best found solution, fitness value of best found solution, initial best fitness value and
    # edgelist of network states corresponding to global best position updates.
    return global_best_solution['position'], global_best_solution['fitness'], initial_fitness, edgelists


if __name__ == '__main__':

    ### PARSE ARGUMENTS ###
    parser = argparse.ArgumentParser(description='Approximate solution to TSP using simulated annealing.')
    parser.add_argument('--num-nodes', type=int, default=70, help='Number of nodes to use')
    parser.add_argument('--dist-func', type=str, default='geodesic', choices=['geodesic', 'learned'], 
            help='Distance function to use')
    parser.add_argument('--prediction-model', type=str, default='xgboost', choices=['gboosting', 'rf'], 
            help='Prediction model to use for learned distance function')
    parser.add_argument('--max-it', type=int, default=3000, help='Maximum iterations to perform')
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
    solution_position, solution_fitness, initial_fitness, edgelists = pso(network, max_it=args.max_it)

    # Save list of edge lists for animation.
    np.save('./results/edgelists/edgelist_tsp_pso.npy', list(map(np.vstack, edgelists)))
    nx.write_gpickle(network, './results/networks/network_tsp_pso.gpickle')
   
    # Print best solution fitness.
    print('Fitness of best found solution: {0}'.format(solution_fitness))
    
    # Print initial best fitness.
    print('Fitness of initial solution: {0}'.format(initial_fitness))

    # Print increase in fitness.
    print('Fitness value improved by: {0}%'.format(initial_fitness/solution_fitness))

