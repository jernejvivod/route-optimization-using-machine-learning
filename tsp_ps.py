import numpy as np
import networkx as nx
import argparse
import random
from models.distance import get_dist_func


def crossover(particle_1, particle_2, p_mut, node_list):
    """
    Perform crossover using specified particles to obtain two new offspring.

    Args:
        particle_1 (dict): First particle
        particle_2 (dict): Second particle
        p_mut (float): Mutation probability
        node_list (list): List of node IDs

    Returns:
        (tuple): Two offspring obtained by crossover (particle dicts).
    """
    
    # 
    perm1 = np.array([el[0] for el in zip(*np.where(particle_1['position']))])
    perm2 = np.array([el[0] for el in zip(*np.where(particle_2['position']))])
    c1 = perm1[:np.random.randint(1, len(perm1))]
    c2 = perm2[:np.random.randint(1, len(perm2))]

    # Append elements in second solution in order found.
    offspring1 = np.hstack((c1, perm2[~np.in1d(perm2, c1)]))
    offspring2 = np.hstack((c2, perm1[~np.in1d(perm1, c2)]))

    # Apply mutations with specified probability.
    if np.random.rand() < p_mut:
        p1 = np.random.randint(0, len(offspring1))
        p2 = np.random.randint(0, len(offspring1))
        offspring1[[p1, p2]] = offspring1[[p2, p1]]
    if np.random.rand() < p_mut:
        p1 = np.random.randint(0, len(offspring2))
        p2 = np.random.randint(0, len(offspring2))
        offspring2[[p1, p2]] = offspring2[[p2, p1]]
    
    offspring1_mat = perm_to_mat(offspring1)
    offspring2_mat = perm_to_mat(offspring2)
    fitness_off1 = get_fitness(offspring1_mat, node_list)
    fitness_off2 = get_fitness(offspring2_mat, node_list)
    
    particle_off1 = {
            'position' : offspring1_mat,
            'fitness_position' : fitness_off1,
            'p_best_position' : offspring1_mat if fitness_off1 < particle_1['fitness_p_best'] else particle_1['p_best_position'],
            'fitness_p_best' : fitness_off1 if fitness_off1 < particle_1['fitness_p_best'] else particle_1['fitness_p_best'],
            'velocity' : particle_1['velocity']
            }
    
    particle_off2 = {
            'position' : offspring2_mat,
            'fitness_position' : fitness_off2,
            'p_best_position' : offspring2_mat if fitness_off2 < particle_2['fitness_p_best'] else particle_2['p_best_position'],
            'fitness_p_best' : fitness_off2 if fitness_off2 < particle_2['fitness_p_best'] else particle_2['fitness_p_best'],
            'velocity' : particle_2['velocity']
            }

    return particle_off1, particle_off2


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
                if idx == len(perm_list_copy) - 1:
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

    Args:
        sel_prob_coeff_mat (numpy.ndarray): Selection probability coefficient matrix

    Returns:
        (numpy.ndarray): New position matrix

    """
    
    # Initialize empty list for constructed subpermutations.
    perms = []

    # Initialize array for updated position.
    updated_pos = np.zeros_like(sel_prob_coeff_mat, dtype=int) 

    # Create a copy of the selection probability coefficients matrix.
    sel_prob_coeff_mat_copy = sel_prob_coeff_mat.copy()

    # Iterate as many times as rows in selection probability coefficients matrix.
    for idx in range(sel_prob_coeff_mat.shape[0]):

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
        updated_pos[row_sel, col_sel] = 1

        # Remove contradictory elements from probability
        # coefficient matrix.
        sel_prob_coeff_mat_copy[row_sel, :] = 0
        sel_prob_coeff_mat_copy[:, col_sel] = 0

        #if idx < sel_prob_coeff_mat.shape[0]:
        if idx < sel_prob_coeff_mat.shape[0] - 2:
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


def pso(network, n_particles=100, w_init=2.0, w_end=0.9, c1=1.0, c2=1.0, 
        max_it=500, aug=None, p_mut=0.08, breeding_coeff=0.5):
    """
    Approximate solution to travelling salesman problem using particle swarm optimization.

    Args:
        network (object): Networkx representation of the graph
        n_particles (int): Number of particles to use
        w_init (float): The initial inertia weight
        w_end (float): The final inertia weight after all iterations
        c1 (float): First velocity component weight
        c2 (float): Second velocity component weight
        max_it (int): Maximum iterations to perform
        aug (str): Algorithm augmentation to use. If None, use no augmentation. If equal to 'genetic' use 
        replacement of worst particles with crossovers of best particles.
        p_mut (float): Mutation probability (genetic augmentation)
        breeding_coeff (float): Fraction of best particles to use in crossover and fraction 
        of worst particles to replace with offspring (genetic augmentation)
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
    
    # Set inertia weight and multiplication coefficient.
    w = w_init
    w_mult = (w_end/2.0)**(1/max_it)

    # Main iteration loop.
    while it_idx < max_it:

        # Print iteration index and best fitness.
        print('iteration: {0}'.format(it_idx))
        print('best fitness: {0}'.format(global_best_solution['fitness']))
        
        # Reduce intertia weight.
        w *= w_mult
        
        # Increment iteration counter.
        it_idx += 1

        # Process particles.
        for particle in particles:

            # Get velocity of particle.
            r1 = np.random.rand(len(node_list), len(node_list))
            r2 = np.random.rand(len(node_list), len(node_list))
            particle['velocity'] = clip_plus(w*particle['velocity'], clip_mult(c1, r1)) * \
                    clip_plus(clip_minus(particle['p_best_position'], particle['position']), clip_mult(c2, r2)) * \
                    clip_minus(global_best_solution['position'], particle['position'])

            # Get probability matrix.
            prob_mat = get_prob_mat(particle['position'])

            # Get selection probability coefficient matrix.
            sel_prob_coeff_mat = get_selection_prob_coeff_mat(prob_mat, particle['velocity'])

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

        if aug == 'genetic':
            # If using genetic augmentation.
        
            # Sort particles by fitness.
            particles = sorted(particles, key=lambda x: x['fitness_position'])
            n_new_particles = int(np.ceil(breeding_coeff*len(particles)))

            # Initialize list for offspring.
            new_particles = []
            
            # Breed specified fraction of best particles.
            for idx in range(0, n_new_particles, 2):
                off1, off2 = crossover(particles[idx], particles[idx+1], p_mut=0.08, node_list=node_list)
                new_particles.extend([off1, off2])
            
            # Replace specified worst fraction of particles with offspring of best.
            particles[-len(new_particles):] = new_particles
    
    # Return best found solution, fitness value of best found solution, initial best fitness value and
    # edgelist of network states corresponding to global best position updates.
    return global_best_solution['position'], global_best_solution['fitness'], initial_fitness, map(np.vstack, edgelists)


if __name__ == '__main__':

    ### PARSE ARGUMENTS ###
    parser = argparse.ArgumentParser(description='Approximate solution to TSP using particle swarm optimization.')
    parser.add_argument('--num-nodes', type=int, default=30, help='Number of nodes to use')
    parser.add_argument('--dist-func', type=str, default='geodesic', choices=['geodesic', 'learned'], 
            help='Distance function to use')
    parser.add_argument('--prediction-model', type=str, default='xgboost', choices=['gboosting', 'rf'], 
            help='Prediction model to use for learned distance function')
    parser.add_argument('--max-it', type=int, default=200, help='Maximum iterations to perform')
    parser.add_argument('--n-particles', type=int, default=100, help='Number of particles to use')
    parser.add_argument('--w-init', type=float, default=2.0, help='Initial intertia weight')
    parser.add_argument('--w-end', type=float, default=0.8, help='Intertia weight at last iteration')
    parser.add_argument('--c1', type=float, default=1.0, help='First velocity component weight')
    parser.add_argument('--c2', type=float, default=1.0, help='Second velocity component weight')
    parser.add_argument('--aug', type=str, default=None, choices=['genetic'], help='Augmentation to use')
    parser.add_argument('--p-mut', type=float, default=0.08, help='Mutation rate (genetic augmentatio)')
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
    
    # Get solution using partice swarm.
    solution_position, solution_fitness, initial_fitness, edgelists = pso(network, max_it=args.max_it, n_particles=args.n_particles, 
            w_init=args.w_init, w_end=args.w_end, c1=args.c1, c2=args.c2, aug=args.aug, p_mut=args.p_mut, breeding_coeff=args.breeding_coeff)

    # Save list of edge lists for animation.
    np.save('./results/edgelists/edgelist_tsp_ps.npy', list(map(np.vstack, edgelists)))
    nx.write_gpickle(network, './results/networks/network_tsp_ps.gpickle')
   
    # Print best solution fitness.
    print('Fitness of best found solution: {0:.3f}'.format(solution_fitness))
    
    # Print initial best fitness.
    print('Fitness of initial solution: {0:.3f}'.format(initial_fitness))

    # Print increase in fitness.
    print('Fitness value improved by: {0:.3f}%'.format(100*initial_fitness/solution_fitness))

