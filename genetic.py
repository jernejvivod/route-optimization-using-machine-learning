from copy import copy
import numpy as np
import networkx as nx
from models.distance import get_dist_func
import random


def get_parents(instances, best_num, rand_num, mutation_rate, network):
    scores = fitness_scores(instances, network)
    scores_sorted = sorted(scores, key=lambda x: x[1])
    new_gen = [i[0] for i in scores_sorted[:best_num]]
    best = new_gen[0]
    for i in range(rand_num):
        x = np.random.randint(len(instances))
        new_gen.append(instances[x])
    np.random.shuffle(new_gen)
    return new_gen, best


def get_children(p1, p2):
    p_size = len(p1) // 2
    parent_list1 = list(np.random.choice(p1, replace=False, size=p_size))
    child = [-999] * len(p1)
    for i in range(0, len(parent_list1)):
        child[i] = p1[i]
    for i, v in enumerate(child):
        if v == -999:
            for v2 in p2:
                if v2 not in child:
                    child[i] = v2
                    break
    child[-1] = child[0]
    return child


def crossover(current_gen, children_n):
    mid_point = len(current_gen) // 2
    next_gen = []
    for i, parent in enumerate(current_gen[:mid_point]):
        for x in range(children_n):
            next_gen.append(get_children(parent, current_gen[-i - 1]))
    return next_gen


def evolution(current_gen, max_gen_n, best_num, rand_num, mutation_rate, children_n, network):
    fitness = []
    best = -1
    for i in range(max_gen_n):
        print("Generation" + str(i) + ": len=" + str(len(current_gen)))
        if i < 0:
            print("best sore: " + str(fitness[-1]))
        parents, best = get_parents(current_gen, best_num, rand_num, mutation_rate, network)
        fitness.append(calculate_fitness(best, network=network))
        current_gen = crossover(parents, children_n)
    return fitness, best


def dist_func(n1, n2, function, network):
    dist_function = None
    if function == 'geodesic':
        dist_function = get_dist_func(network, which='geodesic')
    elif function == 'learned':
        dist_function = get_dist_func(network, which='learned')
    if dist_function is None:
        raise Exception("Specify distance function geodesic or learned ")

    return dist_function(n1, n2)


def generate_path(nodes):
    instance = copy(nodes)
    np.random.shuffle(instance)
    instance.append(instance[0])
    return list(instance)


def make_generation(nodes, population_size):
    gen = [generate_path(nodes) for i in range(population_size)]
    return gen


def calculate_fitness(instance, network):
    s = 0
    for c in range(len(instance) - 1):
        s += dist_func(instance[c], instance[c + 1], 'geodesic', network)
    return s


def fitness_scores(instances, network):
    scores = []
    for i in instances:
        s = calculate_fitness(i, network)
        scores.append((i, s))
        return scores


def main():
    # !!! RUN get_model.py !!!

    # read network
    network = nx.read_gpickle('./data/grid_data/grid_network.gpickle')

    # number of nodes in the network.
    NUM_NODES = 30

    # get the correct num of nodes with sampling and removing nodes
    to_remove = network.number_of_nodes() - NUM_NODES
    network.remove_nodes_from(random.sample(list(network.nodes), to_remove))

    test_population = make_generation(list(nx.nodes(network)), 10)

    fitness, best = evolution(test_population, 100, 150, 70, 0.5, 3, network)
    print("final fitness :")
    print(fitness)
    print("best instamce: ")
    print(best)


if __name__ == '__main__':
    main()
