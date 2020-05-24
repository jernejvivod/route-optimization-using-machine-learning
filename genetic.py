from copy import copy
import numpy as np
from geopy.geocoders import Nominatim
import networkx as nx
from models.distance import get_dist_func
import random


def get_parents(instances, best_num, rand_num, mutation_rate):
    scores = fitness_scores(instances)
    scores_sorted = sorted(scores, key=lambda x: x[i])
    new_gen =[i[0] for i in scores_sorted[:best_num]]
    best = new_gen[0]
    for i in range(rand_num):
        x = np.random.randint(len(instances))
        new_gen.append(instances[x])
    np.random.shuffle(new_gen)
    return new_gen, best


def get_children(p1,p2):
    p_size =len(p1)//2
    parent_list1 = list(np.random.choice(p1, replace=False, size=p_size))
    child = [-999]*len(p1)
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
    mid_point = len(current_gen)//2
    next_gen = []
    for i, parent in enumerate(current_gen[:mid_point]):
        for x in range(children_n):
            next_gen.append(get_children(parent,current_gen[-i-1]))
    return next_gen


def evolution(current_gen, max_gen_n, best_num, rand_num, mutation_rate, children_n,):
    fitness =[]
    best = -1
    for i in range(max_gen_n):
        print("Generation" + str(i) + ": len=" + str(len(current_gen)))
        print("best sore: " + str(fitness[-1]))
        parents, best = get_parents(current_gen,best_num, rand_num, mutation_rate)
        fitness.append(calculate_fitness(best))
        current_gen = crossover(parents, children_n)
    return fitness, best


def dist_func(n1, n2):
    pass


def generate_path(points):
    instance = copy(points)
    np.random.shuffle(instance)
    instance.append(instance[0])
    return list(instance)


def make_generation(points, population_size):
    gen = [generate_path(points) for i in range(population_size)]
    return gen


def calculate_fitness(instance):
    s = 0
    for c in range(instance - 1):
        s += dist_func(c, c + 1)
    return s


def fitness_scores(instances):
    scores = []
    for i in instances:
        s = calculate_fitness(i)
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

    # get distance functions for two nodes which = 'geodesic' or 'learned'
    dist_func_geodesic = get_dist_func(network, which='geodesic')
    dist_func_learned = get_dist_func(network, which='learned')

    # Examples of distance calculation

    """
    n1 = list(network.nodes())[0]
    n2 = list(network.nodes())[2]
    dist1 = dist_func_geodesic(n1, n2)
    dist2 = dist_func_learned(n1, n2)
    """

    # TODO: figure out how to get the nodes in a format for genetic algorithms as below - print out nodes first
    # could just take random nodes as a generation ....

    test_locations = {'L1': (40.819688, -73.915091),
                      'L2': (40.815421, -73.941761),
                      'L3': (40.764198, -73.910785),
                      'L4': (40.768790, -73.953285),
                      'L5': (40.734851, -73.952950),
                      'L6': (40.743613, -73.977998),
                      'L7': (40.745313, -73.993793),
                      'L8': (40.662713, -73.946101),
                      'L9': (40.703761, -73.886496),
                      'L10': (40.713620, -73.943076),
                      'L11': (40.725212, -73.809179)
                      }

    # key_names = list(test_locations.keys())

    locator = Nominatim(user_agent="my-application")
    addresses = []

    for key in test_locations:
        location = locator.reverse(test_locations[key])
        addresses.append(location.address)

    t_addresses = []
    for a in addresses:
        s = "".join(a.split(",")[0:2])
        s = s + " NY"
        # print(s)
        t_addresses.append(s)

    # test_addresses = dict(zip(key_names, t_addresses))
    # print(test_addresses)
    test_population = make_generation(list(test_locations.keys()), 10)

    fitness, best = evolution(test_population,100,150,70,0.5,3)
    print("final fitness :")
    print(fitness)
    print("best instamce: ")
    print(best)


if __name__ == '__main__':
    main()
