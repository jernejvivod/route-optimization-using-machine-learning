import networkx as nx
from models.distance import get_dist_func
import random

### POZOR! Najprej pozeni skripto get_model.py ###

# Preberi omrezje
network = nx.read_gpickle('./data/grid_data/grid_network.gpickle')

# Zeljeno stevilo vozlisc v omrezju.
NUM_NODES = 30

# Pridobi omrezje z zeljenim stevilom vozlisc tako da nakljucno vzorcis
# in izbrises potrebno stevilo vozlisc.
to_remove = network.number_of_nodes() - NUM_NODES
network.remove_nodes_from(random.sample(list(network.nodes), to_remove))

# Pridobi funkcije za merjenje razdalje med vozliščema.
# parameter which ima lahko vrednosti 'geodesic' za geodezicno razdaljo
# in 'learned' za nauceno razdaljo.
dist_func_geodesic = get_dist_func(network, which='geodesic')
dist_func_learned = get_dist_func(network, which='learned')

# Primer izracuna razdalje.
n1 = list(network.nodes())[0]
n2 = list(network.nodes())[2]
dist1 = dist_func_geodesic(n1, n2)
dist2 = dist_func_learned(n1, n2)

# Mislim, da je to vse, kar se potrebuje za algoritme, ki jih bova razvijala :)

