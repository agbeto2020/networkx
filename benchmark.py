from networkx.algorithms.isomorphism.isomorphfastiso import GraphMatcher as FIGraphMatcher
from networkx.algorithms.isomorphism.isomorphfastiso import DiGraphMatcher as FIDiGraphMatcher
from networkx.algorithms.isomorphism import GraphMatcher as nxGraphMatcher
from networkx.algorithms.isomorphism import DiGraphMatcher as nxDiGraphMatcher
from itertools import product
import networkx as nx
import time
from memory_profiler import profile
from memory_profiler import memory_usage
import threading
import multiprocessing
import csv
import random
from timeoutpool import TimeoutPool
import pickle
    
#
def generate_erdos_renyi_graph(n, p=0.1, directed=False):
    """
        Generates an Erdős-Rényi graph.

        Parameters:
        n (int): The number of nodes in the graph.
        p (float): The probability for each pair of nodes to be connected by an edge.
        directed (bool, optional): If True, creates a directed graph. Defaults to False.
    """
    if directed:
        return nx.gnp_random_graph(n, p, directed=True)
    else:
        return nx.gnp_random_graph(n, p)
    
def generate_barabasi_albert_graph(n, m=10):
    """
        Generates a Barabási-Albert graph.

        Parameters:
        n (int): The number of nodes in the graph.
        m (int): The number of edges to attach from a new node to existing nodes.
    """
    return nx.barabasi_albert_graph(n, m)

def generate_watts_strogatz_graph(n, k=10, p=0.1):
    """
        Generates a Watts-Strogatz small-world graph.

        Parameters:
        n (int): The number of nodes in the graph.
        k (int): Each node is connected to k nearest neighbors in a ring topology.
        p (float): The probability of rewiring each edge.
    """
    return nx.watts_strogatz_graph(n, k, p)

def generate_random_bipartite_graph(n, m, p=0.1):
    """
        Generates a random bipartite graph.

        Parameters:
        n (int): The number of nodes in the first partition.
        m (int): The number of nodes in the second partition.
        p (float): The probability for each pair of nodes (one from each partition) to be connected by an edge.
    """
    return nx.bipartite.random_graph(n, m, p)

def generate_random_regular_graph(n, d=100):
    """
        Generates a random regular graph.

        Parameters:
        d (int): The degree of each node.
        n (int): The number of nodes in the graph.
    """
    return nx.random_regular_graph(d, n)

def generate_powerlaw_graph(n, exponent=2.5):
    """_summary_"""
    return nx.powerlaw_cluster_graph(n, 3, 0.1)

def fast_iso(G1,G2):
    def _fast_iso(G1, G2, result):
        start_time = time.time()
        if G1.is_directed():
            GM = FIDiGraphMatcher(G2, G1)
        else:
            GM = FIGraphMatcher(G2, G1)
        iso_count = len(list(GM.isomorphisms_iter()))
        duration = time.time() - start_time
        result["duration"]=duration
        result["iso_count"]=iso_count
    
    result = {}
    mem_usage = memory_usage((_fast_iso, (G1,G2,result)),) #max_iterations=1
    memory = max(mem_usage) - min(mem_usage)
    return memory, result['duration'], result['iso_count']
    
    
    
def vf2(G1,G2):
    def _vf2(G1, G2, result):
        start_time = time.time()
        if G1.is_directed():
            GM = nxDiGraphMatcher(G2, G1)
        else:
            GM = nxGraphMatcher(G2, G1)
        iso_count = len(list(GM.isomorphisms_iter()))
        duration = time.time() - start_time
        result["duration"]=duration
        result["iso_count"]=iso_count
    result = {}
    mem_usage = memory_usage((_vf2, (G1,G2,result)),) #max_iterations=1
    memory = max(mem_usage) - min(mem_usage)
    return memory, result['duration'], result['iso_count']
    
    
    
def vf2pp(G1,G2):
    def _vf2pp(G1, G2, result):
        start_time = time.time()
        iso_count = len(list(nx.vf2pp_all_isomorphisms(G2, G1, node_label=None)))
        duration = time.time() - start_time
        result["duration"]=duration
        result["iso_count"]=iso_count
    result = {}
    mem_usage = memory_usage((_vf2pp, (G1,G2,result)),) #max_iterations=1
    memory = max(mem_usage) - min(mem_usage)
    return memory, result['duration'], result['iso_count']


def generate_shuffled_graph(G):
    nodes = list(G.nodes)
    random.shuffle(nodes)
    mapping = {old_label: new_label for old_label, new_label in zip(G.nodes, nodes)}
    return nx.relabel_nodes(G, mapping)
#



#benchmatk
BENCHMARKS = {
    "path_graph":nx.path_graph,
    "cycle_graph":nx.cycle_graph,
    "path_graph":nx.path_graph,
    "erdos_renyi_graph":generate_erdos_renyi_graph,
    "Barabasi_Albert":generate_barabasi_albert_graph,
    "Watts_Strogatz":generate_watts_strogatz_graph,
    "Random_Bipartite":generate_random_bipartite_graph,
    "Random_Regular":generate_random_regular_graph,
    "Powerlaw":generate_powerlaw_graph
    }
ALGOS = {
    'fast_iso':fast_iso,
    'vf2':vf2,
    'vf2pp':vf2pp
}
TIMEOUT=3600#1H
NB_PROCS=4

def do_work(algo, benchmark, size):
    G1 = BENCHMARKS[benchmark](size)
    G2 = generate_shuffled_graph(G1)
    memory, duration, iso_count = ALGOS[algo](G1, G2)

    return memory, duration, iso_count, algo, benchmark, size


def main():
    todo = product(ALGOS, BENCHMARKS, range(500, 10500, 500))

    tpool = TimeoutPool(n_jobs=NB_PROCS, timeout=TIMEOUT)
    results = tpool.apply(do_work, todo)

    with open('dump.pickle', 'wb') as f:
        pickle.dump(results, f)



if __name__ == '__main__':
    main()

