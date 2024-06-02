from networkx.algorithms.isomorphism.isomorphfastiso import GraphMatcher as FIGraphMatcher
from networkx.algorithms.isomorphism.isomorphfastiso import DiGraphMatcher as FIDiGraphMatcher
from networkx.algorithms.isomorphism import GraphMatcher as nxGraphMatcher
from networkx.algorithms.isomorphism import DiGraphMatcher as nxDiGraphMatcher
import networkx as nx
import time
from memory_profiler import profile
from memory_profiler import memory_usage
import threading
import multiprocessing
import csv    
#
def generate_erdos_renyi_graph(n, p=0.1, directed=False):
    """_summary_
    n : Le nombre de nœuds dans le graphe.
    p : La probabilité pour chaque paire de nœuds d'être connectée par une arête.
    directed (optionnel) : Si True, crée un graphe dirigé
    """
    if directed:
        return nx.gnp_random_graph(n, p, directed=True)
    else:
        return nx.gnp_random_graph(n, p)
    
def generate_barabasi_albert_graph(n, m=10):
    """_summary_
    n : Le nombre de nœuds dans le graphe.
    m : Le nombre d'arêtes à attacher de chaque nouveau nœud aux nœuds existants.
    """
    return nx.barabasi_albert_graph(n, m)

def generate_watts_strogatz_graph(n, k=10, p=0.1):
    """_summary_
    n : Le nombre de nœuds dans le graphe.
    k : Chaque nœud est connecté à k voisins proches dans un anneau.
    p : La probabilité de réarranger chaque arête.
    """
    return nx.watts_strogatz_graph(n, k, p)

def generate_random_bipartite_graph(n, m, p=0.1):
    """_summary_
    n : Le nombre de nœuds dans la première partition.
    m : Le nombre de nœuds dans la deuxième partition.
    p : La probabilité pour chaque paire de nœuds (un de chaque partition) d'être connectée par une arête.
    """
    return nx.bipartite.random_graph(n, m, p)

def generate_random_regular_graph(n, d=100):
    """_summary_
    d : Le degré de chaque nœud.
    n : Le nombre de nœuds dans le graphe.
    """
    return nx.random_regular_graph(d, n)

def generate_powerlaw_graph(n, exponent=2.5):
    """_summary_
    n : Le nombre de nœuds dans le graphe.
    exponent (optionnel) : L'exposant de la distribution de degré (paramètre par défaut est 2.5). Cependant, cette valeur n'est pas directement utilisée dans powerlaw_cluster_graph.
    Note : Cette fonction utilise en fait deux autres paramètres : m, le nombre de nouvelles arêtes attachées à chaque nouveau nœud, et p, la probabilité de création d'un triangle. Les valeurs utilisées ici sont m=3 et p=0.1.
    """
    return nx.powerlaw_cluster_graph(n, 3, 0.1)

def fast_iso(G1,G2,result):
    start_time = time.time()
    if G1.is_directed():
        GM = FIDiGraphMatcher(G2, G1)
    else:
        GM = FIGraphMatcher(G2, G1)
    iso_count = len(list(GM.isomorphisms_iter()))
    duration = time.time() - start_time
    result["duration"]=duration
    result["iso_count"]=iso_count
    
def vf2(G1,G2,result):
    start_time = time.time()
    if G1.is_directed():
        GM = nxDiGraphMatcher(G2, G1)
    else:
        GM = nxGraphMatcher(G2, G1)
    iso_count = len(list(GM.isomorphisms_iter()))
    duration = time.time() - start_time
    result["duration"]=duration
    result["iso_count"]=iso_count
    
def vf2pp(G1,G2,result):
    start_time = time.time()
    iso_count = len(list(nx.vf2pp_all_isomorphisms(G2, G1, node_label=None)))
    duration = time.time() - start_time
    result["duration"]=duration
    result["iso_count"]=iso_count
    

def measure_fast_iso(G1,G2,result):
    #start_time = time.time()
    mem_usage = memory_usage((fast_iso, (G1,G2,result)),) #max_iterations=1
    #duration = time.time() - start_time
    result["memory"]=max(mem_usage) - min(mem_usage)
    
def measure_vf2pp(G1,G2,result):
    #start_time = time.time()
    mem_usage = memory_usage((vf2pp, (G1,G2,result)),) #max_iterations=1
    #duration = time.time() - start_time
    result["memory"]=max(mem_usage) - min(mem_usage)
    
def measure_vf2(G1,G2,result):
    #start_time = time.time()
    mem_usage = memory_usage((vf2, (G1,G2,result)),) #max_iterations=1
    #duration = time.time() - start_time
    result["memory"]=max(mem_usage) - min(mem_usage)
    

def compare_fast_iso(G1,G2,result,timeout_duration):
    process = multiprocessing.Process(target=measure_fast_iso,args=(G1,G2,result,))
    process.start()
    process.join(timeout_duration)  # Attends la fin du processus avec un timeout

    if process.is_alive():
        result["duration"]=timeout_duration
        result["memory"]=0
        result["iso_count"]=0
        process.terminate()  # Terminer le processus s'il dépasse le timeout

    return result

def compare_vf2(G1,G2,result,timeout_duration):
    process = multiprocessing.Process(target=measure_vf2,args=(G1,G2,result,))
    process.start()
    process.join(timeout_duration)# Attends la fin du processus avec un timeout

    if process.is_alive():
        result["duration"]=timeout_duration
        result["memory"]=0
        result["iso_count"]=0
        process.terminate()  # Terminer le processus s'il dépasse le timeout

    return result

def compare_vf2pp(G1,G2,result,timeout_duration):
    process = multiprocessing.Process(target=measure_vf2pp,args=(G1,G2,result,))
    process.start()
    process.join(timeout_duration)# Attends la fin du processus avec un timeout

    if process.is_alive():
        result["duration"]=timeout_duration
        result["memory"]=0
        result["iso_count"]=0
        process.terminate()  # Terminer le processus s'il dépasse le timeout

    return result

def save_file(file_name,_size,_memory,_time):
    with open(file_name, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Size", "Memory", "Time"])
        for size, mem, time in zip(_size,_memory,_time):
            writer.writerow([size, mem, time])
    

#benchmatk
benchmarkx = {
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
#
timeout_duration=3600
result = multiprocessing.Manager().dict()

# Parcourir les benchmarks et comparer les algorithmes
for name, generator in benchmarkx.items():
    print(f"Testing {name} graph...")
    #
    fastiso_memory=[]
    fastiso_time=[]
    #
    vf2_memory=[]
    vf2_time=[]
    #
    vf2pp_memory=[]
    vf2pp_time=[]
    sizes=[]
    #
    for size in range(500, 2000, 500):
        G1 = generator(size)
        G2 = generator(size)
        
        compare_vf2(G1, G2, result,timeout_duration)
        vf2_memory.append(result["memory"])
        vf2_time.append(result["duration"])
        
        compare_vf2pp(G1,G2,result,timeout_duration)
        vf2pp_memory.append(result["memory"])
        vf2pp_time.append(result["duration"])
        
        compare_fast_iso(G1,G2,result,timeout_duration)
        fastiso_memory.append(result["memory"])
        fastiso_time.append(result["duration"])
        
        sizes.append(size)
    ##
    save_file(name+"_fastiso_results.csv", sizes, fastiso_memory, fastiso_time)
    save_file(name+"_vf2_results.csv", sizes, vf2_memory, vf2_time)
    save_file(name+"_vf2pp_results.csv", sizes, vf2pp_memory, vf2pp_time)