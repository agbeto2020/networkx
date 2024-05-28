import itertools
import sys
import time

__all__ = ["GraphMatcher", "DiGraphMatcher", "MultiGraphMatcher", "MultiDiGraphMatcher"]

# class BitArray:
#     """Implementation of bit array"""

#     def __init__(self, _bytes_per_block, _size):
#         _python_int_octect_step = sys.getsizeof(1) - sys.getsizeof(0)
#         if _bytes_per_block % _python_int_octect_step != 0:
#             print("")
#             return
#         self.bytes_per_block = _bytes_per_block
#         self.bits_per_block = 8 * self.bytes_per_block
#         # block value in decimale
#         # self.block_value = pow(2, self.bits_per_block) - 1
#         self.block_value = 0
#         self.bits = []
#         self.nblocks = 0
#         #
#         if _size != None:
#             curr_size = 0
#             while curr_size < _size:
#                 self.bits.append(self.block_value)
#                 self.nblocks += 1
#                 curr_size += self.bits_per_block

#     def set(self, ind, value):
#         curr_size = self.nblocks * self.bits_per_block
#         if ind >= 0:
#             self.resize(ind + 1)
#         if value != self.get(ind):
#             block_index = ind // self.bits_per_block
#             bit_index = ind % self.bits_per_block
#             self.bits[block_index] ^= 1 << bit_index

#     def get(self, ind):
#         curr_size = self.nblocks * self.bits_per_block
#         if ind >= 0 and ind < curr_size:
#             block_index = ind // self.bits_per_block
#             bit_index = ind % self.bits_per_block
#             ## shift to the right
#             return (self.bits[block_index] >> bit_index) & 1
#         return 0

#     def resize(self, new_size):
#         curr_size = self.nblocks * self.bits_per_block
#         while curr_size < new_size:
#             self.bits.append(self.block_value)
#             self.nblocks += 1
#             curr_size += self.bits_per_block


class BitArray:
    """Implementation of bit array"""

    def __init__(self, _size):
        self.size = _size
        self.bits = [False] * _size

    def set(self, ind, value):
        if ind >= self.size:
            self.resize(ind + 1)

        self.bits[ind] = value

    def get(self, ind):
        if ind >= 0 and ind < self.size:
            return self.bits[ind]
        return 0

    def resize(self, new_size):
        self.bits = [False] * new_size
        self.size = new_size


class NodeOrdoring:
    """_summary_

    Returns:
        _type_: _description_
    """

    def __init__(self, id_, node_, prob_, deg_):
        self.id = id_  # Identifier (numerical) of the node
        self.node = node_  # Identifier (numerical) of the node
        self.prob = prob_  # Probability associated with the node
        self.deg = deg_  # degree of the node without counting duplicate nodes
        self.degNeigh = 0  # degree of the node with counting duplicate nodes
        self.degM = 0  # number of neighbors contained in the mapping
        self.degMNeigh = 0  # the sum of the degM attribute of the neighbors (that have neighbors in the mapping)
        self.degMNeighMax = 0  # max of the degM attribute of the neighbors
        self.degMo = (
            0  # the number of neighbors that do not have neighbors in the mapping
        )
        # Flags indicating whether the node is present or a candidate for selection
        self.present = 0
        self.candidate = 0


class NodeCommand:
    """_summary_

    Returns:
        _type_: _description_
    """

    def __init__(
        self,
    ):
        pass

    def compute_node_ordoring(self, graph_matcher):
        self.graph_matcher = graph_matcher
        size = self.graph_matcher.G1.number_of_nodes()
        #
        self.nodes = [None] * size
        self.candidates = []
        self.parents = [None] * size
        self.nodes_order = [-1] * size
        self.nodes_degMNeighMax = [0] * size
        #
        sg_domain = 0
        for ind, node in enumerate(self.graph_matcher.G1_nodes):
            # node = self.graph_matcher.G1_nodes[ind]
            no = NodeOrdoring(
                ind,
                node,
                self.graph_matcher.domains_size[ind],
                self.graph_matcher.G1_degrees[ind],
            )
            self.nodes[ind] = no
            # if self.graph_matcher.domains_size[ind] == 1:
            #     sg_domain += 1
        ##select first node
        selected_node = None
        selected_no = None
        n = 0
        if self.graph_matcher.labelled or sg_domain > 0:
            selected_no = min(self.nodes, key=lambda obj: obj.prob)
        else:
            selected_no = min(self.nodes, key=lambda obj: -obj.deg)

        # selected_no = self.nodes[selected_node]
        #
        self.nodes_order[n] = selected_no.id
        self.nodes_degMNeighMax[n] = selected_no.degMNeighMax
        self.parents[selected_no.id] = None
        #
        self.update_no(selected_no)
        self.compute_degMNeigh_and_degMo()
        n += 1
        # # singleton selection
        # if sg_domain > 0:
        #     for no in self.nodes:
        #         if no.present == 0 and no.prob == 1:
        #             self.nodes_order[n] = no.id
        #             self.nodes_degMNeighMax[n] = no.degMNeighMax
        #             self.update_no(no)
        #             self.compute_degMNeigh_and_degMo()
        #             n += 1
        #
        while n < size:
            if len(self.candidates) != 0:
                if self.graph_matcher.labelled:
                    selected_no = min(
                        self.candidates,
                        key=lambda k: (
                            -k.degM,
                            -k.degNeigh,  # not used for undirect graph
                            -k.degMNeigh,
                            -k.degMo,
                            k.prob,
                            -k.deg,
                        ),
                    )
                else:
                    selected_no = min(
                        self.candidates,
                        key=lambda k: (
                            -k.degM,
                            -k.degNeigh,  # not used for undirect graph
                            -k.degMNeigh,
                            -k.degMo,
                            -k.prob,
                            -k.deg,
                        ),
                    )
            # node which don't have neighbor
            else:
                selected_no = min(self.nodes, key=lambda k: k.present)
            #
            self.nodes_order[n] = selected_no.id
            self.nodes_degMNeighMax[n] = selected_no.degMNeighMax
            #
            self.update_no(selected_no)
            self.compute_degMNeigh_and_degMo()
            n += 1

    def compute_degMNeigh_and_degMo(self):
        max = 0
        node = None
        candidates_tmp = []
        for no in self.candidates:
            if no.present == 0:
                candidates_tmp.append(no)
                max = 0
                # reset
                no.degMNeigh = 0
                no.degMo = 0
                node = no.node
                for neighbor in self.graph_matcher.G1.neighbors(node):
                    neighbor_ind = self.graph_matcher.G1_nodes_ind[neighbor]
                    no_neigh = self.nodes[neighbor_ind]
                    if no_neigh.present == 0:
                        if no_neigh.degM > 0:
                            no.degMNeigh += no_neigh.degM
                            #
                            if no_neigh.degM > max:
                                max = no_neigh.degM
                        else:
                            no.degMo += 1
                #
                no.degMNeighMax = max

        self.candidates = candidates_tmp

    def update_no(self, no):
        no.present = 1
        no.candidate = 1
        no.degM = 0
        node = no.node
        # s
        # node=self.graph_matcher.G1_nodes[no.id]
        for neighbor in self.graph_matcher.G1.neighbors(node):
            neighbor_ind = self.graph_matcher.G1_nodes_ind[neighbor]
            no_neigh = self.nodes[neighbor_ind]
            if no_neigh.present == 0:
                no_neigh.degM += 1
                # no need DegNeigh for undirect graph
            if no_neigh.candidate == 0:
                no_neigh.candidate = 1
                self.candidates.append(no_neigh)
                self.parents[no_neigh.id] = no.id


class GraphMatcher:
    """fastiso implementation for undirect graph"""

    def __init__(self, G1, G2, labelled, node_match=None, edge_match=None):
        self.node_match = node_match
        self.edge_match = edge_match
        #
        self.labelled = labelled
        self.G1 = G2
        self.G2 = G1
        self.G1_nodes = list(self.G1.nodes())
        self.G1_nodes_ind = {node: ind for ind, node in enumerate(self.G1_nodes)}
        self.G2_nodes = list(self.G2.nodes())
        self.G2_nodes_ind = {node: ind for ind, node in enumerate(self.G2_nodes)}
        self.G1_degrees = [self.G1.degree[node] for node in self.G1_nodes]
        self.G2_degrees = [self.G2.degree[node] for node in self.G2_nodes]
        self.mapping = {}
        # Set recursion limit.
        self.old_recursion_limit = sys.getrecursionlimit()
        expected_max_recursion_level = len(self.G2)
        if self.old_recursion_limit < 1.5 * expected_max_recursion_level:
            # Give some breathing room.
            sys.setrecursionlimit(int(1.5 * expected_max_recursion_level))

        # Declare that we will be searching for a graph-graph isomorphism.
        self.test = "iso"
        # self.initialize()

    def is_isomorphic(self):
        """Returns True if G1 and G2 are isomorphic graphs."""
        try:
            x = next(self.isomorphisms_iter())
            return True
        except StopIteration:
            return False

    def isomorphisms_iter(self):
        """Generator over isomorphisms between G1 and G2."""
        # Declare that we are looking for a graph-graph isomorphism.
        self.test = "iso"
        if self.initialize():
            yield from self.match(0)

    def subgraph_is_isomorphic(self):
        """Returns True if a subgraph of G1 is isomorphic to G2."""
        try:
            x = next(self.subgraph_isomorphisms_iter())
            return True
        except StopIteration:
            return False

    def subgraph_is_monomorphic(self):
        """Returns True if a subgraph of G1 is monomorphic to G2."""
        try:
            x = next(self.subgraph_monomorphisms_iter())
            return True
        except StopIteration:
            return False

    def subgraph_isomorphisms_iter(self):
        """Generator over isomorphisms between a subgraph of G1 and G2."""
        # Declare that we are looking for graph-subgraph isomorphism.
        self.test = "sub-iso"
        if self.initialize():
            yield from self.match(0)

    def subgraph_monomorphisms_iter(self):
        """Generator over monomorphisms between a subgraph of G1 and G2."""
        # Declare that we are looking for graph-subgraph monomorphism.
        self.test = "mono"
        if self.initialize():
            yield from self.match(0)

    def match(self, k=0):
        if k == self.G1.number_of_nodes():
            self.mapping = self.state.copy_mapping()
            yield self.mapping
        else:
            G1_node_ind = self.nodes_order[k]
            G1_node = self.G1_nodes[G1_node_ind]
            #
            domain = None
            parent_id = self.parents[G1_node_ind]
            if parent_id == None:
                domain = self.G2_nodes
            else:
                parent_id = self.state.G1_sub_state.m[parent_id]
                parent_node = self.G2_nodes[parent_id]
                domain = self.G2[parent_node]
            #
            G1_node_info = self.state.G1_nodes_info[k]
            G2_node_info = self.state.G2_nodes_info[k]
            #
            if not G1_node_info.also_do:
                self.state.compute_G1_node_info(G1_node_info, G1_node_ind, G1_node)
                self.state.G1_sub_state.add_node(G1_node_ind, G1_node)
            #
            G2_node_ind = 0
            for G2_node in domain:
                G2_node_ind = self.G2_nodes_ind[G2_node]
                if (
                    self.state.G2_sub_state.m[G2_node_ind] == None
                    and self.domains[G1_node_ind][G2_node_ind]
                    and self.syntactic_feasibility(
                        G1_node_ind,
                        G1_node,
                        G2_node_ind,
                        G2_node,
                        G1_node_info,
                        G2_node_info,
                    )
                ):
                    self.state.add_node_pair(G1_node_ind, G1_node, G2_node_ind, G2_node)
                    yield from self.match(k + 1)
                    self.state.remove_node_pair(
                        G1_node_ind, G1_node, G2_node_ind, G2_node
                    )

                if G2_node_info.also_do:
                    G2_node_info.clear()

    def initialize(self):
        """_summary_"""
        G1_size = self.G1.number_of_nodes()
        G2_size = self.G2.number_of_nodes()
        #
        start_time = time.time()
        (
            self.G1_sum_neighbors_degree,
            self.G1_max_neighbors_degree,
        ) = self.compute_node_prop(self.G1, self.G1_nodes_ind, self.G1_degrees)
        (
            self.G2_sum_neighbors_degree,
            self.G2_max_neighbors_degree,
        ) = self.compute_node_prop(self.G2, self.G2_nodes_ind, self.G2_degrees)
        search_time = time.time() - start_time
        print(f"fastiso compute_node_prop_time {search_time}")
        #
        self.domains = [[False] * G2_size for _ in range(G1_size)]
        self.domains_size = [0] * G1_size
        self.parents = None
        self.nodes_order = None
        self.nodes_degMNeighMax = None
        #
        start_time = time.time()
        # compute initial domain
        if not self.compute_initial_domain():
            return False
        search_time = time.time() - start_time
        print(f"fastiso compute_initial_domain_time {search_time}")
        #
        # self.G1_max_neighbors_degree=None
        # self.G1_sum_neighbors_degree=None
        # self.G2_max_neighbors_degree=None
        # self.G2_sum_neighbors_degree=None
        start_time = time.time()
        # compute node ordoring
        self.compute_node_ordoring()
        search_time = time.time() - start_time
        print(f"fastiso compute_node_ordoring_time {search_time}")
        # print(self.parents)
        #
        self.domains_size = None
        start_time = time.time()
        # initialize state
        self.state = GMState(self)
        search_time = time.time() - start_time
        print(f"fastiso state_time {search_time}")
        # self.nodes_degMNeighMax=None
        # matching
        return True

    def compute_node_prop(self, G, G_nodes_ind, G_degrees):
        """_summary_

        Args:
            sel (_type_): _description_
        """
        G_size = G.number_of_nodes()
        # sum of neighbors degree
        sum_neighbors_degree = [0] * G_size
        # max of neighbors degree
        max_neighbors_degree = [0] * G_size
        #
        sum = 0
        max = 0
        deg = 0
        ind = 0
        #
        for node in G:
            for neighbor in G[node]:
                neighbor_ind = G_nodes_ind[neighbor]
                deg = G_degrees[neighbor_ind]
                sum += deg
                if deg > max:
                    max = deg
            sum_neighbors_degree[ind] = sum
            max_neighbors_degree[ind] = max
            ind += 1
            max = 0
            sum = 0
        return sum_neighbors_degree, max_neighbors_degree

    def compute_initial_domain(self):
        #
        # G1_range=list(range(self.G1.number_of_nodes()))
        # G2_range=list(range(self.G2.number_of_nodes() - 1, -1, -1))
        #
        G1_degrees = self.G1_degrees
        G2_degrees = self.G2_degrees
        G1_self_edges = [self.G1.number_of_edges(node, node) for node in self.G1_nodes]
        G2_self_edges = [self.G2.number_of_edges(node, node) for node in self.G2_nodes]
        G1_sum_neighbors = self.G1_sum_neighbors_degree
        G2_sum_neighbors = self.G2_sum_neighbors_degree
        G1_max_neighbors = self.G1_max_neighbors_degree
        G2_max_neighbors = self.G2_max_neighbors_degree
        domain_size = 0
        #
        domain = None
        # Create dictionaries to group nodes based on their characteristics.
        G1_nodes_dict = {}
        G2_nodes_dict = {}
        #
        if self.test == "iso":
            ## Initialize the domains
            for ind, node in enumerate(self.G1_nodes):
                key = (
                    G1_degrees[ind],
                    G1_self_edges[ind],
                    G1_sum_neighbors[ind],
                    G1_max_neighbors[ind],
                )
                if key not in G1_nodes_dict:
                    G1_nodes_dict[key] = []
                G1_nodes_dict[key].append((ind, node))

            for ind, node in enumerate(self.G2_nodes):
                key = (
                    G2_degrees[ind],
                    G2_self_edges[ind],
                    G2_sum_neighbors[ind],
                    G2_max_neighbors[ind],
                )
                if key not in G2_nodes_dict:
                    G2_nodes_dict[key] = []
                G2_nodes_dict[key].append((ind, node))
            # Avoid the worst case.
            if len(G1_nodes_dict) == 1 and len(G2_nodes_dict) == 1:
                if list(G1_nodes_dict.keys())[0] == list(G1_nodes_dict.keys())[0]:
                    G1_size = self.G1.number_of_nodes()
                    G2_size = self.G2.number_of_nodes()
                    self.domains = [[True] * G2_size for _ in range(G1_size)]
                    self.domains_size = [G2_size] * G1_size
                else:
                    return True
            else:
                for key in G1_nodes_dict:
                    if key in G2_nodes_dict:
                        G1_group = G1_nodes_dict[key]
                        G2_group = G2_nodes_dict[key]
                        for ind1, node1 in G1_group:
                            domain = self.domains[ind1]
                            for ind2, node2 in G2_group:
                                if self.compare_node_attr(node1, node2):
                                    domain[ind2] = True
                                    self.domains_size[ind1] += 1
                    else:
                        return False
        #
        elif self.test == "sub-iso":
            for ind1, node1 in enumerate(self.G1_nodes):
                domain = self.domains[ind1]
                for ind2, node2 in enumerate(self.G2_nodes):
                    if (
                        G1_degrees[ind1] <= G2_degrees[ind2]
                        and G1_self_edges[ind1] == G2_self_edges[ind2]
                        and G1_sum_neighbors[ind1] <= G2_sum_neighbors[ind2]
                        and G1_max_neighbors[ind1] <= G2_max_neighbors[ind2]
                        and self.compare_node_attr(node1, node2)
                    ):
                        domain[ind2] = True
                        domain_size += 1
                #
                if domain_size == 0:
                    return False
                self.domains_size[ind1] = domain_size
                domain_size = 0
        #
        else:
            for ind1, node1 in enumerate(self.G1_nodes):
                domain = self.domains[ind1]
                for ind2, node2 in enumerate(self.G2_nodes):
                    if (
                        G1_degrees[ind1] <= G2_degrees[ind2]
                        and G1_self_edges[ind1] <= G2_self_edges[ind2]
                        and G1_sum_neighbors[ind1] <= G2_sum_neighbors[ind2]
                        and G1_max_neighbors[ind1] <= G2_max_neighbors[ind2]
                        and self.compare_node_attr(node1, node2)
                    ):
                        domain[ind2] = True
                        domain_size += 1
                #
                if domain_size == 0:
                    return False
                self.domains_size[ind1] = domain_size
                domain_size = 0

        return True

    def compute_node_ordoring(self):
        no_cmd = NodeCommand()
        no_cmd.compute_node_ordoring(self)
        self.nodes_order = no_cmd.nodes_order
        self.parents = no_cmd.parents
        self.nodes_degMNeighMax = no_cmd.nodes_degMNeighMax

    def syntactic_feasibility(
        self, G1_node_ind, G1_node, G2_node_ind, G2_node, G1_node_info, G2_node_info
    ):
        c = 0
        # nc=0
        if self.test == "iso":
            c = self.state.G2_sub_state.c[G2_node_ind]
            # nc=self.G2.degree[G2_node]-c
            if (
                G1_node_info.c != c
                or G1_node_info.nc != self.G2_degrees[G2_node_ind] - c
                or G1_node_info.c_sum != self.state.G2_sub_state.c_sum[G2_node_ind]
            ):
                return False

            if not self.state.compute_G2_node_info_and_syntactic_feasibility(
                G1_node_ind, G1_node, G2_node_ind, G2_node, G2_node_info
            ):
                return False
            #
            if G1_node_info.num_c != G2_node_info.num_c:
                return False
            #
            for ind in range(len(G1_node_info.DegMNeigh)):
                if G1_node_info.DegMNeigh[ind] != G2_node_info.DegMNeigh[ind]:
                    return False

        elif self.test == "sub-iso":
            c = self.state.G2_sub_state.c[G2_node_ind]
            # nc=self.G2.degree[G2_node]-c
            if (
                G1_node_info.c != c
                or G1_node_info.nc > self.G2_degrees[G2_node_ind] - c
                or G1_node_info.c_sum != self.state.G2_sub_state.c_sum[G2_node_ind]
            ):
                return False

            if not self.state.compute_G2_node_info_and_syntactic_feasibility(
                G1_node_ind, G1_node, G2_node_ind, G2_node, G2_node_info
            ):
                return False
            #
            if G1_node_info.num_c > G2_node_info.num_c:
                return False
            #
            for ind in range(len(G1_node_info.DegMNeigh)):
                if G1_node_info.DegMNeigh[ind] > G2_node_info.DegMNeigh[ind]:
                    return False

        elif self.test == "sub-iso2":
            c = self.state.G2_sub_state.c[G2_node_ind]
            # nc=self.G2.degree[G2_node]-c
            if (
                G1_node_info.c != c
                or G1_node_info.nc > self.G2_degrees[G2_node_ind] - c
                or G1_node_info.c_sum != self.state.G2_sub_state.c_sum[G2_node_ind]
            ):
                return False

            if not self.state.compute_G2_node_info_and_syntactic_feasibility(
                G1_node_ind, G1_node, G2_node_ind, G2_node, G2_node_info
            ):
                return False
            #
            if (
                G1_node_info.num_c > G2_node_info.num_c
                or G1_node_info.num_nc > G2_node_info.num_nc
                or G1_node_info.DegMNeigh_sum > G2_node_info.DegMNeigh_sum
                or G1_node_info.DegMNeigh_max > G2_node_info.DegMNeigh_max
            ):
                return False

        else:
            c = self.state.G2_sub_state.c[G2_node_ind]
            # nc=self.G2.degree[G2_node]-c
            if G1_node_info.c > c or G1_node_info.nc > self.G2.degree[G2_node] - c:
                return False

            if not self.state.compute_G2_node_info_and_syntactic_feasibility(
                G1_node_ind, G1_node, G2_node_ind, G2_node, G2_node_info
            ):
                return False
            #
            if (
                G1_node_info.num_c > G2_node_info.num_c
                or G1_node_info.DegMNeigh_sum > G2_node_info.DegMNeigh_sum
                or G1_node_info.DegMNeigh_max > G2_node_info.DegMNeigh_max
            ):
                return False

        # valid
        return True

    # def semantic_feasibility(self,G1_node_ind, G1_node, G2_node_ind, G2_node):
    #     """_summary_

    #     Returns:
    #         _type_: _description_
    #     """
    #     if self.edge_match is not None:
    #         # Cached lookups
    #         G1nbrs = self.G1_adj[G1_node]
    #         G2nbrs = self.G2_adj[G2_node]
    #         edge_match = self.edge_match
    #         #
    #         for neighbor in G1nbrs:
    #             #
    #     return True

    def compare_edge_attr(
        self, G1_node_neighbors, G1_neighbors, G2_node_neighbors, G2_neighbors
    ):
        if self.edge_match != None:
            return self.edge_match(
                G1_node_neighbors[G1_neighbors], G2_node_neighbors[G2_neighbors]
            )
        return True

    def compare_node_attr(self, G1_node, G2_node):
        return True


class GMState:
    """_summary_"""

    def __init__(self, graph_matcher):
        self.graph_matcher = graph_matcher
        #
        self.G1_sub_state = GMSubState(
            self.graph_matcher.G1, self.graph_matcher.G1_nodes_ind
        )
        self.G2_sub_state = GMSubState(
            self.graph_matcher.G2, self.graph_matcher.G2_nodes_ind
        )
        #
        G1_size = self.graph_matcher.G1.number_of_nodes()
        self.G1_nodes_info = [None] * G1_size
        self.G2_nodes_info = [None] * G1_size
        #
        if self.graph_matcher.test == "iso" or self.graph_matcher.test == "sub-iso":
            size_tmp = 0
            for ind in range(G1_size):
                size_tmp = self.graph_matcher.nodes_degMNeighMax[ind] + 1
                self.G1_nodes_info[ind] = GMNodeInfo(size_tmp)
                self.G2_nodes_info[ind] = GMNodeInfo(size_tmp)
        else:
            for ind in range(G1_size):
                self.G1_nodes_info[ind] = GMNodeInfo(0)
                self.G2_nodes_info[ind] = GMNodeInfo(0)

    def copy_mapping(self):
        mapping = {}
        for G1_node_ind, G1_node in enumerate(self.graph_matcher.G1_nodes):
            G2_node_ind = self.G1_sub_state.m[G1_node_ind]
            G2_node = self.graph_matcher.G2_nodes[G2_node_ind]
            mapping[G1_node] = G2_node

        return mapping

    def add_node_pair(self, G1_node_ind, G1_node, G2_node_ind, G2_node):
        self.G1_sub_state.m[G1_node_ind] = G2_node_ind
        self.G2_sub_state.m[G2_node_ind] = G1_node_ind
        #
        # self.G1_sub_state.add_node(G1_node_ind,G1_node)
        self.G2_sub_state.add_node_(G2_node_ind, G2_node, G1_node_ind)

    def remove_node_pair(self, G1_node_ind, G1_node, G2_node_ind, G2_node):
        self.G2_sub_state.remove_node_(G2_node_ind, G2_node, G1_node_ind)
        #
        self.G1_sub_state.m[G1_node_ind] = None
        self.G2_sub_state.m[G2_node_ind] = None

    def compute_G1_node_info(self, node_info, G1_node_ind, G1_node):
        node_info.also_do = True
        node_info.c = self.G1_sub_state.c[G1_node_ind]
        node_info.nc = self.graph_matcher.G1_degrees[G1_node_ind] - node_info.c
        #
        #
        if self.graph_matcher.test != "mono":
            node_info.c_sum = self.G1_sub_state.c_sum[G1_node_ind]
        #
        neighbor_ind = 0
        neighbor_c = 0
        if self.graph_matcher.test == "iso" or self.graph_matcher.test == "sub-iso":
            for neighbor in self.G1_sub_state.G[G1_node]:
                neighbor_ind = self.G1_sub_state.G_nodes_ind[neighbor]
                # if neighbor not in mapping
                if self.G1_sub_state.m[neighbor_ind] == None:
                    node_info.DegMNeigh[self.G1_sub_state.c[neighbor_ind]] += 1
            #
            node_info.num_c = node_info.nc - node_info.DegMNeigh[0]

        elif self.graph_matcher.test == "sub-iso2":
            for neighbor in self.G1_sub_state.G[G1_node]:
                neighbor_ind = self.G1_sub_state.G_nodes_ind[neighbor]
                # if neighbor not in mapping
                if self.G1_sub_state.m[neighbor_ind] == None:
                    neighbor_c = self.G1_sub_state.c[neighbor_ind]
                    #
                    if neighbor_c == 0:
                        node_info.num_nc += 1
                    #
                    node_info.DegMNeigh_sum += neighbor_c
                    #
                    if neighbor_c > node_info.DegMNeigh_max:
                        node_info.DegMNeigh_max = neighbor_c
            #
            node_info.num_c = node_info.nc - node_info.num_nc

        else:
            for neighbor in self.G1_sub_state.G[G1_node]:
                neighbor_ind = self.G1_sub_state.G_nodes_ind[neighbor]
                # if neighbor not in mapping
                if self.G1_sub_state.m[neighbor_ind] == None:
                    neighbor_c = self.G1_sub_state.c[neighbor_ind]
                    #
                    if neighbor_c == 0:
                        node_info.num_nc += 1
                    # #
                    # node_info.DegMNeigh_sum+=neighbor_c
                    #
                    if neighbor_c > node_info.DegMNeigh_max:
                        node_info.DegMNeigh_max = neighbor_c
            #
            node_info.num_c = node_info.nc - node_info.num_nc

    def compute_G2_node_info_and_syntactic_feasibility(
        self, G1_node_ind, G1_node, G2_node_ind, G2_node, node_info
    ):
        neighbor_ind = 0
        neighbor_c = 0
        neighbor_corr_ind = 0
        neighbor_corr = None
        #
        node_info.also_do = True
        #
        G1_node_neighbors = self.G1_sub_state.G[G1_node]
        G2_node_neighbors = self.G2_sub_state.G[G2_node]
        #
        if self.graph_matcher.test == "iso" or self.graph_matcher.test == "sub-iso":
            for neighbor in G2_node_neighbors:
                neighbor_ind = self.G2_sub_state.G_nodes_ind[neighbor]
                neighbor_corr_ind = self.G2_sub_state.m[neighbor_ind]
                # if neighbor not in mapping
                if neighbor_corr_ind == None:
                    neighbor_c = self.G2_sub_state.c[neighbor_ind]
                    if neighbor_c < len(node_info.DegMNeigh):
                        node_info.DegMNeigh[neighbor_c] += 1

                # neighbor in mapping
                else:
                    neighbor_corr = self.graph_matcher.G1_nodes[neighbor_corr_ind]
                    if self.graph_matcher.G1.number_of_edges(
                        G1_node, neighbor_corr
                    ) != self.graph_matcher.G2.number_of_edges(
                        G2_node, neighbor
                    ) and self.graph_matcher.compare_edge_attr(
                        G1_node_neighbors, neighbor_corr, G2_node_neighbors, neighbor
                    ):
                        return False
            #
            node_info.num_c = (
                self.graph_matcher.G2_degrees[G2_node_ind]
                - self.G2_sub_state.c[G2_node_ind]
                - node_info.DegMNeigh[0]
            )

        elif self.graph_matcher.test == "sub-iso2":
            for neighbor in G2_node_neighbors:
                neighbor_ind = self.G2_sub_state.G_nodes_ind[neighbor]
                neighbor_corr_ind = self.G2_sub_state.m[neighbor_ind]
                # if neighbor not in mapping
                if neighbor_corr_ind == None:
                    neighbor_c = self.G2_sub_state.c[neighbor_ind]
                    #
                    if neighbor_c == 0:
                        node_info.num_nc += 1
                    #
                    node_info.DegMNeigh_sum += neighbor_c
                    #
                    if neighbor_c > node_info.DegMNeigh_max:
                        node_info.DegMNeigh_max = neighbor_c

                # neighbor in mapping
                else:
                    neighbor_corr = self.graph_matcher.G1_nodes[neighbor_corr_ind]
                    if self.graph_matcher.G1.number_of_edges(
                        G1_node, neighbor_corr
                    ) != self.graph_matcher.G2.number_of_edges(
                        G2_node, neighbor
                    ) and self.graph_matcher.compare_edge_attr(
                        G1_node_neighbors, neighbor_corr, G2_node_neighbors, neighbor
                    ):
                        return False

            #
            node_info.num_c = (
                self.graph_matcher.G2_degrees[G2_node_ind]
                - self.G2_sub_state.c[G2_node_ind]
                - node_info.num_nc
            )

        else:
            for neighbor in G1_node_neighbors:
                neighbor_ind = self.G1_sub_state.G_nodes_ind[neighbor]
                neighbor_corr_ind = self.G1_sub_state.m[neighbor_ind]
                # neighbor in mapping
                if neighbor_corr_ind != None:
                    neighbor_corr = self.graph_matcher.G2_nodes[neighbor_corr_ind]
                    if self.graph_matcher.G1.number_of_edges(
                        G1_node, neighbor
                    ) > self.graph_matcher.G2.number_of_edges(
                        G2_node, neighbor_corr
                    ) and self.graph_matcher.compare_edge_attr(
                        G1_node_neighbors, neighbor, G2_node_neighbors, neighbor_corr
                    ):
                        return False
            ##
            for neighbor in self.G2_sub_state.G[G2_node]:
                neighbor_ind = self.G2_sub_state.G_nodes_ind[neighbor]
                neighbor_corr_ind = self.G2_sub_state.m[neighbor_ind]
                # if neighbor not in mapping
                if neighbor_corr_ind == None:
                    neighbor_c = self.G2_sub_state.c[neighbor_ind]
                    #
                    if neighbor_c == 0:
                        node_info.num_nc += 1
                    #
                    # node_info.DegMNeigh_sum+=neighbor_c
                    #
                    if neighbor_c > node_info.DegMNeigh_max:
                        node_info.DegMNeigh_max = neighbor_c
            #
            node_info.num_c = (
                self.graph_matcher.G2_degrees[G2_node_ind]
                - self.G2_sub_state.c[G2_node_ind]
                - node_info.num_nc
            )

        # valide
        return True


class GMSubState:
    """_summary_"""

    def __init__(self, G, G_nodes_ind):
        self.G = G
        self.G_nodes_ind = G_nodes_ind
        G_size = self.G.number_of_nodes()
        # mapping
        self.m = [None] * G_size
        # corresponding to degM in NodeOrdoring class
        self.c = [0] * G_size
        #
        # self.c_in=[0]*G_size
        # self.c_out=[0]*G_size
        # Sum of the IDs (numerical) of the neighbors in the mapping
        self.c_sum = [0] * G_size

    def add_node(self, node_id, node):
        neighbor_ind = 0
        for neighbor in self.G[node]:
            neighbor_ind = self.G_nodes_ind[neighbor]
            self.c[neighbor_ind] += 1
            self.c_sum[neighbor_ind] += node_id

    def remove_node(self, node_id, node):
        neighbor_ind = 0
        for neighbor in self.G[node]:
            neighbor_ind = self.G_nodes_ind[neighbor]
            self.c[neighbor_ind] -= 1
            self.c_sum[neighbor_ind] -= node_id

    def add_node_(self, node_id, node, t_node_id):
        neighbor_ind = 0
        for neighbor in self.G[node]:
            neighbor_ind = self.G_nodes_ind[neighbor]
            self.c[neighbor_ind] += 1
            self.c_sum[neighbor_ind] += t_node_id

    def remove_node_(self, node_id, node, t_node_id):
        neighbor_ind = 0
        for neighbor in self.G[node]:
            neighbor_ind = self.G_nodes_ind[neighbor]
            self.c[neighbor_ind] -= 1
            self.c_sum[neighbor_ind] -= t_node_id

        # self.m[node_id]=-1


class GMNodeInfo:
    """_summary_"""

    def __init__(self, size):
        self.DegMNeigh = [0] * size
        self.also_do = False
        # corresponding to degM in NodeOrdoring class
        self.c = 0
        # number of neighbors not in the mapping
        self.nc = 0
        # self.c_in=0
        # self.c_out=0
        # self.nc_in=0
        # self.nc_out=0
        # Sum of the IDs (numerical) of the neighbors in the mapping
        self.c_sum = 0
        # number of neighbors having neighbors in the mapping
        self.num_c = 0
        # corresponding to degMo in NodeOrdoring class
        self.num_nc = 0
        # for monomorphism and sub-iso2
        self.DegMNeigh_sum = 0
        self.DegMNeigh_max = 0

    def clear(self):
        self.c = 0
        self.nc = 0
        self.num_c = 0
        self.num_nc = 0
        self.DegMNeigh_max = 0
        self.DegMNeigh_sum = 0
        self.also_do = False
        #
        for ind in range(len(self.DegMNeigh)):
            self.DegMNeigh[ind] = 0
