import sys
import time

__all__ = ["GraphMatcher", "DiGraphMatcher", "MultiGraphMatcher", "MultiDiGraphMatcher"]

"""_summary_
definitions : 
- Number of neighbors: This is the number of distinct nodes to which a node is connected by one or more edges.
- Number of edges (degree): This is the total number of edges connected to a node, including multiple edges.

let G is a simple graph, and n, node in G :
Number_of_neighbors(n) equal to Number_of_edges(n)
if G is a multi-graph Number_of_neighbors(n) not necessacary equal to Number_of_edges(n)
Number_of_neighbors(n) <= Number_of_edges(n)
"""

"""""" """""" """""" """""" """""" """""" """""" """""" """""" """""" """""" """""" """""" """""" """""" """""" """""" """""" """""" """""" """""" """

                                                         graph

""" """""" """""" """""" """""" """""" """""" """""" """""" """""" """""" """""" """""" """""" """""" """""" """""" """""" """""" """""" """""" """"""


class NodeOrdoringProp:
    """Class representing the properties used for node ordering in graph matching.

    Attributes:
        id : Identifier (numerical) of the node.
        node : Identifier of the node.
        prob : Probability associated with the node.
        deg : Degree of the node.
        degNeigh : Number of edges contained in the mapping.
        degM : Number of neighbors contained in the mapping.
        degMNeigh : Sum of the degM attribute of the neighbors that have neighbors in the mapping.
        degMNeighMax : Maximum of the degM attribute of the neighbors.
        degMo : Number of neighbors that do not have neighbors in the mapping.
        present : Flag indicating if the node is present (1) in mapping or not (0).
        candidate : Flag indicating if the node is a candidate for selection (1) or not (0).
    """

    def __init__(self, id_, node_, prob_, deg_, degMo_):
        self.id = id_
        self.node = node_
        self.prob = prob_
        self.deg = deg_
        self.degNeigh = 0
        self.degM = 0
        self.degMNeigh = 0
        self.degMNeighMax = 0
        self.degMo = degMo_
        self.present = 0
        self.candidate = 0


class NodeCommand:
    """
    Compute the node matching order using attributes of the NodeOrdoringProp class.
    """

    def __init__(
        self,
    ):
        pass

    #
    def compute_node_ordoring(self, graph_matcher):
        """Computes the node ordering for graph matching.

        Parameters:
            graph_matcher.
        """
        self.graph_matcher = graph_matcher
        size = self.graph_matcher.G1.number_of_nodes()
        #
        self.nodes = [None] * size
        self.candidates = []
        self.parents = [None] * size
        self.nodes_order = [-1] * size
        self.nodes_degMNeighMax = [0] * size
        #
        for ind, node in enumerate(self.graph_matcher.G1_nodes):
            # node = self.graph_matcher.G1_nodes[ind]
            no = NodeOrdoringProp(
                ind,
                node,
                self.graph_matcher.node_prob[ind],
                self.graph_matcher.G1_degrees[ind],
                len(self.graph_matcher.G1[node]),
            )
            self.nodes[ind] = no
        ##select first node
        selected_no = None
        n = 0
        if self.graph_matcher.labelled:
            selected_no = min(self.nodes, key=lambda obj: obj.prob)
        else:
            selected_no = min(self.nodes, key=lambda obj: -obj.deg)
        #
        self.nodes_order[n] = selected_no.id
        self.nodes_degMNeighMax[n] = selected_no.degMNeighMax
        self.parents[selected_no.id] = None
        #
        self.update_no(selected_no)
        n += 1

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
                            k.prob,
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
            n += 1

    def update_degMNeigh_and_degMo(self, no):
        node = no.node
        degM = no.degM - 1
        for neighbor in self.graph_matcher.G1.neighbors(node):
            neighbor_ind = self.graph_matcher.G1_nodes_ind[neighbor]
            no_neigh = self.nodes[neighbor_ind]
            if no_neigh.present == 0:
                if degM == 0:
                    no_neigh.degMo -= 1
                else:
                    no_neigh.degMNeigh += 1
                #
                if no.degM > no_neigh.degMNeighMax:
                    no_neigh.degMNeighMax = no.degM

    def update_no(self, no):
        """Marks a node as present and updates its neighbors' properties.

        Parameters:
            no: NodeOrdoringProp corresponding to the node to be updated.
        """
        no.present = 1
        no.candidate = 1
        no.degM = 0
        node = no.node
        #
        for neighbor in self.graph_matcher.G1.neighbors(node):
            neighbor_ind = self.graph_matcher.G1_nodes_ind[neighbor]
            no_neigh = self.nodes[neighbor_ind]
            if no_neigh.present == 0:
                no_neigh.degM += 1
                no_neigh.degNeigh += len(self.graph_matcher.G1[node][neighbor])
                self.update_degMNeigh_and_degMo(no_neigh)
                # no need DegNeigh for undirect graph
            if no_neigh.candidate == 0:
                no_neigh.candidate = 1
                self.candidates.append(no_neigh)
                self.parents[no_neigh.id] = no.id

        #
        self.candidates = [no for no in self.candidates if no.present == 0]


class GraphMatcher:
    """fastiso implementation for matching undirect graphs"""

    def __init__(self, G1, G2, node_label=None, node_match=None, edge_match=None):
        """Initialize GraphMatcher.

        Parameters
        ----------
        G1,G2: NetworkX Graph or MultiGraph instances.
           The two graphs to check for isomorphism or monomorphism.

        Examples
        --------
        To create a GraphMatcher which checks for isomorphism:

        >>> from networkx.algorithms import isomorphism
        >>> G1 = nx.path_graph(4)
        >>> G2 = nx.path_graph(4)
        >>> GM = isomorphism.GraphMatcher(G1, G2)
        """
        self.node_match = node_match
        self.edge_match = edge_match
        #
        self.node_label = node_label
        self.labelled = False
        if self.node_label != None:
            self.labelled == True
        #
        self.G1 = G2
        self.G2 = G1
        self.G1_nodes = list(self.G1.nodes())
        self.G1_nodes_ind = {node: ind for ind, node in enumerate(self.G1_nodes)}
        self.G2_nodes = list(self.G2.nodes())
        self.G2_nodes_ind = {node: ind for ind, node in enumerate(self.G2_nodes)}
        #
        self.G1_degrees = [self.G1.degree[node] for node in self.G1_nodes]
        self.G2_degrees = [self.G2.degree[node] for node in self.G2_nodes]
        #
        self.G1_self_edges = [
            self.G1.number_of_edges(node, node) for node in self.G1_nodes
        ]
        self.G2_self_edges = [
            self.G2.number_of_edges(node, node) for node in self.G2_nodes
        ]
        #
        self.mapping = {}
        # Set recursion limit.
        self.old_recursion_limit = sys.getrecursionlimit()
        expected_max_recursion_level = len(self.G2)
        if self.old_recursion_limit < 1.5 * expected_max_recursion_level:
            # Give some breathing room.
            sys.setrecursionlimit(int(1.5 * expected_max_recursion_level))

        # Declare that we will be searching for a graph-graph isomorphism.
        self.test = "iso"

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
                    # and self.domains[G1_node_ind][G2_node_ind]
                    and self._feasibility(
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

    def initialize_sate(self):
        self.state = GMState(self)

    def initialize(self):
        """_summary_"""
        G1_size = self.G1.number_of_nodes()
        G2_size = self.G2.number_of_nodes()
        # fast check
        if self.test == "iso":
            if self.G1.order() != self.G2.order():
                return False
            if sorted(self.G1_degrees) != sorted(self.G2_degrees):
                return False
        else:
            if G1_size > G2_size:
                return False

            if self.G1.number_of_edges() > self.G2.number_of_edges():
                return False

            if (
                sorted(self.G1_degrees)[G1_size - 1]
                > sorted(self.G2_degrees)[G2_size - 1]
            ):
                return False
        #
        (
            self.G1_sum_neighbors_degree,
            self.G1_max_neighbors_degree,
        ) = self.compute_node_prop(
            self.G1, self.G1_nodes, self.G1_nodes_ind, self.G1_degrees
        )
        (
            self.G2_sum_neighbors_degree,
            self.G2_max_neighbors_degree,
        ) = self.compute_node_prop(
            self.G2, self.G2_nodes, self.G2_nodes_ind, self.G2_degrees
        )
        #
        # node probability
        self.node_prob = [0] * G1_size
        self.parents = None
        self.nodes_order = None
        self.nodes_degMNeighMax = None
        #
        # compute initial domain
        if not self.compute_node_probability():
            return False
        #
        # self.G1_max_neighbors_degree=None
        # self.G1_sum_neighbors_degree=None
        # self.G2_max_neighbors_degree=None
        # self.G2_sum_neighbors_degree=None
        # compute node ordoring
        self.compute_node_ordoring()
        #
        self.node_prob = None
        # initialize state
        self.initialize_sate()
        # self.nodes_degMNeighMax=None
        # matching
        return True

    def compute_node_prop(self, G, G_nodes, G_nodes_ind, G_degrees):
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
        for ind, node in enumerate(G_nodes):
            neighbors = G[node]
            neighbor_degrees = [
                G_degrees[G_nodes_ind[neighbor]] for neighbor in neighbors
            ]
            sum_neighbors_degree[ind] = sum(neighbor_degrees)
            max_neighbors_degree[ind] = max(neighbor_degrees)

        return sum_neighbors_degree, max_neighbors_degree

    def compute_node_probability(self):
        G1_size = self.G1.number_of_nodes()
        G2_size = self.G2.number_of_nodes()
        #
        max_degree = max(self.G2_degrees) + 1
        degree_counter = [0] * max_degree
        label_counter = {}
        #
        degree = 0
        node_label = None
        if self.node_label != None:
            for ind, node in enumerate(self.G2_nodes):
                degree = self.G2_degrees[ind]
                degree_counter[degree] += 1
                #
                node_label = self.G2.nodes[node][self.node_label]
                if node_label in label_counter:
                    label_counter[node_label] += 1
                else:
                    label_counter[node_label] = 1
            #
            label_counter = {k: v / G2_size for k, v in label_counter.items()}
        else:
            for ind, node in enumerate(self.G2_nodes):
                degree = self.G2_degrees[ind]
                degree_counter[degree] += 1
        #
        degree_counter = [x / G2_size for x in degree_counter]
        # calcul prob
        prob = 0
        if self.test == "iso":
            for ind, node in enumerate(self.G1_nodes):
                degree = self.G1_degrees[ind]
                if self.node_label == None:
                    prob = degree_counter[degree]
                else:
                    node_label = self.G1.nodes[node][self.node_label]
                    prob == degree_counter[degree] * label_counter[node_label]
                #
                if prob == 0:
                    return False
                self.node_prob[ind] = prob
        else:
            prob_deg_sum = 0
            for ind, node in enumerate(self.G1_nodes):
                degree = self.G1_degrees[ind]
                prob_deg_sum = sum(
                    [value for i, value in enumerate(degree_counter) if i >= degree]
                )
                if self.node_label == None:
                    prob = prob_deg_sum
                else:
                    prob = prob_deg_sum * label_counter[node_label]
                #
                if prob == 0:
                    return False
                self.node_prob[ind] = prob
        return True

    #
    def compute_node_ordoring(self):
        no_cmd = NodeCommand()
        no_cmd.compute_node_ordoring(self)
        self.nodes_order = no_cmd.nodes_order
        self.parents = no_cmd.parents
        self.nodes_degMNeighMax = no_cmd.nodes_degMNeighMax

    def _feasibility(
        self, G1_node_ind, G1_node, G2_node_ind, G2_node, G1_node_info, G2_node_info
    ):
        c = 0
        # nc=0
        if self.test == "iso":
            c = self.state.G2_sub_state.c[G2_node_ind]
            # nc=self.G2.degree[G2_node]-c
            if (
                G1_node_info.c != c
                # or G1_node_info.nc != self.G2_degrees[G2_node_ind] - c
                or G1_node_info.nc != len(self.G2[G2_node]) - c
                or G1_node_info.c_sum != self.state.G2_sub_state.c_sum[G2_node_ind]
                # prop
                or self.G1_max_neighbors_degree[G1_node_ind]
                != self.G2_max_neighbors_degree[G2_node_ind]
                or self.G1_max_neighbors_degree[G1_node_ind]
                != self.G2_max_neighbors_degree[G2_node_ind]
                or self.G1_self_edges[G1_node_ind] != self.G2_self_edges[G2_node_ind]
            ):
                return False

            if not self.compare_node_attr(G1_node, G2_node):
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
                # or G1_node_info.nc > self.G2_degrees[G2_node_ind] - c
                or G1_node_info.nc > len(self.G2[G2_node]) - c
                or G1_node_info.c_sum != self.state.G2_sub_state.c_sum[G2_node_ind]
                # prop
                or self.G1_max_neighbors_degree[G1_node_ind]
                > self.G2_max_neighbors_degree[G2_node_ind]
                or self.G1_max_neighbors_degree[G1_node_ind]
                > self.G2_max_neighbors_degree[G2_node_ind]
                or self.G1_self_edges[G1_node_ind] != self.G2_self_edges[G2_node_ind]
            ):
                return False

            if not self.compare_node_attr(G1_node, G2_node):
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
                # or G1_node_info.nc > self.G2_degrees[G2_node_ind] - c
                or G1_node_info.nc > len(self.G2[G2_node]) - c
                or G1_node_info.c_sum != self.state.G2_sub_state.c_sum[G2_node_ind]
                # prop
                or self.G1_max_neighbors_degree[G1_node_ind]
                > self.G2_max_neighbors_degree[G2_node_ind]
                or self.G1_max_neighbors_degree[G1_node_ind]
                > self.G2_max_neighbors_degree[G2_node_ind]
                or self.G1_self_edges[G1_node_ind] != self.G2_self_edges[G2_node_ind]
            ):
                return False

            if not self.compare_node_attr(G1_node, G2_node):
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
            if (
                G1_node_info.c > c
                # or G1_node_info.nc > self.G2.degree[G2_node] - c
                or G1_node_info.nc > len(self.G2[G2_node]) - c
                # prop
                or self.G1_max_neighbors_degree[G1_node_ind]
                > self.G2_max_neighbors_degree[G2_node_ind]
                or self.G1_max_neighbors_degree[G1_node_ind]
                > self.G2_max_neighbors_degree[G2_node_ind]
                or self.G1_self_edges[G1_node_ind] > self.G2_self_edges[G2_node_ind]
            ):
                return False

            if not self.compare_node_attr(G1_node, G2_node):
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

    def compare_edge_attr(
        self, G1_node_neighbors, G1_neighbor, G2_node_neighbors, G2_neighbor
    ):
        # if self.edge_match != None:
        #     return self.edge_match(
        #         G1_node_neighbors[G1_neighbor], G2_node_neighbors[G2_neighbor]
        #     )
        return True

    def compare_node_attr(self, G1_node, G2_node):
        # if self.node_match != None:
        #    return self.node_match(self.G1.nodes[G1_node],self.G2.nodes[G2_node])
        return True


class GMState:
    """Internal representation of state"""

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
            mapping[G2_node] = G1_node

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
        """Compute cutting rule informations for the node "G1_node" in G1

        Args:
            node_info (GMNodeInfo): _description_
            G1_node_ind (int): numerical id for G1_node
            G1_node (any): _description_
        """
        node_info.also_do = True
        node_info.c = self.G1_sub_state.c[G1_node_ind]
        # node_info.nc = self.graph_matcher.G1_degrees[G1_node_ind] - node_info.c
        node_info.nc = len(self.graph_matcher.G1[G1_node]) - node_info.c
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
                    # if self.graph_matcher.G1.number_of_edges(
                    #     G1_node, neighbor_corr
                    # ) != self.graph_matcher.G2.number_of_edges(
                    #     G2_node, neighbor
                    # ) and self.graph_matcher.compare_edge_attr(
                    #     G1_node_neighbors, neighbor_corr, G2_node_neighbors, neighbor
                    # ):
                    #     return False
                    if (
                        neighbor_corr not in self.graph_matcher.G1[G1_node]
                        or
                        # not self.graph_matcher.compare_edge_attr(
                        #  G1_node_neighbors, neighbor_corr, G2_node_neighbors, neighbor
                        # )
                        not self.check_edge(
                            G1_node,
                            G2_node,
                            G1_node_neighbors,
                            neighbor_corr,
                            G2_node_neighbors,
                            neighbor,
                        )
                    ):
                        return False
            #
            node_info.num_c = (
                len(self.graph_matcher.G2[G2_node])
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
                    # if self.graph_matcher.G1.number_of_edges(
                    #     G1_node, neighbor_corr
                    # ) != self.graph_matcher.G2.number_of_edges(
                    #     G2_node, neighbor
                    # ) and self.graph_matcher.compare_edge_attr(
                    #     G1_node_neighbors, neighbor_corr, G2_node_neighbors, neighbor
                    # ):
                    #     return False
                    if (
                        neighbor_corr not in self.graph_matcher.G1[G1_node]
                        or
                        # not self.graph_matcher.compare_edge_attr(
                        #  G1_node_neighbors, neighbor_corr, G2_node_neighbors, neighbor
                        # )
                        not self.check_edge(
                            G1_node,
                            G2_node,
                            G1_node_neighbors,
                            neighbor_corr,
                            G2_node_neighbors,
                            neighbor,
                        )
                    ):
                        return False

            #
            node_info.num_c = (
                len(self.graph_matcher.G2[G2_node])
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
                    # if self.graph_matcher.G1.number_of_edges(
                    #     G1_node, neighbor
                    # ) > self.graph_matcher.G2.number_of_edges(
                    #     G2_node, neighbor_corr
                    # ) and self.graph_matcher.compare_edge_attr(
                    #     G1_node_neighbors, neighbor, G2_node_neighbors, neighbor_corr
                    # ):
                    #     return False
                    if (
                        neighbor_corr not in self.graph_matcher.G2[G2_node]
                        or
                        # not self.graph_matcher.compare_edge_attr(
                        #  G1_node_neighbors, neighbor, G2_node_neighbors, neighbor_corr
                        # )
                        not self.check_edge(
                            G1_node,
                            G2_node,
                            G1_node_neighbors,
                            neighbor,
                            G2_node_neighbors,
                            neighbor_corr,
                        )
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
                len(self.graph_matcher.G2[G2_node])
                - self.G2_sub_state.c[G2_node_ind]
                - node_info.num_nc
            )

        # valide
        return True

    def check_edge(
        self,
        G1_node,
        G2_node,
        G1_node_neighbors,
        G1_neighbor,
        G2_node_neighbors,
        G2_neighbor,
    ):
        return self.graph_matcher.compare_edge_attr(
            G1_node_neighbors, G1_neighbor, G2_node_neighbors, G2_neighbor
        )


class GMSubState:
    """class for managing feasibility sets and the state of the mapping."""

    def __init__(self, G, G_nodes_ind):
        self.G = G
        self.G_nodes_ind = G_nodes_ind
        G_size = self.G.number_of_nodes()
        # mapping
        self.m = [None] * G_size
        # corresponding to degM in NodeOrdoring class (Number of neighbors contained in the mapping)
        self.c = [0] * G_size
        # Sum of the IDs (numerical) of the neighbors in the mapping
        self.c_sum = [0] * G_size

    # upadate Feasibility Sets

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


class GMNodeInfo:
    """
    A class to store and manage node information for cutting rule.

    Attributes
    ----------
    DegMNeigh (only for isomorphism and sub-graph isomorphism) : list of int
        A list where DegMNeigh[i]=val implies that the node has 'val' neighbors
        who have 'i' neighbors in the mapping. For instance, DegMNeigh[0]=2
        implies that the node has 2 neighbors who have no neighbors in the mapping.
    c : int
        Corresponds to degM in the NodeOrdoring class. It represents the number
        of neighbors contained in the mapping.

    nc : int
        Represents the number of neighbors not in the mapping.

    c_sum : int
        The sum of the IDs (numerical) of the neighbors in the mapping.

    num_c : int
        The number of neighbors having neighbors in the mapping.

    also_do : bool
        A flag used for additional processing if required.

    num_nc (only for monomorphism and sub-iso2) : int
        Corresponds to degMo in the NodeOrdoring class. It represents the number
        of neighbors that do not have neighbors in the mapping.
        DegMNeigh[0] equal to num_nc

    DegMNeigh_sum (only for monomorphism and sub-iso2) : int
        The sum of the degM attribute of the neighbors.

    DegMNeigh_max (only for monomorphism and sub-iso2) : int
        The maximum of the degM attribute of the neighbors.
    """

    def __init__(self, size):
        """
        Initialize the GMNodeInfo instance.

        Parameters
        ----------
        size : int
            The size of the DegMNeigh list.
        """
        self.DegMNeigh = [0] * size  # Initialize DegMNeigh with zeros
        self.c = 0  # Number of neighbors in the mapping
        self.nc = 0  # Number of neighbors not in the mapping
        self.c_sum = 0  # Sum of IDs of neighbors in the mapping
        self.num_c = 0  # Number of neighbors having neighbors in the mapping
        self.also_do = False  # Additional processing flag

        # Only for monomorphism and sub-isomorphism
        self.num_nc = 0  # Number of neighbors that do not have neighbors in the mapping
        self.DegMNeigh_sum = 0  # Sum of the degM attribute of the neighbors
        self.DegMNeigh_max = 0  # Maximum of the degM attribute of the neighbors

    def clear(self):
        """
        Reset all attributes to their default values.
        """
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


"""""" """""" """""" """""" """""" """""" """""" """""" """""" """""" """""" """""" """""" """""" """""" """""" """""" """""" """""" """""" """""" """

                                                            multi-graph

""" """""" """""" """""" """""" """""" """""" """""" """""" """""" """""" """""" """""" """""" """""" """""" """""" """""" """""" """""" """""" """"""


class MultiGraphMatcher(GraphMatcher):
    def __init__(self, G1, G2, node_label=None, node_match=None, edge_match=None):
        super().__init__(G1, G2, node_label, node_match, edge_match)

    def initialize_sate(self):
        self.state = MultiGMState(self)

    def _feasibility(
        self, G1_node_ind, G1_node, G2_node_ind, G2_node, G1_node_info, G2_node_info
    ):
        #
        c_e = self.state.G2_sub_state.c_e[G2_node_ind]
        if self.test == "iso":
            if (
                G1_node_info.c_e != c_e
                or G1_node_info.nc_e != self.G2_degrees[G2_node_ind] - c_e
            ):
                return False
        elif self.test == "sub-iso" or self.test == "sub-iso2":
            if (
                G1_node_info.c_e != c_e
                or G1_node_info.nc_e > self.G2_degrees[G2_node_ind] - c_e
            ):
                return False
        else:
            if (
                G1_node_info.c_e > c_e
                or G1_node_info.nc_e > self.G2_degrees[G2_node_ind] - c_e
            ):
                return False
        #
        return super()._feasibility(
            G1_node_ind, G1_node, G2_node_ind, G2_node, G1_node_info, G2_node_info
        )


class MultiGMState(GMState):
    """_summary_"""

    def __init__(self, graph_matcher):
        self.graph_matcher = graph_matcher
        #
        self.G1_sub_state = MultiGMSubState(
            self.graph_matcher.G1, self.graph_matcher.G1_nodes_ind
        )
        self.G2_sub_state = MultiGMSubState(
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
                self.G1_nodes_info[ind] = MultiGMNodeInfo(size_tmp)
                self.G2_nodes_info[ind] = MultiGMNodeInfo(size_tmp)
        else:
            for ind in range(G1_size):
                self.G1_nodes_info[ind] = MultiGMNodeInfo(0)
                self.G2_nodes_info[ind] = MultiGMNodeInfo(0)

    def compute_G1_node_info(self, node_info, G1_node_ind, G1_node):
        #
        super().compute_G1_node_info(node_info, G1_node_ind, G1_node)
        node_info.c_e = self.G1_sub_state.c_e[G1_node_ind]
        node_info.nc_e = self.graph_matcher.G1_degrees[G1_node_ind] - node_info.c_e

    def compute_G2_node_info_and_syntactic_feasibility(
        self, G1_node_ind, G1_node, G2_node_ind, G2_node, node_info
    ):
        return super().compute_G2_node_info_and_syntactic_feasibility(
            G1_node_ind, G1_node, G2_node_ind, G2_node, node_info
        )

    def check_edge(
        self,
        G1_node,
        G2_node,
        G1_node_neighbors,
        G1_neighbor,
        G2_node_neighbors,
        G2_neighbor,
    ):
        if self.graph_matcher.test != "mono":
            return len(G1_node_neighbors[G1_neighbor]) == len(
                G2_node_neighbors[G2_neighbor]
            ) and super().check_edge(
                G1_node,
                G2_node,
                G1_node_neighbors,
                G1_neighbor,
                G2_node_neighbors,
                G2_neighbor,
            )
        else:
            return len(G1_node_neighbors[G1_neighbor]) <= len(
                G2_node_neighbors[G2_neighbor]
            ) and super().check_edge(
                G1_node,
                G2_node,
                G1_node_neighbors,
                G1_neighbor,
                G2_node_neighbors,
                G2_neighbor,
            )


class MultiGMSubState(GMSubState):
    """_summary_"""

    def __init__(self, G, G_nodes_ind):
        super().__init__(G, G_nodes_ind)
        # Number of edges in the mapping
        self.c_e = [0] * self.G.number_of_nodes()

    # upadate Feasibility Sets
    def add_node(self, node_id, node):
        neighbor_ind = 0
        for neighbor in self.G[node]:
            neighbor_ind = self.G_nodes_ind[neighbor]
            self.c[neighbor_ind] += 1
            self.c_e[neighbor_ind] += len(self.G[node][neighbor])
            self.c_sum[neighbor_ind] += node_id

    def add_node_(self, node_id, node, t_node_id):
        neighbor_ind = 0
        for neighbor in self.G[node]:
            neighbor_ind = self.G_nodes_ind[neighbor]
            self.c[neighbor_ind] += 1
            self.c_e[neighbor_ind] += len(self.G[node][neighbor])
            self.c_sum[neighbor_ind] += t_node_id

    def remove_node(self, node_id, node):
        neighbor_ind = 0
        for neighbor in self.G[node]:
            neighbor_ind = self.G_nodes_ind[neighbor]
            self.c[neighbor_ind] -= 1
            self.c_e[neighbor_ind] -= len(self.G[node][neighbor])
            self.c_sum[neighbor_ind] -= node_id

    def remove_node_(self, node_id, node, t_node_id):
        neighbor_ind = 0
        for neighbor in self.G[node]:
            neighbor_ind = self.G_nodes_ind[neighbor]
            self.c[neighbor_ind] -= 1
            self.c_e[neighbor_ind] -= len(self.G[node][neighbor])
            self.c_sum[neighbor_ind] -= t_node_id


class MultiGMNodeInfo(GMNodeInfo):
    """_summary_"""

    def __init__(self, size):
        super().__init__(size)
        # Number of edges in the mapping
        self.c_e = 0
        # Number of edges not in the mapping
        self.nc_e = 0

    def clear(self):
        super().clear()
        self.c_e = 0
        self.nc_e = 0


"""""" """""" """""" """""" """""" """""" """""" """""" """""" """""" """""" """""" """""" """""" """""" """""" """""" """""" """""" """""" """""" """

                                                            di-graph

""" """""" """""" """""" """""" """""" """""" """""" """""" """""" """""" """""" """""" """""" """""" """""" """""" """""" """""" """""" """""" """"""


class DiNodeCommand(NodeCommand):
    def update_no(self, no):
        """Marks a node as present and updates its neighbors' properties.

        Parameters:
            no: NodeOrdoringProp corresponding to the node to be updated.
        """
        no.present = 1
        no.candidate = 1
        no.degM = 0
        node = no.node
        #
        for neighbor in self.graph_matcher.G1.neighbors(node):
            neighbor_ind = self.graph_matcher.G1_nodes_ind[neighbor]
            no_neigh = self.nodes[neighbor_ind]
            if no_neigh.present == 0:
                no_neigh.degM += 1
                no_neigh.degNeigh += self.graph_matcher.G1_o.number_of_edges(
                    node, neighbor
                ) + self.graph_matcher.G1_o.number_of_edges(neighbor, node)
                self.update_degMNeigh_and_degMo(no_neigh)
                # no need DegNeigh for undirect graph
            if no_neigh.candidate == 0:
                no_neigh.candidate = 1
                self.candidates.append(no_neigh)
                self.parents[no_neigh.id] = no.id

        #
        self.candidates = [no for no in self.candidates if no.present == 0]


class DiGraphMatcher(GraphMatcher):
    """_summary_"""

    def __init__(self, G1, G2, node_label=None, node_match=None, edge_match=None):
        super().__init__(G1, G2, node_label, node_match, edge_match)
        #
        self.G1_degrees_in = [self.G1.in_degree[node] for node in self.G1_nodes]
        self.G1_degrees_out = [self.G1.out_degree[node] for node in self.G1_nodes]
        #
        self.G2_degrees_in = [self.G2.in_degree[node] for node in self.G2_nodes]
        self.G2_degrees_out = [self.G2.out_degree[node] for node in self.G2_nodes]

    def initialize_sate(self):
        self.state = DiGMState(self)

    def initialize(self):
        #
        self.G1_o = self.G1
        self.G2_o = self.G2
        # create undirect graph view
        self.G1 = self.G1_o.to_undirected(as_view=True)
        self.G2 = self.G2_o.to_undirected(as_view=True)
        #
        return super().initialize()

    def compute_node_ordoring(self):
        no_cmd = DiNodeCommand()
        no_cmd.compute_node_ordoring(self)
        self.nodes_order = no_cmd.nodes_order
        self.parents = no_cmd.parents
        self.nodes_degMNeighMax = no_cmd.nodes_degMNeighMax

    def _feasibility(
        self, G1_node_ind, G1_node, G2_node_ind, G2_node, G1_node_info, G2_node_info
    ):
        #
        c_e_in = self.state.G2_sub_state.c_e_in[G2_node_ind]
        c_e_out = self.state.G2_sub_state.c_e_out[G2_node_ind]
        if self.test == "iso":
            if (
                G1_node_info.c_e_in != c_e_in
                or G1_node_info.nc_e_in != self.G2_degrees_out[G2_node_ind] - c_e_in
                or G1_node_info.c_e_out != c_e_out
                or G1_node_info.nc_e_out != self.G2_degrees_in[G2_node_ind] - c_e_out
            ):
                return False
        elif self.test == "sub-iso" or self.test == "sub-iso2":
            if (
                G1_node_info.c_e_in != c_e_in
                or G1_node_info.nc_e_in > self.G2_degrees_out[G2_node_ind] - c_e_in
                or G1_node_info.c_e_out != c_e_out
                or G1_node_info.nc_e_out > self.G2_degrees_in[G2_node_ind] - c_e_out
            ):
                return False
        else:
            if (
                G1_node_info.c_e_in > c_e_in
                or G1_node_info.nc_e_in > self.G2_degrees_out[G2_node_ind] - c_e_in
                or G1_node_info.c_e_out > c_e_out
                or G1_node_info.nc_e_out > self.G2_degrees_in[G2_node_ind] - c_e_out
            ):
                return False
        #
        return super()._feasibility(
            G1_node_ind, G1_node, G2_node_ind, G2_node, G1_node_info, G2_node_info
        )

    def reset_node_view(self):
        if self.G1_o != None:
            self.G1 = None
            self.G1 = self.G1_o
            self.G1_o = None
            #
            self.G2 = None
            self.G2 = self.G2_o
            self.G2_o = None

    def is_isomorphic(self):
        result = super().is_isomorphic()
        self.reset_node_view()
        return result

    def subgraph_is_isomorphic(self):
        result = super().subgraph_is_isomorphic()
        self.reset_node_view()
        return result

    def subgraph_is_monomorphic(self):
        result = super().subgraph_is_monomorphic()
        self.reset_node_view()
        return result

    def isomorphisms_iter(self):
        yield from super().isomorphisms_iter()
        self.reset_node_view()

    def subgraph_isomorphisms_iter(self):
        yield from super().subgraph_isomorphisms_iter()
        self.reset_node_view()

    def subgraph_monomorphisms_iter(self):
        yield from super().subgraph_monomorphisms_iter()
        self.reset_node_view()


class DiGMState(GMState):
    """_summary_"""

    def __init__(self, graph_matcher):
        self.graph_matcher = graph_matcher
        #
        self.G1_sub_state = DiGMSubState(
            self.graph_matcher.G1,
            self.graph_matcher.G1_o,
            self.graph_matcher.G1_nodes_ind,
        )
        self.G2_sub_state = DiGMSubState(
            self.graph_matcher.G2,
            self.graph_matcher.G2_o,
            self.graph_matcher.G2_nodes_ind,
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
                self.G1_nodes_info[ind] = DiGMNodeInfo(size_tmp)
                self.G2_nodes_info[ind] = DiGMNodeInfo(size_tmp)
        else:
            for ind in range(G1_size):
                self.G1_nodes_info[ind] = DiGMNodeInfo(0)
                self.G2_nodes_info[ind] = DiGMNodeInfo(0)

    def compute_G1_node_info(self, node_info, G1_node_ind, G1_node):
        #
        super().compute_G1_node_info(node_info, G1_node_ind, G1_node)
        # in
        node_info.c_e_in = self.G1_sub_state.c_e_in[G1_node_ind]
        node_info.nc_e_in = (
            self.graph_matcher.G1_degrees_out[G1_node_ind] - node_info.c_e_in
        )
        # out
        node_info.c_e_out = self.G1_sub_state.c_e_out[G1_node_ind]
        node_info.nc_e_out = (
            self.graph_matcher.G1_degrees_in[G1_node_ind] - node_info.c_e_out
        )

    def compute_G2_node_info_and_syntactic_feasibility(
        self, G1_node_ind, G1_node, G2_node_ind, G2_node, node_info
    ):
        return super().compute_G2_node_info_and_syntactic_feasibility(
            G1_node_ind, G1_node, G2_node_ind, G2_node, node_info
        )

    def check_edge(
        self,
        G1_node,
        G2_node,
        G1_node_neighbors,
        G1_neighbor,
        G2_node_neighbors,
        G2_neighbor,
    ):
        n_G1_node_G1_neighbor = self.graph_matcher.G1_o.number_of_edges(
            G1_node, G1_neighbor
        )
        n_G1_neighbor_G1_node = self.graph_matcher.G1_o.number_of_edges(
            G1_neighbor, G1_node
        )
        n_G2_node_G2_neighbor = self.graph_matcher.G2_o.number_of_edges(
            G2_node, G2_neighbor
        )
        n_G2_neighbor_G2_node = self.graph_matcher.G2_o.number_of_edges(
            G2_neighbor, G2_node
        )
        if self.graph_matcher.test != "mono":
            if (
                n_G1_node_G1_neighbor != n_G2_node_G2_neighbor
                or n_G1_neighbor_G1_node != n_G2_neighbor_G2_node
            ):
                return False

        else:
            if (
                n_G1_node_G1_neighbor > n_G2_node_G2_neighbor
                or n_G1_neighbor_G1_node > n_G2_neighbor_G2_node
            ):
                return False

        if n_G1_node_G1_neighbor != 0 and not super().check_edge(
            G1_node,
            G2_node,
            self.graph_matcher.G1_o.succ[G1_node],
            G1_neighbor,
            self.graph_matcher.G2_o.succ[G2_node],
            G2_neighbor,
        ):
            return False

        if n_G1_neighbor_G1_node != 0 and not super().check_edge(
            G1_node,
            G2_node,
            self.graph_matcher.G1_o.pred[G1_node],
            G1_neighbor,
            self.graph_matcher.G2_o.pred[G2_node],
            G2_neighbor,
        ):
            return False

        return True


class DiGMSubState(GMSubState):
    """_summary_"""

    def __init__(self, G, G_o, G_nodes_ind):
        super().__init__(G, G_nodes_ind)
        self.G_o = G_o
        # Number of in edges in the mapping
        self.c_e_in = [0] * self.G.number_of_nodes()
        # Number of out edges in the mapping
        self.c_e_out = [0] * self.G.number_of_nodes()

    # upadate Feasibility Sets
    def add_node(self, node_id, node):
        neighbor_ind = 0
        for neighbor in self.G[node]:
            neighbor_ind = self.G_nodes_ind[neighbor]
            self.c[neighbor_ind] += 1
            self.c_e_out[neighbor_ind] += self.G_o.number_of_edges(node, neighbor)
            self.c_e_in[neighbor_ind] += self.G_o.number_of_edges(neighbor, node)
            self.c_sum[neighbor_ind] += node_id

    def add_node_(self, node_id, node, t_node_id):
        neighbor_ind = 0
        for neighbor in self.G[node]:
            neighbor_ind = self.G_nodes_ind[neighbor]
            self.c[neighbor_ind] += 1
            self.c_e_out[neighbor_ind] += self.G_o.number_of_edges(node, neighbor)
            self.c_e_in[neighbor_ind] += self.G_o.number_of_edges(neighbor, node)
            self.c_sum[neighbor_ind] += t_node_id

    def remove_node(self, node_id, node):
        neighbor_ind = 0
        for neighbor in self.G[node]:
            neighbor_ind = self.G_nodes_ind[neighbor]
            self.c[neighbor_ind] -= 1
            self.c_e_out[neighbor_ind] -= self.G_o.number_of_edges(node, neighbor)
            self.c_e_in[neighbor_ind] -= self.G_o.number_of_edges(neighbor, node)
            self.c_sum[neighbor_ind] -= node_id

    def remove_node_(self, node_id, node, t_node_id):
        neighbor_ind = 0
        for neighbor in self.G[node]:
            neighbor_ind = self.G_nodes_ind[neighbor]
            self.c[neighbor_ind] -= 1
            self.c_e_out[neighbor_ind] -= self.G_o.number_of_edges(node, neighbor)
            self.c_e_in[neighbor_ind] -= self.G_o.number_of_edges(neighbor, node)
            self.c_sum[neighbor_ind] -= t_node_id


class DiGMNodeInfo(GMNodeInfo):
    """_summary_"""

    def __init__(self, size):
        super().__init__(size)
        # Number of in edges in the mapping
        self.c_e_in = 0
        # Number of out edges in the mapping
        self.c_e_out = 0
        # Number of in edges not in the mapping
        self.nc_e_in = 0
        # Number of out edges not in the mapping
        self.nc_e_out = 0

    def clear(self):
        super().clear()
        self.c_e_in = 0
        self.c_e_out = 0
        self.nc_e_in = 0
        self.nc_e_out = 0


"""""" """""" """""" """""" """""" """""" """""" """""" """""" """""" """""" """""" """""" """""" """""" """""" """""" """""" """""" """""" """""" """

                                                        multi-di-graph

""" """""" """""" """""" """""" """""" """""" """""" """""" """""" """""" """""" """""" """""" """""" """""" """""" """""" """""" """""" """""" """"""


class MultiDiGraphMatcher(DiGraphMatcher):
    def __init__(self, G1, G2, node_label=None, node_match=None, edge_match=None):
        super().__init__(G1, G2, node_label, node_match, edge_match)
