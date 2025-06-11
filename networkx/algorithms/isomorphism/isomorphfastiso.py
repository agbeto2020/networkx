import sys
import time

__all__ = ["GraphMatcher", "DiGraphMatcher", "MultiGraphMatcher", "MultiDiGraphMatcher"]

"""_summary_
definitions : 
- Number of neighbors: This is the number of distinct nodes to which a node is connected by one or more edges.
- Number of edges: This is the total number of edges connected to a node, including multiple edges.

let G is a simple graph, and n, node in G :
Number_of_neighbors(n) equal to Number_of_edges(n)
if G is a multi-graph Number_of_neighbors(n) not necessacary equal to Number_of_edges(n)
Number_of_neighbors(n) <= Number_of_edges(n)
"""

"""
*****************
FASTiso Algorithm
*****************

An implementation of the FASTiso algorithm [1]_ for Graph Isomorphism testing.

Introduction
------------

The classes GraphMatcher, MultiGraphMatcher, DiGraphMatcher, and MultiDiGraphMatcher 
handle the mapping between undirected graphs, multigraphs, directed graphs, 
and directed multigraphs, respectively. It is possible to test graph isomorphism, 
non-induced graph isomorphism, and monomorphism (non-induced graph isomorphism).  

Some heuristics for computing the order of variables are not effective on path graphs.
Therefore, when working with this type of graph, it is recommended to set 
the parameter path_graph=True to avoid using these heuristics.  

To handle attributes on nodes and edges, the parameters node_match and edge_match 
must be defined.


References
----------
"""

"""""" """""" """""" """""" """""" """""" """""" """""" """""" """""" """""" """""" """""" """""" """""" """""" """""" """""" """""" """""" """""" """

                                                         graph

""" """""" """""" """""" """""" """""" """""" """""" """""" """""" """""" """""" """""" """""" """""" """""" """""" """""" """""" """""" """""" """"""


class NodeOrderingProp:
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
        self.degMNeigh = 0  # not use for path graph
        self.degMNeighMax = 0
        self.degMo = degMo_  # not use for path graph
        self.present = 0
        self.candidate = 0


class NodeCommand:
    """
    Computes the node odering using attributes of the NodeOrderingProp class.
    """

    def __init__(
        self,
    ):
        self.limit = 4

    #
    def update_degMNeigh_and_degMo(self, no):
        if self.graph_matcher.path_graph == True:
            return
        node = no.node
        for neighbor in self.graph_matcher.G1.neighbors(node):
            neighbor_ind = self.graph_matcher.G1_nodes_ind[neighbor]
            no_neigh = self.nodes[neighbor_ind]
            if no_neigh.present == 0:
                if no.degM == 0:
                    no_neigh.degMo -= 1
                no_neigh.degMNeigh += 1

    #
    def update_no(self, no, level):
        """Marks a node as present and updates its neighbors' properties.

        Parameters:
            no: NodeOrderingProp corresponding to the node to be updated.
        """
        no.present = 1
        no.candidate = 1
        degM = no.degM if no.degM < self.limit else self.limit
        no.degM = 0
        node = no.node
        #
        for neighbor in self.graph_matcher.G1.neighbors(node):
            neighbor_ind = self.graph_matcher.G1_nodes_ind[neighbor]
            no_neigh = self.nodes[neighbor_ind]
            #
            if no.degMNeighMax < no_neigh.degM:
                no.degMNeighMax = no_neigh.degM
            #
            if no_neigh.present == 0:
                if self.nodes_limit[no_neigh.id] < self.limit:
                    self.update_degMNeigh_and_degMo(no_neigh)
                    self.nodes_limit[no_neigh.id] += 1
                #
                no_neigh.degM += 1
                no_neigh.degNeigh += len(self.graph_matcher.G1[node][neighbor])
                no_neigh.degMNeigh -= degM
                if level == 0:
                    no_neigh.degMo -= 1

                # no need DegNeigh for undirect graph
            if no_neigh.candidate == 0:
                no_neigh.candidate = 1
                self.candidates.append(no_neigh)
                self.parents[no_neigh.id] = no.id

        # We remove the selected node from the list of candidates.
        self.candidates = [no for no in self.candidates if no.present == 0]

    #
    def compute_variable_ordering(self, graph_matcher):
        """Computes the node ordering.

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
        self.nodes_limit = [0] * size
        #
        path_graph = self.graph_matcher.path_graph
        # Create NodeOrderingProp class for each node.
        for ind, node in enumerate(self.graph_matcher.G1_nodes):
            no = NodeOrderingProp(
                ind,
                node,
                self.graph_matcher.node_prob[ind],
                self.graph_matcher.G1_degrees[ind],
                len(self.graph_matcher.G1[node]) if path_graph == False else 0,
            )
            self.nodes[ind] = no
        ##Select first node
        selected_no = None
        n = 0
        # Selection of the first
        if self.graph_matcher.labelled:
            selected_no = min(self.nodes, key=lambda obj: obj.prob)
        else:
            selected_no = min(self.nodes, key=lambda obj: -obj.deg)
        #
        self.update_no(selected_no, n)
        self.nodes_order[n] = selected_no.id
        self.nodes_degMNeighMax[n] = selected_no.degMNeighMax
        self.parents[selected_no.id] = None
        n += 1

        while n < size:
            if len(self.candidates) != 0:
                selected_no = min(
                    self.candidates,
                    key=lambda k: (
                        -k.degM,
                        -k.degNeigh,  # Not used for undirect graph
                        -k.degMNeigh,
                        -k.degMo,
                        k.prob,
                        -k.deg,
                    ),
                )
            # Node which don't have neighbor
            else:
                selected_no = min(self.nodes, key=lambda k: k.present)
            #
            self.update_no(selected_no, n)
            self.nodes_order[n] = selected_no.id
            self.nodes_degMNeighMax[n] = selected_no.degMNeighMax
            n += 1


class GraphMatcher:
    """fastiso implementation for matching undirect graphs"""

    def __init__(
        self,
        G1,
        G2,
        node_label=None,
        node_match=None,
        edge_match=None,
        path_graph=False,
    ):
        """Initializes GraphMatcher.

        Parameters
        ----------
        G1,G2: NetworkX Graph.
           The two graphs to check for isomorphism or monomorphism.

        node_label: The name of the node attribute to be used when comparing nodes.

        node_natch: callable.
           A function that returns True if the node attribute dictionary for
           the pair of nodes (u, v) respects the isomorphism conditions.
           The function will be called like: node_natch(G1.nodes[u],G2.nodes[v]).

        edge_natch: callable.
            A function that returns True if the edge attribute dictionary for
            the pairs of nodes (u1, v1) in G1 and (u2, v2) in G2 respects the isomorphism conditions.
            The function will be called like: edge_match(G1[u1][v1], G2[u2][v2]).

        path_graph: bool
            to indicate that G1 and G2 are path graphs.

        Examples
        --------
        To create a GraphMatcher which checks for isomorphism:

        >>> from networkx.algorithms.isomorphism.isomorphfastiso import GraphMatcher
        >>> G1 = nx.path_graph(4)
        >>> G2 = nx.path_graph(4)
        >>> GraphMatcher(G1, G2)
        """
        self.node_match = node_match
        self.edge_match = edge_match
        #
        self.node_label = node_label
        # Indicates whether the graph is labeled or not.
        self.labelled = False
        if self.node_label != None:
            self.labelled == True
        # Indicates whether is path graph or not.
        self.path_graph = path_graph
        self.G1 = G2
        self.G2 = G1
        self.G1_nodes = list(self.G1.nodes())
        self.G1_nodes_ind = {node: ind for ind, node in enumerate(self.G1_nodes)}
        self.G2_nodes = list(self.G2.nodes())
        self.G2_nodes_ind = {node: ind for ind, node in enumerate(self.G2_nodes)}
        #
        self.G1_degrees = [self.G1.degree[node] for node in self.G1_nodes]
        self.G2_degrees = [self.G2.degree[node] for node in self.G2_nodes]
        # for manage loops
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

    def subgraph_is_isomorphic(self):
        """Returns True if a subgraph of G1 is isomorphic to G2."""
        try:
            x = next(self.subgraph_isomorphisms_iter())
            return True
        except StopIteration:
            return False

    def subgraph_is_isomorphic_M(self):
        """Returns True if a subgraph of G1 is isomorphic to G2."""
        try:
            x = next(self.subgraph_isomorphisms_iter_M())
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

    ##
    def isomorphisms_iter(self):
        """Generator over isomorphisms between G1 and G2."""
        # Declare that we are looking for a graph-graph isomorphism.
        self.test = "iso"
        if self.initialize():
            yield from self.match(0)

    def subgraph_isomorphisms_iter(self):
        """Generator over isomorphisms between a subgraph of G1 and G2."""
        # Declare that we are looking for graph-subgraph isomorphism.
        self.test = "sub-iso"
        if self.initialize():
            yield from self.match(0)

    def subgraph_isomorphisms_iter_M(self):
        """Generator over isomorphisms between a subgraph of G1 and G2."""
        # Declare that we are looking for graph-subgraph isomorphism.
        self.test = "sub-isoM"
        if self.initialize():
            yield from self.match(0)

    def subgraph_monomorphisms_iter(self):
        """Generator over monomorphisms between a subgraph of G1 and G2."""
        # Declare that we are looking for graph-subgraph monomorphism.
        self.test = "mono"
        if self.initialize():
            yield from self.match(0)

    def match(self, k=0):
        """Extends isomorphism mapping.

        This function is called recursively to determine
        if there is a matching between G1 and G2.

        Parameters
            k (int): the level of the search tree
        """
        if k == self.G1.number_of_nodes():
            self.mapping = self.state.copy_mapping()
            yield self.mapping
        else:
            # The k-th node in the variable ordering.
            G1_node_ind = self.nodes_order[k]
            G1_node = self.G1_nodes[G1_node_ind]
            # The candidate nodes that will be mapped to G1_node
            # are chosen from among the neighbors
            # of the node corresponding to the parent of G1_node
            # if G1_node does not have a parent, the set of candidates
            # is the set of all nodes in the target graph.
            domain = None
            parent_id = self.parents[G1_node_ind]
            if parent_id == None:
                domain = self.G2_nodes
            else:
                parent_id = self.state.G1_sub_state.m[parent_id]
                parent_node = self.G2_nodes[parent_id]
                domain = self.G2[parent_node]
            # Get the nodeInfo object for the current node pair
            G1_node_info = self.state.G1_nodes_info[k]
            G2_node_info = self.state.G2_nodes_info[k]
            # The information on the feasibility sets is calculated
            # only 1 time for each node of the pattern graph.
            if not G1_node_info.also_do:
                self.state.compute_G1_node_info(G1_node_info, G1_node_ind, G1_node)
                self.state.G1_sub_state.add_G1_node(G1_node_ind, G1_node)
            #
            G2_node_ind = 0
            for G2_node in domain:
                G2_node_ind = self.G2_nodes_ind[G2_node]
                if self.state.G2_sub_state.m[G2_node_ind] == None and self._feasibility(
                    G1_node_ind,
                    G1_node,
                    G2_node_ind,
                    G2_node,
                    G1_node_info,
                    G2_node_info,
                ):
                    self.state.add_node_pair(G1_node_ind, G1_node, G2_node_ind, G2_node)
                    yield from self.match(k + 1)
                    self.state.remove_node_pair(
                        G1_node_ind, G1_node, G2_node_ind, G2_node
                    )

                if G2_node_info.also_do:
                    G2_node_info.clear()

    def initialize_sate(self):
        """(Re)Initializes the state of GraphMatcher."""
        self.state = GMState(self)

    def initialize(self):
        """ "Initializes (Reinitializes) the state of the algorithm

        This function also does some fast checks on the existence
        of isomorphism before initializing the state of the algorithm.
        """
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
        ) = self.compute_nodes_prop(
            self.G1, self.G1_nodes, self.G1_nodes_ind, self.G1_degrees
        )
        (
            self.G2_sum_neighbors_degree,
            self.G2_max_neighbors_degree,
        ) = self.compute_nodes_prop(
            self.G2, self.G2_nodes, self.G2_nodes_ind, self.G2_degrees
        )
        #
        # node probability
        self.node_prob = [0] * G1_size
        self.parents = None
        self.nodes_order = None
        self.nodes_degMNeighMax = None
        #
        # compute node_probability
        if not self.compute_node_probability():
            return False
        # compute node ordoring
        self.compute_variable_ordering()
        #
        self.node_prob = None

        # Initialize state
        self.initialize_sate()
        # Reset degMNeighMax
        self.nodes_degMNeighMax = None
        return True

    def compute_nodes_prop(self, G, G_nodes, G_nodes_ind, G_degrees):
        """calculates some invariant on the nodes which will be used
        to prune the search tree.
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
            if len(neighbor_degrees) != 0:
                sum_neighbors_degree[ind] = sum(neighbor_degrees)
                max_neighbors_degree[ind] = max(neighbor_degrees)
            else:
                sum_neighbors_degree[ind] = 0
                max_neighbors_degree[ind] = 0

        return sum_neighbors_degree, max_neighbors_degree

    def compute_node_probability(self):
        G1_size = self.G1.number_of_nodes()
        G2_size = self.G2.number_of_nodes()
        #
        max_degree = max(self.G2_degrees) + 1
        degree_counter = [0] * max_degree
        sum_degree_counter = {}
        label_counter = {}
        #
        degree = 0
        sum_degree = 0
        node_label = None
        if self.node_label != None:
            for ind, node in enumerate(self.G2_nodes):
                degree = self.G2_degrees[ind]
                degree_counter[degree] += 1
                # will be use just for isomorphism
                sum_degree = self.G2_sum_neighbors_degree[ind]
                if sum_degree in sum_degree_counter:
                    sum_degree_counter[sum_degree] += 1
                else:
                    sum_degree_counter[sum_degree] = 1
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
                # will be use just for isomorphism
                sum_degree = self.G2_sum_neighbors_degree[ind]
                if sum_degree in sum_degree_counter:
                    sum_degree_counter[sum_degree] += 1
                else:
                    sum_degree_counter[sum_degree] = 1
        #
        degree_counter = [x / G2_size for x in degree_counter]
        # calcul prob
        prob = 0
        if self.test == "iso":
            for ind, node in enumerate(self.G1_nodes):
                degree = self.G1_degrees[ind]
                sum_degree = self.G1_sum_neighbors_degree[ind]
                #
                if sum_degree not in sum_degree_counter:
                    return False
                #
                if self.node_label == None:
                    prob = degree_counter[degree] * sum_degree_counter[sum_degree]
                    # prob = degree_counter[degree]
                else:
                    node_label = self.G1.nodes[node][self.node_label]
                    prob == degree_counter[degree] * sum_degree_counter[
                        sum_degree
                    ] * label_counter[node_label]
                    # prob == degree_counter[degree] * label_counter[node_label]
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
    def compute_variable_ordering(self):
        """Calls the NodeCommand class to compute the variable ordering."""
        no_cmd = NodeCommand()
        no_cmd.compute_variable_ordering(self)
        self.nodes_order = no_cmd.nodes_order
        self.parents = no_cmd.parents
        # Maximum of the degM attribute (NodeOrderingProp) of the neighbors
        # for each node of pattern graph.
        self.nodes_degMNeighMax = no_cmd.nodes_degMNeighMax

    def _feasibility(
        self, G1_node_ind, G1_node, G2_node_ind, G2_node, G1_node_info, G2_node_info
    ):
        """check if the addition of the pair (G1_node, G2_node)
        respects the feasibility rules.

        this function first checks isomorphism conditions then
        uses look-ahead functions to detect dead states.
        """
        c = 0
        # nc=0
        if self.test == "iso":
            # Begin isomorphism conditions
            c = self.state.G2_sub_state.c[G2_node_ind]
            if (
                G1_node_info.c != c
                or G1_node_info.nc != len(self.G2[G2_node]) - c
                or G1_node_info.c_sum != self.state.G2_sub_state.c_sum[G2_node_ind]
                # prop
                or self.G1_max_neighbors_degree[G1_node_ind]
                != self.G2_max_neighbors_degree[G2_node_ind]
                or self.G1_sum_neighbors_degree[G1_node_ind]
                != self.G2_sum_neighbors_degree[G2_node_ind]
                or self.G1_self_edges[G1_node_ind] != self.G2_self_edges[G2_node_ind]
            ):
                return False

            if not self.compare_node_attr(
                self.G1.nodes[G1_node], self.G2.nodes[G2_node]
            ):
                return False
            # End isomorphism conditions

            # Calculates the information on G2_node that
            # we need for the look-ahead functions
            # and also checks the isomorphism conditions for the edges.
            if not self.state.compute_G2_node_info_and_verify_edge_feasibility(
                G1_node_ind, G1_node, G2_node_ind, G2_node, G2_node_info
            ):
                return False
            # Begin look-ahead
            if (
                G1_node_info.num_c != G2_node_info.num_c
                or G1_node_info.num_nc != G2_node_info.num_nc
                or G1_node_info.c_sum_neigh != G2_node_info.c_sum_neigh
            ):
                return False
            # End look-ahead

        elif self.test == "sub-iso":
            # Begin subgraph isomorphism conditions
            c = self.state.G2_sub_state.c[G2_node_ind]
            if (
                G1_node_info.c != c
                or G1_node_info.nc > len(self.G2[G2_node]) - c
                or G1_node_info.c_sum != self.state.G2_sub_state.c_sum[G2_node_ind]
                # prop
                or self.G1_max_neighbors_degree[G1_node_ind]
                > self.G2_max_neighbors_degree[G2_node_ind]
                or self.G1_sum_neighbors_degree[G1_node_ind]
                > self.G2_sum_neighbors_degree[G2_node_ind]
                or self.G1_self_edges[G1_node_ind] != self.G2_self_edges[G2_node_ind]
            ):
                return False

            if not self.compare_node_attr(
                self.G1.nodes[G1_node], self.G2.nodes[G2_node]
            ):
                return False
            # End subgraph isomorphism conditions

            # Calculates the information on G2_node that
            # we need for the look-ahead functions
            # and also checks the subgraph isomorphism conditions for the edges.
            if not self.state.compute_G2_node_info_and_verify_edge_feasibility(
                G1_node_ind, G1_node, G2_node_ind, G2_node, G2_node_info
            ):
                return False
            # Begin look-ahead
            if G1_node_info.num_c > G2_node_info.num_c:
                return False
            #
            for ind in range(len(G1_node_info.DegMNeigh)):
                if G1_node_info.DegMNeigh[ind] > G2_node_info.DegMNeigh[ind]:
                    return False
            # End look-ahead

        elif self.test == "sub-isoM":
            # Begin subgraph isomorphism conditions
            c = self.state.G2_sub_state.c[G2_node_ind]
            if (
                G1_node_info.c != c
                or G1_node_info.nc > len(self.G2[G2_node]) - c
                or G1_node_info.c_sum != self.state.G2_sub_state.c_sum[G2_node_ind]
                # prop
                or self.G1_max_neighbors_degree[G1_node_ind]
                > self.G2_max_neighbors_degree[G2_node_ind]
                or self.G1_sum_neighbors_degree[G1_node_ind]
                > self.G2_sum_neighbors_degree[G2_node_ind]
                or self.G1_self_edges[G1_node_ind] != self.G2_self_edges[G2_node_ind]
            ):
                return False

            if not self.compare_node_attr(
                self.G1.nodes[G1_node], self.G2.nodes[G2_node]
            ):
                return False
            # End subgraph isomorphism conditions

            # Calculates the information on G2_node that
            # we need for the look-ahead functions
            # and also checks the subgraph isomorphism conditions for the edges.
            if not self.state.compute_G2_node_info_and_verify_edge_feasibility(
                G1_node_ind, G1_node, G2_node_ind, G2_node, G2_node_info
            ):
                return False
            ##Begin look-ahead
            if (
                G1_node_info.num_c > G2_node_info.num_c
                or G1_node_info.num_nc > G2_node_info.num_nc
                or G1_node_info.DegMNeigh_sum > G2_node_info.DegMNeigh_sum
                or G1_node_info.DegMNeigh_max > G2_node_info.DegMNeigh_max
            ):
                return False
            # End look-ahead

        else:
            # Begin monomorphism conditions
            c = self.state.G2_sub_state.c[G2_node_ind]
            if (
                G1_node_info.c > c
                or G1_node_info.nc > len(self.G2[G2_node]) - c
                # prop
                or self.G1_max_neighbors_degree[G1_node_ind]
                > self.G2_max_neighbors_degree[G2_node_ind]
                or self.G1_sum_neighbors_degree[G1_node_ind]
                > self.G2_sum_neighbors_degree[G2_node_ind]
                or self.G1_self_edges[G1_node_ind] > self.G2_self_edges[G2_node_ind]
            ):
                return False

            if not self.compare_node_attr(
                self.G1.nodes[G1_node], self.G2.nodes[G2_node]
            ):
                return False
            # End monomorphism conditions

            # Calculates the information on G2_node that
            # we need for the look-ahead functions
            # and also checks the monomorphism conditions for the edges.
            if not self.state.compute_G2_node_info_and_verify_edge_feasibility(
                G1_node_ind, G1_node, G2_node_ind, G2_node, G2_node_info
            ):
                return False
            ##Begin look-ahead
            if (
                G1_node_info.num_c > G2_node_info.num_c
                or G1_node_info.DegMNeigh_sum > G2_node_info.DegMNeigh_sum
                or G1_node_info.DegMNeigh_max > G2_node_info.DegMNeigh_max
            ):
                return False
            # End look-ahead

        # valid
        return True

    def compare_edge_attr(self, G1_node_edges_attrs, G2_node_edges_attrs):
        """Calls edge_match to check if the Edge attribute dictionary for
        the pairs of nodes (u1, v1) in G1 and (u2, v2) in G2 respects
        the isomorphism conditions.

        Return True if this is the case, and False otherwise.
        """
        if self.edge_match != None:
            return self.edge_match(G2_node_edges_attrs, G1_node_edges_attrs)
        else:
            return True

    def compare_node_attr(self, G1_node_attrs, G2_node_attrs):
        """Calls node_match to check if the node attribute dictionary for
        the pair of nodes (u, v) respects the isomorphism conditions.

        Return True if this is the case, and False otherwise.
        """
        if self.node_match != None:
            return self.node_match(G2_node_attrs, G1_node_attrs)
        else:
            return True


class GMState:
    """Internal representation of state for the GraphMatcher class

    This function will make it possible in particular to manage
    the state corresponding to the current mapping by adding
    pairs of nodes (u,v).
    It also makes it possible to calculate for (u,v) the information
    for verifying the feasibility rules.

    for performance reasons, when calculating feasibility rules information for v,
    we check isomorphism constraints regarding edges.

    u in G1, v in G2
    """

    def __init__(self, graph_matcher):
        """Initializes GMState object."""
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
        if self.graph_matcher.test == "sub-iso":
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
        """Adds the pair of nodes (G1_node,G2_node) to the current mapping"""
        self.G1_sub_state.m[G1_node_ind] = G2_node_ind
        self.G2_sub_state.m[G2_node_ind] = G1_node_ind
        #
        self.G2_sub_state.add_G2_node(G2_node_ind, G2_node, G1_node_ind)

    def remove_node_pair(self, G1_node_ind, G1_node, G2_node_ind, G2_node):
        """Removes the pair of nodes (G1_node,G2_node) to the current mapping"""
        self.G2_sub_state.remove_G2_node(G2_node_ind, G2_node, G1_node_ind)
        #
        self.G1_sub_state.m[G1_node_ind] = None
        self.G2_sub_state.m[G2_node_ind] = None

    def compute_G1_node_info(self, node_info, G1_node_ind, G1_node):
        """Computes feasibility rules information for the node "G1_node" in G1"""
        node_info.also_do = True
        # Number of neighbors in the mapping
        node_info.c = self.G1_sub_state.c[G1_node_ind]
        # Number of neighbors not in the mapping
        node_info.nc = len(self.graph_matcher.G1[G1_node]) - node_info.c
        #
        #
        if self.graph_matcher.test != "mono":
            # Sum of the IDs (numerical) of the neighbors in the mapping
            node_info.c_sum = self.G1_sub_state.c_sum[G1_node_ind]
        #
        neighbor_ind = 0
        neighbor_c = 0
        #
        if self.graph_matcher.test == "iso":
            for neighbor in self.G1_sub_state.G[G1_node]:
                neighbor_ind = self.G1_sub_state.G_nodes_ind[neighbor]
                # if neighbor not in the mapping
                if self.G1_sub_state.m[neighbor_ind] == None:
                    neighbor_c = self.G1_sub_state.c[neighbor_ind]
                    #
                    if neighbor_c == 0:
                        # Number of neighbors that do not have neighbors in the mapping
                        node_info.num_nc += 1
                    #
                    node_info.c_sum_neigh += self.G1_sub_state.c_sum[neighbor_ind]

            ##Number of neighbors having neighbors in the mapping
            node_info.num_c = node_info.nc - node_info.num_nc

        elif self.graph_matcher.test == "sub-iso":
            for neighbor in self.G1_sub_state.G[G1_node]:
                neighbor_ind = self.G1_sub_state.G_nodes_ind[neighbor]
                # if neighbor not in the mapping
                if self.G1_sub_state.m[neighbor_ind] == None:
                    # Node classification
                    node_info.DegMNeigh[self.G1_sub_state.c[neighbor_ind]] += 1
            ##Number of neighbors having neighbors in the mapping
            node_info.num_c = node_info.nc - node_info.DegMNeigh[0]

        elif self.graph_matcher.test == "sub-isoM":
            for neighbor in self.G1_sub_state.G[G1_node]:
                neighbor_ind = self.G1_sub_state.G_nodes_ind[neighbor]
                # if neighbor not in the mapping
                if self.G1_sub_state.m[neighbor_ind] == None:
                    neighbor_c = self.G1_sub_state.c[neighbor_ind]
                    #
                    if neighbor_c == 0:
                        # Number of neighbors that do not have neighbors in the mapping
                        node_info.num_nc += 1
                    #
                    node_info.DegMNeigh_sum += neighbor_c
                    #
                    if neighbor_c > node_info.DegMNeigh_max:
                        node_info.DegMNeigh_max = neighbor_c

            ##Number of neighbors having neighbors in the mapping
            node_info.num_c = node_info.nc - node_info.num_nc

        else:
            for neighbor in self.G1_sub_state.G[G1_node]:
                neighbor_ind = self.G1_sub_state.G_nodes_ind[neighbor]
                # if neighbor not in the mapping
                if self.G1_sub_state.m[neighbor_ind] == None:
                    neighbor_c = self.G1_sub_state.c[neighbor_ind]
                    #
                    if neighbor_c == 0:
                        # Number of neighbors that do not have neighbors in the mapping
                        node_info.num_nc += 1

                    if neighbor_c > node_info.DegMNeigh_max:
                        node_info.DegMNeigh_max = neighbor_c
            # Number of neighbors having neighbors in the mapping
            node_info.num_c = node_info.nc - node_info.num_nc

    def compute_G2_node_info_and_verify_edge_feasibility(
        self, G1_node_ind, G1_node, G2_node_ind, G2_node, node_info
    ):
        """Computes feasibility rules information for the node "G2_node" in G2
        and checks isomorphism constraints regarding edges

        Returns True if edges are valid and False otherwise.
        """
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
        if self.graph_matcher.test == "iso":
            for neighbor in G2_node_neighbors:
                neighbor_ind = self.G2_sub_state.G_nodes_ind[neighbor]
                neighbor_corr_ind = self.G2_sub_state.m[neighbor_ind]
                # if neighbor not in the mapping
                if neighbor_corr_ind == None:
                    neighbor_c = self.G2_sub_state.c[neighbor_ind]
                    #
                    if neighbor_c == 0:
                        # Number of neighbors that do not have neighbors in the mapping
                        node_info.num_nc += 1
                    #
                    node_info.c_sum_neigh += self.G2_sub_state.c_sum[neighbor_ind]
                # neighbor in the mapping
                else:
                    neighbor_corr = self.graph_matcher.G1_nodes[neighbor_corr_ind]
                    if (
                        # neighbor_corr not in self.graph_matcher.G1[G1_node]
                        # or
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

            # Number of neighbors having neighbors in the mapping
            node_info.num_c = (
                len(self.graph_matcher.G2[G2_node])
                - self.G2_sub_state.c[G2_node_ind]
                - node_info.num_nc
            )

        elif self.graph_matcher.test == "sub-iso":
            for neighbor in G2_node_neighbors:
                neighbor_ind = self.G2_sub_state.G_nodes_ind[neighbor]
                neighbor_corr_ind = self.G2_sub_state.m[neighbor_ind]
                # if neighbor not in the mapping
                if neighbor_corr_ind == None:
                    neighbor_c = self.G2_sub_state.c[neighbor_ind]
                    # Node classification
                    if neighbor_c < len(node_info.DegMNeigh):
                        node_info.DegMNeigh[neighbor_c] += 1

                # neighbor in the mapping
                else:
                    neighbor_corr = self.graph_matcher.G1_nodes[neighbor_corr_ind]
                    if (
                        # neighbor_corr not in self.graph_matcher.G1[G1_node]
                        # or
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
            # Number of neighbors having neighbors in the mapping
            node_info.num_c = (
                len(self.graph_matcher.G2[G2_node])
                - self.G2_sub_state.c[G2_node_ind]
                - node_info.DegMNeigh[0]
            )

        elif self.graph_matcher.test == "sub-isoM":
            for neighbor in G2_node_neighbors:
                neighbor_ind = self.G2_sub_state.G_nodes_ind[neighbor]
                neighbor_corr_ind = self.G2_sub_state.m[neighbor_ind]
                # if neighbor not in the mapping
                if neighbor_corr_ind == None:
                    neighbor_c = self.G2_sub_state.c[neighbor_ind]
                    #
                    if neighbor_c == 0:
                        # Number of neighbors that do not have neighbors in the mapping
                        node_info.num_nc += 1
                    #
                    node_info.DegMNeigh_sum += neighbor_c
                    #
                    if neighbor_c > node_info.DegMNeigh_max:
                        node_info.DegMNeigh_max = neighbor_c

                # neighbor in the mapping
                else:
                    neighbor_corr = self.graph_matcher.G1_nodes[neighbor_corr_ind]
                    if (
                        # neighbor_corr not in self.graph_matcher.G1[G1_node]
                        # or
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

            # Number of neighbors having neighbors in the mapping
            node_info.num_c = (
                len(self.graph_matcher.G2[G2_node])
                - self.G2_sub_state.c[G2_node_ind]
                - node_info.num_nc
            )
        # For monomorphism
        else:
            for neighbor in G1_node_neighbors:
                neighbor_ind = self.G1_sub_state.G_nodes_ind[neighbor]
                neighbor_corr_ind = self.G1_sub_state.m[neighbor_ind]
                # neighbor in the mapping
                if neighbor_corr_ind != None:
                    neighbor_corr = self.graph_matcher.G2_nodes[neighbor_corr_ind]
                    if (
                        # neighbor_corr not in self.graph_matcher.G2[G2_node]
                        # or
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
                # if neighbor not in the mapping
                if neighbor_corr_ind == None:
                    neighbor_c = self.G2_sub_state.c[neighbor_ind]
                    # Number of neighbors that do not have neighbors in the mapping
                    if neighbor_c == 0:
                        node_info.num_nc += 1

                    if neighbor_c > node_info.DegMNeigh_max:
                        node_info.DegMNeigh_max = neighbor_c
            # Number of neighbors having neighbors in the mapping
            node_info.num_c = (
                len(self.graph_matcher.G2[G2_node])
                - self.G2_sub_state.c[G2_node_ind]
                - node_info.num_nc
            )

        # Valid
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
        """Checks if the pairs of nodes (G1_node, G1_neighbor), (G2_node,G2_neighbor)
        respect the isomorphism conditions.
        """
        if self.graph_matcher.test != "mono":
            return (
                G1_neighbor in G1_node_neighbors
                and self.graph_matcher.compare_edge_attr(
                    G1_node_neighbors[G1_neighbor], G2_node_neighbors[G2_neighbor]
                )
            )
        else:
            return (
                G2_neighbor in G2_node_neighbors
                and self.graph_matcher.compare_edge_attr(
                    G1_node_neighbors[G1_neighbor], G2_node_neighbors[G2_neighbor]
                )
            )


class GMSubState:
    """Class for managing feasibility sets for GraphMatcher."""

    def __init__(self, G, G_nodes_ind):
        """Initializes GMSubState object."""
        self.G = G
        self.G_nodes_ind = G_nodes_ind
        G_size = self.G.number_of_nodes()
        # mapping
        self.m = [None] * G_size
        # corresponding to degM in NodeOrdoring class (Number of neighbors contained in the mapping)
        self.c = [0] * G_size
        # Sum of the IDs (numerical) of the neighbors in the mapping
        self.c_sum = [0] * G_size

    # Upadate Feasibility Sets

    def add_G1_node(self, node_id, node):
        neighbor_ind = 0
        for neighbor in self.G[node]:
            neighbor_ind = self.G_nodes_ind[neighbor]
            self.c[neighbor_ind] += 1
            self.c_sum[neighbor_ind] += node_id

    def remove_G1_node(self, node_id, node):
        neighbor_ind = 0
        for neighbor in self.G[node]:
            neighbor_ind = self.G_nodes_ind[neighbor]
            self.c[neighbor_ind] -= 1
            self.c_sum[neighbor_ind] -= node_id

    def add_G2_node(self, node_id, node, t_node_id):
        neighbor_ind = 0
        for neighbor in self.G[node]:
            neighbor_ind = self.G_nodes_ind[neighbor]
            self.c[neighbor_ind] += 1
            self.c_sum[neighbor_ind] += t_node_id

    def remove_G2_node(self, node_id, node, t_node_id):
        neighbor_ind = 0
        for neighbor in self.G[node]:
            neighbor_ind = self.G_nodes_ind[neighbor]
            self.c[neighbor_ind] -= 1
            self.c_sum[neighbor_ind] -= t_node_id

    # Upadate Feasibility Sets


class GMNodeInfo:
    """
    A class to store node information about feasibility rules for GraphMatcher.

    Attributes
    ----------
    DegMNeigh (Only sub-graph isomorphism, node classification) : list of int
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

    num_nc (For isomorphism, monomorphism and sub-iso2) : int
        Corresponds to degMo in the NodeOrdoring class. It represents the number
        of neighbors that do not have neighbors in the mapping.
        DegMNeigh[0] equal to num_nc

    DegMNeigh_sum (only for monomorphism and sub-isoM) : int
        The sum of the degM attribute of the neighbors.

    DegMNeigh_max (only for monomorphism and sub-isoM) : int
        The maximum of the degM attribute of the neighbors.
    """

    def __init__(self, size):
        """
        Initializes the GMNodeInfo object.

        Parameters
        ----------
        size : int
            The size of the DegMNeigh list.
        """
        self.DegMNeigh = [0] * size  # Initialize DegMNeigh with zeros
        self.c = 0  # Number of neighbors in the mapping
        self.nc = 0  # Number of neighbors not in the mapping
        self.c_sum = 0  # Sum of IDs of neighbors in the mapping
        self.c_sum_neigh = 0  #
        self.num_c = 0  # Number of neighbors having neighbors in the mapping
        self.also_do = False  # Additional processing flag

        #
        self.num_nc = 0  # Number of neighbors that do not have neighbors in the mapping
        self.DegMNeigh_sum = 0  # Sum of the degM attribute of the neighbors
        self.DegMNeigh_max = 0  # Maximum of the degM attribute of the neighbors

    def clear(self):
        """Resets all attributes to their default values."""
        self.c = 0
        self.nc = 0
        self.num_c = 0
        self.num_nc = 0
        self.DegMNeigh_max = 0
        self.DegMNeigh_sum = 0
        self.also_do = False
        self.c_sum_neigh = 0
        #
        for ind in range(len(self.DegMNeigh)):
            self.DegMNeigh[ind] = 0


"""""" """""" """""" """""" """""" """""" """""" """""" """""" """""" """""" """""" """""" """""" """""" """""" """""" """""" """""" """""" """""" """

                                                            multi-graph

""" """""" """""" """""" """""" """""" """""" """""" """""" """""" """""" """""" """""" """""" """""" """""" """""" """""" """""" """""" """""" """"""


class MultiGraphMatcher(GraphMatcher):
    """fastiso implementation for matching undirect MultiGraphs."""

    def __init__(
        self,
        G1,
        G2,
        node_label=None,
        node_match=None,
        edge_match=None,
        path_graph=False,
    ):
        """Initializes MultiGraphMatcher.
        G1 and G2 have to be nx.MultiGraph instances.

        Examples
        --------
        >>> from networkx.algorithms.isomorphism.isomorphfastiso import MultiGraphMatcher
        >>> G1 = nx.DiGraph(nx.path_graph(4, create_using=nx.MultiGraph()))
        >>> G2 = nx.DiGraph(nx.path_graph(4, create_using=nx.MultiGraph()))
        >>> gm = MultiGraphMatcher(G1, G2, path_graph=True)
        """
        super().__init__(G1, G2, node_label, node_match, edge_match, path_graph)

    def initialize_sate(self):
        """(Re)Initializes the state of MultiGraphMatcher."""
        self.state = MultiGMState(self)

    def _feasibility(
        self, G1_node_ind, G1_node, G2_node_ind, G2_node, G1_node_info, G2_node_info
    ):
        """check if the addition of the pair (G1_node, G2_node)
        respects the feasibility rules.
        """
        #
        c_e = self.state.G2_sub_state.c_e[G2_node_ind]
        if self.test == "iso":
            if (
                # Number of edges with nodes in the mapping.
                G1_node_info.c_e != c_e
                # Number of edges with nodes that are not in the mapping.
                or G1_node_info.nc_e != self.G2_degrees[G2_node_ind] - c_e
            ):
                return False
        elif self.test == "sub-iso" or self.test == "sub-isoM":
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
    """Internal representation of state for the MultiGraphMatcher class."""

    def __init__(self, graph_matcher):
        """Initializes MultiGMState object."""
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
        if self.graph_matcher.test == "sub-iso":
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
        """Computes feasibility rules information for the node "G1_node" in G1."""
        #
        super().compute_G1_node_info(node_info, G1_node_ind, G1_node)
        # Number of edges with nodes in the mapping.
        node_info.c_e = self.G1_sub_state.c_e[G1_node_ind]
        # Number of edges with nodes that are not in the mapping.
        node_info.nc_e = self.graph_matcher.G1_degrees[G1_node_ind] - node_info.c_e

    def compute_G2_node_info_and_verify_edge_feasibility(
        self, G1_node_ind, G1_node, G2_node_ind, G2_node, node_info
    ):
        """Computes feasibility rules information for the node "G2_node" in G2."""
        return super().compute_G2_node_info_and_verify_edge_feasibility(
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
        """Checks if the pairs of nodes (G1_node, G1_neighbor), (G2_node,G2_neighbor)
        respect the isomorphism conditions.
        """
        if self.graph_matcher.test != "mono":
            return (
                G1_neighbor in G1_node_neighbors
                and len(G1_node_neighbors[G1_neighbor])
                == len(G2_node_neighbors[G2_neighbor])
                and self.graph_matcher.compare_edge_attr(
                    # G1_node,
                    # G2_node,
                    G1_node_neighbors[G1_neighbor],
                    G2_node_neighbors[G2_neighbor],
                )
            )
        else:
            return (
                G2_neighbor in G2_node_neighbors
                and len(G1_node_neighbors[G1_neighbor])
                <= len(G2_node_neighbors[G2_neighbor])
                and self.graph_matcher.compare_edge_attr(
                    # G1_node,
                    # G2_node,
                    G1_node_neighbors[G1_neighbor],
                    G2_node_neighbors[G2_neighbor],
                )
            )


class MultiGMSubState(GMSubState):
    """Class for managing feasibility sets for GraphMatcher."""

    def __init__(self, G, G_nodes_ind):
        """Initializes MultiGMSubState object."""
        super().__init__(G, G_nodes_ind)
        # Number of edges in the mapping
        self.c_e = [0] * self.G.number_of_nodes()

    # upadate Feasibility Sets

    def add_G1_node(self, node_id, node):
        neighbor_ind = 0
        for neighbor in self.G[node]:
            neighbor_ind = self.G_nodes_ind[neighbor]
            self.c[neighbor_ind] += 1
            self.c_e[neighbor_ind] += len(self.G[node][neighbor])
            self.c_sum[neighbor_ind] += node_id

    def add_G2_node(self, node_id, node, t_node_id):
        neighbor_ind = 0
        for neighbor in self.G[node]:
            neighbor_ind = self.G_nodes_ind[neighbor]
            self.c[neighbor_ind] += 1
            self.c_e[neighbor_ind] += len(self.G[node][neighbor])
            self.c_sum[neighbor_ind] += t_node_id

    def remove_G1_node(self, node_id, node):
        neighbor_ind = 0
        for neighbor in self.G[node]:
            neighbor_ind = self.G_nodes_ind[neighbor]
            self.c[neighbor_ind] -= 1
            self.c_e[neighbor_ind] -= len(self.G[node][neighbor])
            self.c_sum[neighbor_ind] -= node_id

    def remove_G2_node(self, node_id, node, t_node_id):
        neighbor_ind = 0
        for neighbor in self.G[node]:
            neighbor_ind = self.G_nodes_ind[neighbor]
            self.c[neighbor_ind] -= 1
            self.c_e[neighbor_ind] -= len(self.G[node][neighbor])
            self.c_sum[neighbor_ind] -= t_node_id

    # upadate Feasibility Sets


class MultiGMNodeInfo(GMNodeInfo):
    """A class to store node information about feasibility rules for MultiGraphMatcher"""

    def __init__(self, size):
        """Initialize the MultiGMNodeInfo object."""
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
    """Computes the node odering for DiGraph using attributes
    of the NodeOrderingProp class.
    """

    def update_no(self, no, level):
        """Marks a node as present and updates its neighbors' properties.

        Parameters:
            no: NodeOrderingProp corresponding to the node to be updated.
        """
        no.present = 1
        no.candidate = 1
        degM = no.degM if no.degM < self.limit else self.limit
        no.degM = 0
        node = no.node
        #
        for neighbor in self.graph_matcher.G1.neighbors(node):
            neighbor_ind = self.graph_matcher.G1_nodes_ind[neighbor]
            no_neigh = self.nodes[neighbor_ind]
            #
            if no.degMNeighMax < no_neigh.degM:
                no.degMNeighMax = no_neigh.degM
            #
            if no_neigh.present == 0:
                if self.nodes_limit[no_neigh.id] < self.limit:
                    self.update_degMNeigh_and_degMo(no_neigh)
                    self.nodes_limit[no_neigh.id] += 1
                #
                no_neigh.degM += 1
                no_neigh.degNeigh += self.graph_matcher.G1_o.number_of_edges(
                    node, neighbor
                ) + self.graph_matcher.G1_o.number_of_edges(neighbor, node)
                #
                no_neigh.degMNeigh -= degM
                if level == 0:
                    no_neigh.degMo -= 1

            if no_neigh.candidate == 0:
                no_neigh.candidate = 1
                self.candidates.append(no_neigh)
                self.parents[no_neigh.id] = no.id

        #
        self.candidates = [no for no in self.candidates if no.present == 0]


class DiGraphMatcher(GraphMatcher):
    """fastiso implementation for matching Digraphs"""

    def __init__(
        self,
        G1,
        G2,
        node_label=None,
        node_match=None,
        edge_match=None,
        path_label=False,
    ):
        """Initializes DiGraphMatcher.
        G1 and G2 have to be nx.DiGraph or nx.MultiDiGraph instances.

        Examples
        --------
        >>> from networkx.algorithms.isomorphism.isomorphfastiso import MultiGraphMatcher
        >>> G1 = nx.DiGraph(nx.path_graph(4, create_using=nx.DiGraph()))
        >>> G2 = nx.DiGraph(nx.path_graph(4, create_using=nx.DiGraph()))
        >>> gm = DiGraphMatcher(G1, G2, path_graph=True)
        """
        super().__init__(G1, G2, node_label, node_match, edge_match, path_label)
        #
        self.G1_degrees_in = [self.G1.in_degree[node] for node in self.G1_nodes]
        self.G1_degrees_out = [self.G1.out_degree[node] for node in self.G1_nodes]
        #
        self.G2_degrees_in = [self.G2.in_degree[node] for node in self.G2_nodes]
        self.G2_degrees_out = [self.G2.out_degree[node] for node in self.G2_nodes]

    def initialize_sate(self):
        """(Re)Initializes the state of DiGraphMatcher."""
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

    def compute_variable_ordering(self):
        """
        Calls the DiNodeCommand class to compute the variable ordering.
        """
        no_cmd = DiNodeCommand()
        no_cmd.compute_variable_ordering(self)
        self.nodes_order = no_cmd.nodes_order
        self.parents = no_cmd.parents
        self.nodes_degMNeighMax = no_cmd.nodes_degMNeighMax

    def _feasibility(
        self, G1_node_ind, G1_node, G2_node_ind, G2_node, G1_node_info, G2_node_info
    ):
        """check if the addition of the pair (G1_node, G2_node)
        respects the feasibility rules.
        """
        #
        c_e_in = self.state.G2_sub_state.c_e_in[G2_node_ind]
        c_e_out = self.state.G2_sub_state.c_e_out[G2_node_ind]
        if self.test == "iso":
            if (
                # Number of in edges in the mapping
                G1_node_info.c_e_in != c_e_in
                # Number of in edges not in the mapping
                or G1_node_info.nc_e_in != self.G2_degrees_out[G2_node_ind] - c_e_in
                # Number of out edges in the mapping
                or G1_node_info.c_e_out != c_e_out
                # Number of out edges not in the mapping
                or G1_node_info.nc_e_out != self.G2_degrees_in[G2_node_ind] - c_e_out
            ):
                return False
        elif self.test == "sub-iso" or self.test == "sub-isoM":
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

    def reset_graph_view(self):
        """Resets undirect graph view."""
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
        self.reset_graph_view()
        return result

    def subgraph_is_isomorphic(self):
        result = super().subgraph_is_isomorphic()
        self.reset_graph_view()
        return result

    def subgraph_is_isomorphic_M(self):
        result = super().subgraph_is_isomorphic_M()
        self.reset_graph_view()
        return result

    def subgraph_is_monomorphic(self):
        result = super().subgraph_is_monomorphic()
        self.reset_graph_view()
        return result

    ##

    def isomorphisms_iter(self):
        yield from super().isomorphisms_iter()
        self.reset_graph_view()

    def subgraph_isomorphisms_iter(self):
        yield from super().subgraph_isomorphisms_iter()
        self.reset_graph_view()

    def subgraph_isomorphisms_iter_M(self):
        yield from super().subgraph_isomorphisms_iter_M()
        self.reset_graph_view()

    def subgraph_monomorphisms_iter(self):
        yield from super().subgraph_monomorphisms_iter()
        self.reset_graph_view()


class DiGMState(GMState):
    """Internal representation of state for the DiGraphMatcher class."""

    def __init__(self, graph_matcher):
        """Initializes DiGMState object."""
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
        if self.graph_matcher.test == "sub-iso":
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
        """Computes feasibility rules information for the node "G1_node" in G1"""
        #
        super().compute_G1_node_info(node_info, G1_node_ind, G1_node)
        # Number of in edges in the mapping
        node_info.c_e_in = self.G1_sub_state.c_e_in[G1_node_ind]
        # Number of in edges not in the mapping
        node_info.nc_e_in = (
            self.graph_matcher.G1_degrees_out[G1_node_ind] - node_info.c_e_in
        )
        # Number of out edges in the mapping
        node_info.c_e_out = self.G1_sub_state.c_e_out[G1_node_ind]
        # Number of out edges not in the mapping
        node_info.nc_e_out = (
            self.graph_matcher.G1_degrees_in[G1_node_ind] - node_info.c_e_out
        )

    def compute_G2_node_info_and_verify_edge_feasibility(
        self, G1_node_ind, G1_node, G2_node_ind, G2_node, node_info
    ):
        """Computes feasibility rules information for the node
        "G2_node" in G2 and checks isomorphism constraints regarding edges
        """
        return super().compute_G2_node_info_and_verify_edge_feasibility(
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
        """Checks if the pairs of nodes (G1_node, G1_neighbor), (G2_node,G2_neighbor)
        respect the isomorphism conditions.
        """
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
                # G1_neighbor in G1_node_neighbors
                n_G1_node_G1_neighbor != n_G2_node_G2_neighbor
                or n_G1_neighbor_G1_node != n_G2_neighbor_G2_node
            ):
                return False

        else:
            if (
                # G2_neighbor in G2_node_neighbors
                n_G1_node_G1_neighbor > n_G2_node_G2_neighbor
                or n_G1_neighbor_G1_node > n_G2_neighbor_G2_node
            ):
                return False

        if n_G1_node_G1_neighbor != 0 and not self.graph_matcher.compare_edge_attr(
            # G1_node,
            # G2_node,
            self.graph_matcher.G1_o.succ[G1_node][G1_neighbor],
            self.graph_matcher.G2_o.succ[G2_node][G2_neighbor],
        ):
            return False

        if n_G1_neighbor_G1_node != 0 and not self.graph_matcher.compare_edge_attr(
            # G1_node,
            # G2_node,
            self.graph_matcher.G1_o.pred[G1_node][G1_neighbor],
            self.graph_matcher.G2_o.pred[G2_node][G2_neighbor],
        ):
            return False

        return True


class DiGMSubState(GMSubState):
    """Class for managing feasibility sets for DiGraphMatcher."""

    def __init__(self, G, G_o, G_nodes_ind):
        """Initializes DiGMSubState object."""
        super().__init__(G, G_nodes_ind)
        self.G_o = G_o
        # Number of in edges in the mapping
        self.c_e_in = [0] * self.G.number_of_nodes()
        # Number of out edges in the mapping
        self.c_e_out = [0] * self.G.number_of_nodes()

    # Upadate Feasibility Sets

    def add_G1_node(self, node_id, node):
        neighbor_ind = 0
        for neighbor in self.G[node]:
            neighbor_ind = self.G_nodes_ind[neighbor]
            self.c[neighbor_ind] += 1
            self.c_e_out[neighbor_ind] += self.G_o.number_of_edges(node, neighbor)
            self.c_e_in[neighbor_ind] += self.G_o.number_of_edges(neighbor, node)
            self.c_sum[neighbor_ind] += node_id

    def add_G2_node(self, node_id, node, t_node_id):
        neighbor_ind = 0
        for neighbor in self.G[node]:
            neighbor_ind = self.G_nodes_ind[neighbor]
            self.c[neighbor_ind] += 1
            self.c_e_out[neighbor_ind] += self.G_o.number_of_edges(node, neighbor)
            self.c_e_in[neighbor_ind] += self.G_o.number_of_edges(neighbor, node)
            self.c_sum[neighbor_ind] += t_node_id

    def remove_G1_node(self, node_id, node):
        neighbor_ind = 0
        for neighbor in self.G[node]:
            neighbor_ind = self.G_nodes_ind[neighbor]
            self.c[neighbor_ind] -= 1
            self.c_e_out[neighbor_ind] -= self.G_o.number_of_edges(node, neighbor)
            self.c_e_in[neighbor_ind] -= self.G_o.number_of_edges(neighbor, node)
            self.c_sum[neighbor_ind] -= node_id

    def remove_G2_node(self, node_id, node, t_node_id):
        neighbor_ind = 0
        for neighbor in self.G[node]:
            neighbor_ind = self.G_nodes_ind[neighbor]
            self.c[neighbor_ind] -= 1
            self.c_e_out[neighbor_ind] -= self.G_o.number_of_edges(node, neighbor)
            self.c_e_in[neighbor_ind] -= self.G_o.number_of_edges(neighbor, node)
            self.c_sum[neighbor_ind] -= t_node_id

    # Upadate Feasibility Sets


class DiGMNodeInfo(GMNodeInfo):
    """A class to store node information about
    feasibility rules for DiGraphMatcher."""

    def __init__(self, size):
        """
        Initializes the DiGMNodeInfo object.
        """
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
        """Resets all attributes to their default values."""
        super().clear()
        self.c_e_in = 0
        self.c_e_out = 0
        self.nc_e_in = 0
        self.nc_e_out = 0


"""""" """""" """""" """""" """""" """""" """""" """""" """""" """""" """""" """""" """""" """""" """""" """""" """""" """""" """""" """""" """""" """

                                                        multi-di-graph

""" """""" """""" """""" """""" """""" """""" """""" """""" """""" """""" """""" """""" """""" """""" """""" """""" """""" """""" """""" """""" """"""


class MultiDiGraphMatcher(DiGraphMatcher):
    def __init__(
        self,
        G1,
        G2,
        node_label=None,
        node_match=None,
        edge_match=None,
        path_label=False,
    ):
        """
        Initializes DiGraphMatcher.
        G1 and G2 have to be nx.DiGraph or nx.MultiDiGraph instances.

        Examples
        --------
        >>> from networkx.algorithms.isomorphism.isomorphfastiso import MultiGraphMatcher
        >>> G1 = nx.DiGraph(nx.path_graph(4, create_using=nx.DiGraph()))
        >>> G2 = nx.DiGraph(nx.path_graph(4, create_using=nx.DiGraph()))
        >>> gm = MultiDiGraphMatcher(G1, G2, path_graph=True)
        """
        super().__init__(G1, G2, node_label, node_match, edge_match, path_label)
