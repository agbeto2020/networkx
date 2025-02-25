"""
    Tests for fastiso isomorphism algorithm.
"""

import importlib.resources
import os
import random
import struct

import networkx as nx
from networkx.algorithms import isomorphism as iso
from networkx.algorithms.isomorphism.isomorphfastiso import (
    DiGraphMatcher,
    GraphMatcher,
    MultiDiGraphMatcher,
    MultiGraphMatcher,
)


class TestWikipediaExample:
    # Source: https://en.wikipedia.org/wiki/Graph_isomorphism

    # Nodes 'a', 'b', 'c' and 'd' form a column.
    # Nodes 'g', 'h', 'i' and 'j' form a column.
    g1edges = [
        ["a", "g"],
        ["a", "h"],
        ["a", "i"],
        ["b", "g"],
        ["b", "h"],
        ["b", "j"],
        ["c", "g"],
        ["c", "i"],
        ["c", "j"],
        ["d", "h"],
        ["d", "i"],
        ["d", "j"],
    ]

    # Nodes 1,2,3,4 form the clockwise corners of a large square.
    # Nodes 5,6,7,8 form the clockwise corners of a small square
    g2edges = [
        [1, 2],
        [2, 3],
        [3, 4],
        [4, 1],
        [5, 6],
        [6, 7],
        [7, 8],
        [8, 5],
        [1, 5],
        [2, 6],
        [3, 7],
        [4, 8],
    ]

    def test_graph(self):
        g1 = nx.Graph()
        g2 = nx.Graph()
        g1.add_edges_from(self.g1edges)
        g2.add_edges_from(self.g2edges)
        gm = GraphMatcher(g1, g2)
        assert gm.is_isomorphic()
        # Just testing some cases
        assert gm.subgraph_is_monomorphic()

        mapping = sorted(gm.mapping.items())

    # this mapping is only one of the possibilities
    # so this test needs to be reconsidered
    #        isomap = [('a', 1), ('b', 6), ('c', 3), ('d', 8),
    #                  ('g', 2), ('h', 5), ('i', 4), ('j', 7)]
    #        assert_equal(mapping, isomap)

    def test_subgraph(self):
        g1 = nx.Graph()
        g2 = nx.Graph()
        g1.add_edges_from(self.g1edges)
        g2.add_edges_from(self.g2edges)
        g3 = g2.subgraph([1, 2, 3, 4])
        gm = GraphMatcher(g1, g3)
        assert gm.subgraph_is_isomorphic()
        assert gm.subgraph_is_isomorphic_M()

    def test_subgraph_mono(self):
        g1 = nx.Graph()
        g2 = nx.Graph()
        g1.add_edges_from(self.g1edges)
        g2.add_edges_from([[1, 2], [2, 3], [3, 4]])
        gm = GraphMatcher(g1, g2)
        assert gm.subgraph_is_monomorphic()


class TestVF2GraphDB:
    # https://web.archive.org/web/20090303210205/http://amalfi.dis.unina.it/graph/db/

    @staticmethod
    def create_graph(filename):
        """Creates a Graph instance from the filename."""

        # The file is assumed to be in the format from the VF2 graph database.
        # Each file is composed of 16-bit numbers (unsigned short int).
        # So we will want to read 2 bytes at a time.

        # We can read the number as follows:
        #   number = struct.unpack('<H', file.read(2))
        # This says, expect the data in little-endian encoding
        # as an unsigned short int and unpack 2 bytes from the file.

        fh = open(filename, mode="rb")

        # Grab the number of nodes.
        # Node numeration is 0-based, so the first node has index 0.
        nodes = struct.unpack("<H", fh.read(2))[0]

        graph = nx.Graph()
        for from_node in range(nodes):
            # Get the number of edges.
            edges = struct.unpack("<H", fh.read(2))[0]
            for edge in range(edges):
                # Get the terminal node.
                to_node = struct.unpack("<H", fh.read(2))[0]
                graph.add_edge(from_node, to_node)

        fh.close()
        return graph

    def test_graph(self):
        head = importlib.resources.files("networkx.algorithms.isomorphism.tests")
        g1 = self.create_graph(head / "iso_r01_s80.A99")
        g2 = self.create_graph(head / "iso_r01_s80.B99")
        gm = GraphMatcher(g1, g2)
        assert gm.is_isomorphic()

    def test_subgraph(self):
        # A is the subgraph
        # B is the full graph
        head = importlib.resources.files("networkx.algorithms.isomorphism.tests")
        subgraph = self.create_graph(head / "si2_b06_m200.A99")
        graph = self.create_graph(head / "si2_b06_m200.B99")
        gm = GraphMatcher(graph, subgraph)
        assert gm.subgraph_is_isomorphic()
        assert gm.subgraph_is_isomorphic_M()
        # Just testing some cases
        assert gm.subgraph_is_monomorphic()

    # There isn't a similar test implemented for subgraph monomorphism,
    # feel free to create one.


# class TestAtlas:
#     @classmethod
#     def setup_class(cls):
#         global atlas
#         from networkx.generators import atlas

#         cls.GAG = atlas.graph_atlas_g()

#     def test_graph_atlas(self):
#         # Atlas = nx.graph_atlas_g()[0:208] # 208, 6 nodes or less
#         Atlas = self.GAG[0:100]
#         alphabet = list(range(26))
#         for graph in Atlas:
#             nlist = list(graph)
#             labels = alphabet[: len(nlist)]
#             for s in range(10):
#                 random.shuffle(labels)
#                 d = dict(zip(nlist, labels))
#                 relabel = nx.relabel_nodes(graph, d)
#                 gm = GraphMatcher(graph, relabel)
#                 assert gm.is_isomorphic()


def test_multiedge():
    # Simple test for multigraphs
    # Need something much more rigorous
    edges = [
        (0, 1),
        (1, 2),
        (2, 3),
        (3, 4),
        (4, 5),
        (5, 6),
        (6, 7),
        (7, 8),
        (8, 9),
        (9, 10),
        (10, 11),
        (10, 11),
        (11, 12),
        (11, 12),
        (12, 13),
        (12, 13),
        (13, 14),
        (13, 14),
        (14, 15),
        (14, 15),
        (15, 16),
        (15, 16),
        (16, 17),
        (16, 17),
        (17, 18),
        (17, 18),
        (18, 19),
        (18, 19),
        (19, 0),
        (19, 0),
    ]
    nodes = list(range(20))

    for g1 in [nx.MultiGraph(), nx.MultiDiGraph()]:
        g1.add_edges_from(edges)
        for _ in range(10):
            new_nodes = list(nodes)
            random.shuffle(new_nodes)
            d = dict(zip(nodes, new_nodes))
            g2 = nx.relabel_nodes(g1, d)
            if not g1.is_directed():
                gm = MultiGraphMatcher(g1, g2)
            else:
                gm = MultiDiGraphMatcher(g1, g2)
            assert gm.is_isomorphic()
            # Testing if monomorphism works in multigraphs
            assert gm.subgraph_is_monomorphic()


def test_selfloop():
    # Simple test for graphs with selfloops
    edges = [
        (0, 1),
        (0, 2),
        (1, 2),
        (1, 3),
        (2, 2),
        (2, 4),
        (3, 1),
        (3, 2),
        (4, 2),
        (4, 5),
        (5, 4),
    ]
    nodes = list(range(6))

    for g1 in [nx.Graph(), nx.DiGraph()]:
        g1.add_edges_from(edges)
        for _ in range(100):
            new_nodes = list(nodes)
            random.shuffle(new_nodes)
            d = dict(zip(nodes, new_nodes))
            g2 = nx.relabel_nodes(g1, d)
            if not g1.is_directed():
                gm = GraphMatcher(g1, g2)
            else:
                gm = DiGraphMatcher(g1, g2)
            assert gm.is_isomorphic()


def test_selfloop_mono():
    # Simple test for graphs with selfloops
    edges0 = [
        (0, 1),
        (0, 2),
        (1, 2),
        (1, 3),
        (2, 4),
        (3, 1),
        (3, 2),
        (4, 2),
        (4, 5),
        (5, 4),
    ]
    edges = edges0 + [(2, 2)]
    nodes = list(range(6))

    for g1 in [nx.Graph(), nx.DiGraph()]:
        g1.add_edges_from(edges)
        for _ in range(100):
            new_nodes = list(nodes)
            random.shuffle(new_nodes)
            d = dict(zip(nodes, new_nodes))
            g2 = nx.relabel_nodes(g1, d)
            g2.remove_edges_from(nx.selfloop_edges(g2))
            if not g1.is_directed():
                gm = GraphMatcher(g2, g1)
            else:
                gm = DiGraphMatcher(g2, g1)
            assert not gm.subgraph_is_monomorphic()


def test_isomorphism_iter1():
    # As described in:
    # http://groups.google.com/group/networkx-discuss/browse_thread/thread/2ff65c67f5e3b99f/d674544ebea359bb?fwc=1
    g1 = nx.DiGraph()
    g2 = nx.DiGraph()
    g3 = nx.DiGraph()
    g1.add_edge("A", "B")
    g1.add_edge("B", "C")
    g2.add_edge("Y", "Z")
    g3.add_edge("Z", "Y")
    gm12 = DiGraphMatcher(g1, g2)
    gm13 = DiGraphMatcher(g1, g3)
    x = list(gm12.subgraph_isomorphisms_iter())
    y = list(gm13.subgraph_isomorphisms_iter_M())
    assert {"A": "Y", "B": "Z"} in x
    assert {"B": "Y", "C": "Z"} in x
    assert {"A": "Z", "B": "Y"} in y
    assert {"B": "Z", "C": "Y"} in y
    assert len(x) == len(y)
    assert len(x) == 2


def test_monomorphism_iter1():
    g1 = nx.DiGraph()
    g2 = nx.DiGraph()
    g1.add_edge("A", "B")
    g1.add_edge("B", "C")
    g1.add_edge("C", "A")
    g2.add_edge("X", "Y")
    g2.add_edge("Y", "Z")
    gm12 = DiGraphMatcher(g1, g2)
    x = list(gm12.subgraph_monomorphisms_iter())
    assert {"A": "X", "B": "Y", "C": "Z"} in x
    assert {"A": "Y", "B": "Z", "C": "X"} in x
    assert {"A": "Z", "B": "X", "C": "Y"} in x
    assert len(x) == 3
    gm21 = DiGraphMatcher(g2, g1)
    # Check if StopIteration exception returns False
    assert not gm21.subgraph_is_monomorphic()


def test_isomorphism_iter2():
    # Path
    for L in range(2, 10):
        g1 = nx.path_graph(L)
        gm = GraphMatcher(g1, g1)
        s = len(list(gm.isomorphisms_iter()))
        assert s == 2
    # Cycle
    for L in range(3, 10):
        g1 = nx.cycle_graph(L)
        gm = GraphMatcher(g1, g1)
        s = len(list(gm.isomorphisms_iter()))
        assert s == 2 * L


def test_multiple():
    # Verify that we can use the graph matcher multiple times
    edges = [("A", "B"), ("B", "A"), ("B", "C")]
    for g1, g2 in [(nx.Graph(), nx.Graph()), (nx.DiGraph(), nx.DiGraph())]:
        g1.add_edges_from(edges)
        g2.add_edges_from(edges)
        g3 = nx.subgraph(g2, ["A", "B"])
        if not g1.is_directed():
            gmA = GraphMatcher(g1, g2)
            gmB = GraphMatcher(g1, g3)
        else:
            gmA = DiGraphMatcher(g1, g2)
            gmB = DiGraphMatcher(g1, g3)
        assert gmA.is_isomorphic()
        g2.remove_node("C")
        if not g1.is_directed():
            gmA = GraphMatcher(g1, g2)
        else:
            gmA = DiGraphMatcher(g1, g2)
        assert gmA.subgraph_is_isomorphic()
        assert gmB.subgraph_is_isomorphic()
        assert gmA.subgraph_is_monomorphic()
        assert gmB.subgraph_is_monomorphic()


#        for m in [gmB.mapping, gmB.mapping]:
#            assert_true(m['A'] == 'A')
#            assert_true(m['B'] == 'B')
#            assert_true('C' not in m)


def test_noncomparable_nodes():
    node1 = object()
    node2 = object()
    node3 = object()

    # Graph
    G = nx.path_graph([node1, node2, node3])
    gm = GraphMatcher(G, G)
    assert gm.is_isomorphic()
    # Just testing some cases
    assert gm.subgraph_is_monomorphic()

    # DiGraph
    G = nx.path_graph([node1, node2, node3], create_using=nx.DiGraph)
    H = nx.path_graph([node3, node2, node1], create_using=nx.DiGraph)
    dgm = DiGraphMatcher(G, H)
    assert dgm.is_isomorphic()
    # Just testing some cases
    assert gm.subgraph_is_monomorphic()


def test_monomorphism_edge_match():
    G = nx.DiGraph()
    G.add_node(1)
    G.add_node(2)
    G.add_edge(1, 2, label="A")
    G.add_edge(2, 1, label="B")
    G.add_edge(2, 2, label="C")

    SG = nx.DiGraph()
    SG.add_node(5)
    SG.add_node(6)
    SG.add_edge(5, 6, label="A")

    gm = DiGraphMatcher(G, SG, edge_match=iso.categorical_edge_match("label", None))
    assert gm.subgraph_is_monomorphic()


def match(datasets1, datasets2):
    values1 = {data.get("color", -1) for data in datasets1.values()}
    values2 = {data.get("color", -1) for data in datasets2.values()}
    return values1 >= values2


def test_monomorphism_multigraph_edge_match():
    # source:
    graph = nx.MultiGraph([[0, 1, {"color": 0}], [0, 1, {"color": 1}]])
    subgraph = nx.MultiGraph([[0, 1, {"color": 0}]])

    gm = MultiGraphMatcher(graph, subgraph, edge_match=match)
    assert gm.subgraph_is_monomorphic()
    ##
    graph = nx.MultiGraph([[0, 1, {"color": 0}], [0, 1, {"color": 0}]])
    subgraph = nx.MultiGraph([[0, 1, {"color": 0}]])
    gm = MultiGraphMatcher(
        graph, subgraph, edge_match=iso.categorical_multiedge_match("color", -1)
    )
    assert gm.subgraph_is_monomorphic()


def test_multidigraph_isomorphism():
    # source: https://github.com/networkx/networkx/issues/6257#issue-1479584767
    g = nx.MultiDiGraph({0: [1, 1, 2, 2, 3], 1: [2, 3, 3], 2: [3]})
    h = nx.MultiDiGraph({0: [1, 1, 2, 2, 3], 1: [2, 3, 3], 3: [2]})
    gm = MultiDiGraphMatcher(g, h)
    assert not gm.is_isomorphic()


def test_small_graph():
    # source: https://github.com/networkx/networkx/issues/4019#issuecomment-649457761
    # FASTiso do it in 64 seconds
    # vf2 do it in 597 seconds
    # vf2pp do it in 1189 seconds
    source_graph_nodes = [
        0,
        1,
        2,
        3,
        4,
        5,
        6,
        7,
        8,
        9,
        10,
        11,
        12,
        13,
        14,
        15,
        16,
        17,
        18,
        19,
        20,
        21,
        22,
        23,
        24,
        25,
        26,
        27,
        28,
        29,
        30,
        31,
        32,
        33,
        34,
        35,
        36,
        37,
        38,
        39,
        40,
        41,
        42,
        43,
        44,
        45,
        46,
        47,
        48,
        49,
        50,
    ]
    source_graph_edges = [
        (0, 1),
        (1, 2),
        (2, 3),
        (3, 4),
        (0, 5),
        (0, 6),
        (0, 7),
        (1, 8),
        (1, 9),
        (2, 10),
        (2, 11),
        (3, 12),
        (3, 13),
        (4, 14),
        (4, 15),
        (4, 16),
        (17, 18),
        (17, 19),
        (17, 20),
        (20, 21),
        (21, 22),
        (20, 23),
        (17, 24),
        (20, 25),
        (21, 26),
        (22, 27),
        (21, 28),
        (22, 29),
        (22, 30),
        (23, 31),
        (23, 32),
        (23, 33),
        (34, 35),
        (35, 36),
        (35, 37),
        (35, 38),
        (34, 39),
        (37, 40),
        (37, 41),
        (37, 42),
        (34, 43),
        (34, 44),
        (38, 45),
        (38, 46),
        (38, 47),
        (36, 48),
        (36, 49),
        (36, 50),
    ]

    target_graph_nodes = [
        0,
        1,
        2,
        3,
        4,
        5,
        6,
        7,
        8,
        9,
        10,
        11,
        12,
        13,
        14,
        15,
        16,
        17,
        18,
        19,
        20,
        21,
        22,
        23,
        24,
        25,
        26,
        27,
        28,
        29,
        30,
        31,
        32,
        33,
        34,
        35,
        36,
        37,
        38,
        39,
        40,
        41,
        42,
        43,
        44,
        45,
        46,
        47,
        48,
        49,
        50,
    ]
    target_graph_edges = [
        (0, 1),
        (1, 2),
        (2, 3),
        (3, 4),
        (4, 5),
        (3, 6),
        (2, 7),
        (1, 8),
        (0, 9),
        (0, 10),
        (0, 11),
        (1, 12),
        (2, 13),
        (3, 14),
        (4, 15),
        (4, 16),
        (17, 18),
        (18, 19),
        (19, 20),
        (17, 21),
        (18, 22),
        (19, 23),
        (20, 24),
        (20, 25),
        (19, 26),
        (20, 27),
        (17, 28),
        (17, 29),
        (18, 30),
        (30, 31),
        (30, 32),
        (30, 33),
        (34, 35),
        (35, 36),
        (36, 37),
        (34, 38),
        (35, 39),
        (36, 40),
        (37, 41),
        (37, 42),
        (37, 43),
        (35, 44),
        (34, 45),
        (34, 46),
        (36, 47),
        (47, 48),
        (47, 49),
        (47, 50),
    ]

    source_graph = nx.Graph()
    source_graph.add_nodes_from(source_graph_nodes)
    source_graph.add_edges_from(source_graph_edges)

    target_graph = nx.Graph()
    target_graph.add_nodes_from(target_graph_nodes)
    target_graph.add_edges_from(target_graph_edges)

    gm = GraphMatcher(source_graph, target_graph)
    assert not gm.is_isomorphic()
