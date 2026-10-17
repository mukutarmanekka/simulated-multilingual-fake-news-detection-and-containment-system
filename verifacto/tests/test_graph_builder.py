"""Phase 1: the language-block graph has clear clusters, hubs, and bridges."""

import networkx as nx
import numpy as np

from verifacto import config
from verifacto.network.graph_builder import (
    block_node_lists,
    build_language_graph,
    cross_language_edge_fraction,
    in_block_density,
    load_graph,
    overall_cross_block_density,
    save_graph,
)

# The wiring ratio p_in/p_out is 10:1 by design; at these finite block
# sizes the measured in-block density is 9.8-11x the cross-block density,
# so the test threshold leaves a safety margin while still proving clusters.
CLUSTER_RATIO_MARGIN = 8.0


def test_node_count_and_languages():
    G = build_language_graph()
    assert G.number_of_nodes() == 500
    languages = {G.nodes[n]["language"] for n in G.nodes}
    assert languages == {"en", "hi", "fr"}
    for block, block_nodes in enumerate(G.graph["partition"]):
        assert len(block_nodes) == config.BLOCK_SIZES[block]


def test_is_directed():
    G = build_language_graph()
    assert G.is_directed()
    assert isinstance(G, nx.DiGraph)


def test_every_edge_has_spread_p():
    G = build_language_graph()
    for u, v, p in G.edges(data="p"):
        assert p == config.P_SPREAD


def test_cluster_structure():
    G = build_language_graph()
    # most edges stay inside a language block
    assert cross_language_edge_fraction(G) <= 0.15
    # edges inside a block are far denser than edges across blocks
    cross_density = overall_cross_block_density(G)
    for block_nodes in G.graph["partition"]:
        ratio = in_block_density(G, sorted(block_nodes)) / cross_density
        assert ratio >= CLUSTER_RATIO_MARGIN


def test_hubs_have_high_out_degree():
    G = build_language_graph()
    hubs = [n for n, d in G.nodes(data=True) if d["is_hub"]]
    assert len(hubs) == 3 * len(config.LANGUAGES)
    for hub in hubs:
        assert G.out_degree(hub) >= config.HUB_DEGREE_TARGET
    non_hubs = [n for n in G.nodes if not G.nodes[n]["is_hub"]]
    mean_non_hub = float(np.mean([G.out_degree(n) for n in non_hubs]))
    assert mean_non_hub < 35.0


def test_bridges_span_languages():
    G = build_language_graph()
    bridges = [n for n, d in G.nodes(data=True) if d["is_bridge"]]
    assert len(bridges) == 2 * len(config.LANGUAGES)
    for bridge in bridges:
        neighbor_languages = {G.nodes[v]["language"] for v in G.successors(bridge)}
        assert len(neighbor_languages) >= 2


def test_deterministic():
    first = build_language_graph()
    second = build_language_graph()
    assert set(first.nodes) == set(second.nodes)
    assert set(first.edges) == set(second.edges)
    for u, v in first.edges:
        assert first[u][v]["p"] == second[u][v]["p"]


def test_save_load_roundtrip(tmp_path):
    G = build_language_graph()
    path = tmp_path / "graph.graphml"
    save_graph(G, path)
    H = load_graph(path)
    assert set(H.nodes) == set(G.nodes)
    assert set(H.edges) == set(G.edges)
    for node in H.nodes:
        assert H.nodes[node]["language"] == G.nodes[node]["language"]
        assert H.nodes[node]["is_hub"] == G.nodes[node]["is_hub"]
        assert H.nodes[node]["is_bridge"] == G.nodes[node]["is_bridge"]
    for u, v in H.edges:
        assert H[u][v]["p"] == G[u][v]["p"]


def test_block_node_lists_match_partition():
    G = build_language_graph()
    rebuilt = block_node_lists(G)
    for block, block_nodes in enumerate(G.graph["partition"]):
        assert rebuilt[block] == sorted(block_nodes)
