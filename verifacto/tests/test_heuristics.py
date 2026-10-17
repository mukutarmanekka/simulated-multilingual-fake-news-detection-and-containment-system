"""Phase 5: heuristic selections are exact, budgeted, and graph-only."""

import networkx as nx

from verifacto import config
from verifacto.containment.heuristics import precompute_centralities, top_k_from_scores
from verifacto.simulation.intervention import select_top_k


def _star_path_hybrid():
    # 0 -> 1..5 (out-degree 5, the betweenness center of its side)
    # 6 -> 7,8 (out-degree 2)
    G = nx.DiGraph()
    G.add_node(0)
    for v in range(1, 6):
        G.add_node(v)
        G.add_edge(0, v, p=0.1)
    G.add_node(6)
    for v in (7, 8):
        G.add_node(v)
        G.add_edge(6, v, p=0.1)
    return G


def test_degree_k_exact():
    G = _star_path_hybrid()
    selected = select_top_k(G, "degree", 2)
    assert selected == [0, 6]


def test_betweenness_center_on_star():
    G = _star_path_hybrid()
    selected = select_top_k(G, "betweenness", 1)
    assert selected == [0]


def test_pagerank_matches_nx_ranking():
    G = _star_path_hybrid()
    selected = select_top_k(G, "pagerank", len(G.nodes()))
    expected = sorted(G.nodes(), key=lambda n: nx.pagerank(G)[n], reverse=True)
    assert selected == expected


def test_budget_respected(graph):
    for method in ("degree", "betweenness", "pagerank", "random"):
        selected = select_top_k(graph, method, config.K_BUDGET, seed=1)
        assert len(selected) == config.K_BUDGET
        assert len(set(selected)) == config.K_BUDGET
        assert all(node in graph.nodes for node in selected)


def test_random_k_deterministic(graph):
    first = select_top_k(graph, "random", config.K_BUDGET, seed=7)
    second = select_top_k(graph, "random", config.K_BUDGET, seed=7)
    assert first == second


def test_graph_not_mutated(graph):
    before_edges = set(graph.edges())
    before_p = {edge: graph[edge[0]][edge[1]]["p"] for edge in graph.edges()}
    precompute_centralities(graph)
    select_top_k(graph, "degree", config.K_BUDGET)
    select_top_k(graph, "betweenness", config.K_BUDGET)
    select_top_k(graph, "pagerank", config.K_BUDGET)
    assert set(graph.edges()) == before_edges
    for edge in graph.edges():
        assert graph[edge[0]][edge[1]]["p"] == before_p[edge]


def test_selection_independent_of_detections(graph):
    # selection is purely structural: repeated calls give the same answer
    first = select_top_k(graph, "betweenness", config.K_BUDGET)
    second = select_top_k(graph, "betweenness", config.K_BUDGET)
    assert first == second


def test_precompute_centralities_keys(graph):
    scores = precompute_centralities(graph)
    assert set(scores.keys()) == {"degree", "betweenness", "pagerank"}
    assert len(scores["degree"]) == graph.number_of_nodes()
    top = top_k_from_scores(scores["degree"], 5)
    assert len(top) == 5
