"""Phase 3: the Independent Cascade engine and its draw-table contract."""

import networkx as nx
import numpy as np

from verifacto.simulation.cascade import build_draw_table, run_ic_cascade


def _path_graph(n, p):
    G = nx.DiGraph()
    for i in range(n):
        G.add_node(i)
    for i in range(n - 1):
        G.add_edge(i, i + 1, p=p)
    return G


def test_single_node():
    G = _path_graph(1, 0.1)
    series = run_ic_cascade(G, 0, seed=1)
    assert series == [1]


def test_p_one_full_spread():
    # every edge succeeds with p=1: one new infection per step
    G = _path_graph(3, 1.0)
    draws = build_draw_table(G, seed=5)
    series = run_ic_cascade(G, 0, draws=draws)
    assert series == [1, 1, 1]
    assert sum(series) == 3


def test_stops_when_frontier_empty():
    # draws above p on every edge: nothing spreads
    G = _path_graph(3, 0.5)
    draws = {(0, 1): 0.9, (1, 2): 0.9}
    series = run_ic_cascade(G, 0, draws=draws)
    assert series == [1]


def test_seed_is_required_without_draws():
    G = _path_graph(2, 0.5)
    try:
        run_ic_cascade(G, 0)
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_deterministic():
    G = _path_graph(3, 0.5)
    first = run_ic_cascade(G, 0, seed=7)
    second = run_ic_cascade(G, 0, seed=7)
    assert first == second


def test_draw_table_alignment():
    # the D3 contract in miniature: halving p on the same draws can only
    # shrink the reached set, never grow it
    rng = np.random.default_rng(99)
    G = nx.DiGraph()
    for u in range(12):
        G.add_node(u)
    for u in range(12):
        for v in range(12):
            if u != v and rng.random() < 0.25:
                G.add_edge(u, v, p=1.0)
    draws = build_draw_table(G, seed=3)

    full = {(u, v): 1.0 for u, v in G.edges()}
    half = {(u, v): 0.5 for u, v in G.edges()}

    series_full = run_ic_cascade(G, 0, p_map=full, draws=draws)
    series_half = run_ic_cascade(G, 0, p_map=half, draws=draws)
    assert sum(series_half) <= sum(series_full)


def test_p_one_matches_reachable_set():
    # with p=1 every edge succeeds, so the cascade reaches exactly the
    # nodes reachable from the source and counts each exactly once
    rng = np.random.default_rng(7)
    G = nx.DiGraph()
    for u in range(30):
        G.add_node(u)
    for u in range(30):
        for v in range(30):
            if u != v and rng.random() < 0.2:
                G.add_edge(u, v, p=1.0)
    series = run_ic_cascade(G, 0, seed=11)
    reachable = set(nx.descendants(G, 0)) | {0}
    assert sum(series) == len(reachable)
