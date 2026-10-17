"""Step 5: targeted containment — throttle the top k users.

Selection methods: out-degree, betweenness, PageRank, plus a random-k
control. All centralities are computed on the ORIGINAL graph, and the
selection never depends on any post's content or detection.
"""

import networkx as nx

from ..simulation.intervention import select_top_k


def precompute_centralities(G):
    """Centrality scores for every deterministic method, computed once."""
    return {
        "degree": dict(G.out_degree()),
        "betweenness": nx.betweenness_centrality(G),
        "pagerank": nx.pagerank(G),
    }


def top_k_from_scores(scores, k):
    """Top-k node ids from a {node: score} dict, ties broken by node id."""
    ranked = sorted(scores, key=lambda node: scores[node], reverse=True)
    return ranked[:k]
