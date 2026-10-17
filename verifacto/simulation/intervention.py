"""Step 4: intervention actions applied to flagged posts.

Interventions never mutate the graph. They produce per-edge probability
maps (D4): effective_p(u,v) = G[u][v]["p"] * post_factor * user_factor(u).
Centralities and throttled-user selection are always computed on the
ORIGINAL graph.
"""

import networkx as nx
import numpy as np

from ..config import BLANKET_REDUCTION, TARGETED_REDUCTION


def blanket_factor(detection):
    """Post-wide spread factor: halve flagged fakes, leave the rest alone."""
    if detection is not None and detection.is_fake is True:
        return BLANKET_REDUCTION
    return 1.0


def build_prob_map(G, post_factor, throttled=None, throttle_factor=TARGETED_REDUCTION):
    """Effective per-edge probabilities for one run; G is never mutated."""
    p_map = {(u, v): G[u][v]["p"] * post_factor for u, v in G.edges()}
    if throttled:
        throttled_set = set(throttled)
        for u, v in G.edges():
            if u in throttled_set:
                p_map[(u, v)] = p_map[(u, v)] * throttle_factor
    return p_map


def run_detection(adapter, posts):
    """One detection per post; the adapter receives only text/language/id."""
    detections = {}
    for post in posts:
        detections[post.id] = adapter.detect(post.text, post.language, post.id)
    return detections


def select_top_k(G, method, k, seed=0):
    """Top-k user ids by the given method, on the original graph.

    Methods: "degree" (out-degree), "betweenness", "pagerank", "random".
    The random control is drawn from a seeded rng so it is reproducible.
    """
    if method == "degree":
        scores = dict(G.out_degree())
    elif method == "betweenness":
        scores = nx.betweenness_centrality(G)
    elif method == "pagerank":
        scores = nx.pagerank(G)
    elif method == "random":
        rng = np.random.default_rng(seed)
        chosen = rng.choice(sorted(G.nodes()), size=k, replace=False)
        return sorted(int(node) for node in chosen)
    else:
        raise ValueError(f"unknown selection method: {method}")
    ranked = sorted(scores, key=lambda node: scores[node], reverse=True)
    return ranked[:k]
