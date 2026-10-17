"""Phase 6: flagged posts are ranked by confidence x expected reach."""

import networkx as nx

from verifacto.containment.priority import expected_reach, rank_flagged_posts
from verifacto.data.posts import Post
from verifacto.detector.base import DetectionResult


def _chain_graph(n):
    G = nx.DiGraph()
    for i in range(n):
        G.add_node(i)
    for i in range(n - 1):
        G.add_edge(i, i + 1, p=1.0)
    return G


def _detection(is_fake, confidence):
    summary = "REFUTED" if is_fake else "SUPPORTED"
    return DetectionResult(
        is_fake=is_fake, confidence=confidence, verdict_summary=summary, provider="stub"
    )


def test_expected_reach_grows_with_audience():
    # an author early in a p=1 chain reaches more users than a late one
    G = _chain_graph(10)
    early = Post(id=0, text="t", true_label="fake", author=0, language="en")
    late = Post(id=1, text="t", true_label="fake", author=8, language="en")
    assert expected_reach(G, early, eval_runs=1) == 10
    assert expected_reach(G, late, eval_runs=1) == 2


def test_rank_flagged_posts_filters_and_orders():
    G = _chain_graph(10)
    posts = [
        Post(id=0, text="t", true_label="fake", author=0, language="en"),
        Post(id=1, text="t", true_label="fake", author=8, language="en"),
        Post(id=2, text="t", true_label="real", author=0, language="en"),
    ]
    detections = {
        0: _detection(True, 0.9),
        1: _detection(True, 0.9),
        2: _detection(False, 0.9),
    }
    ranked = rank_flagged_posts(G, posts, detections, eval_runs=1)
    # only the flagged posts appear, most harmful first
    assert [item[0] for item in ranked] == [0, 1]
    assert ranked[0][1] == 0.9 * 10
    assert ranked[1][1] == 0.9 * 2
