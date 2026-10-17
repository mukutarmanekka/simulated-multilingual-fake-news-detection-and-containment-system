"""Phase 6: greedy selection is exact on small graphs and label-free."""

import networkx as nx

from verifacto.containment.greedy import _flagged_posts, greedy_select
from verifacto.data.posts import Post
from verifacto.detector.base import DetectionResult


def _chain_graph(n):
    G = nx.DiGraph()
    for i in range(n):
        G.add_node(i)
    for i in range(n - 1):
        G.add_edge(i, i + 1, p=1.0)
    return G


def _detection(is_fake):
    summary = "REFUTED" if is_fake else "SUPPORTED"
    return DetectionResult(
        is_fake=is_fake, confidence=0.8, verdict_summary=summary, provider="stub"
    )


def test_greedy_throttles_the_author_first_hop():
    # chain 0 -> 1 -> 2 -> 3: throttling node 1 stops the cascade at the
    # author; any other single throttle saves fewer users
    G = _chain_graph(4)
    posts = [Post(id=0, text="t", true_label="fake", author=0, language="en")]
    detections = {0: _detection(True)}
    selected = greedy_select(G, posts, detections, k=2, n_runs=1, candidates=[1, 2, 3])
    assert selected == [1]


def test_greedy_stops_when_no_gain_remains():
    G = _chain_graph(4)
    posts = [Post(id=0, text="t", true_label="fake", author=0, language="en")]
    detections = {0: _detection(True)}
    # after throttling node 1 nothing is left to save; greedy stops early
    selected = greedy_select(G, posts, detections, k=3, n_runs=1, candidates=[1, 2, 3])
    assert selected == [1]


def test_flagged_posts_only():
    posts = [
        Post(id=0, text="t", true_label="fake", author=0, language="en"),
        Post(id=1, text="t", true_label="real", author=0, language="en"),
        Post(id=2, text="t", true_label="fake", author=0, language="en"),
    ]
    detections = {0: _detection(True), 1: _detection(False), 2: _detection(False)}
    flagged = _flagged_posts(posts, detections)
    assert [post.id for post in flagged] == [0]
