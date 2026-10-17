"""Phase 6: bridge detection finds the planted bilingual bridges."""

from verifacto.containment.bridges import (
    BRIDGE_CROSS_DEGREE_THRESHOLD,
    check_louvain_matches_blocks,
    detect_bridges,
)


def test_detect_bridges_finds_planted(graph):
    bridges = detect_bridges(graph)
    planted = [n for n, d in graph.nodes(data=True) if d["is_bridge"]]
    # every planted bridge has 10+ cross-language out-edges, well above
    # the threshold, so it must appear among the detected bridges
    for node in planted:
        assert node in bridges


def test_detected_bridges_meet_threshold(graph):
    bridges = detect_bridges(graph)
    for node, (cross_degree, languages_reached) in bridges.items():
        assert cross_degree >= BRIDGE_CROSS_DEGREE_THRESHOLD
        assert languages_reached >= 2
        assert node in graph.nodes


def test_bridge_set_is_small(graph):
    # heavy bilingual connectors are rare: a handful, not half the graph
    bridges = detect_bridges(graph)
    assert 6 <= len(bridges) <= 30


def test_louvain_recovers_language_blocks(graph):
    assert check_louvain_matches_blocks(graph) is True
