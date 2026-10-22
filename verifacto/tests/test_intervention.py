"""Phase 4: interventions change only probabilities, never the graph, and
provably cannot increase reach (the D3 dominance contract)."""

import numpy as np
import pytest

from verifacto import config
from verifacto.data.posts import generate_posts
from verifacto.detector.base import DetectionResult
from verifacto.detector.stub import StubDetector
from verifacto.simulation.experiment import (
    build_strategy_p_maps,
    run_post_experiment,
)
from verifacto.simulation.intervention import (
    blanket_factor,
    build_prob_map,
    run_detection,
    select_top_k,
)


def _fake_detection():
    return DetectionResult(is_fake=True, confidence=0.8, verdict_summary="REFUTED", provider="stub")


def _real_detection():
    return DetectionResult(is_fake=False, confidence=0.8, verdict_summary="SUPPORTED", provider="stub")


def _uncertain_detection():
    return DetectionResult(
        is_fake=None,
        confidence=0.5,
        verdict_summary="UNCERTAIN",
        provider="deepseek",
        verdict="UNCERTAIN",
        detector_mode="real",
    )


def test_blanket_halves_flagged(graph):
    p_map = build_prob_map(graph, blanket_factor(_fake_detection()))
    for u, v in graph.edges():
        assert p_map[(u, v)] == graph[u][v]["p"] * config.BLANKET_REDUCTION


def test_blanket_unchanged_when_real_or_uncertain(graph):
    assert blanket_factor(_real_detection()) == 1.0
    assert blanket_factor(None) == 1.0
    p_map = build_prob_map(graph, 1.0)
    for u, v in graph.edges():
        assert p_map[(u, v)] == graph[u][v]["p"]


def test_targeted_only_selected_users(graph):
    selected = select_top_k(graph, "degree", config.K_BUDGET)
    p_map = build_prob_map(graph, 1.0, throttled=selected)
    for u, v in graph.edges():
        if u in selected:
            assert p_map[(u, v)] == graph[u][v]["p"] * config.TARGETED_REDUCTION
        else:
            assert p_map[(u, v)] == graph[u][v]["p"]


def test_real_and_uncertain_get_no_targeted_intervention(graph):
    for detection in (_real_detection(), _uncertain_detection()):
        p_maps = build_strategy_p_maps(graph, detection, "degree", 3)
        assert all(p_map is None for p_map in p_maps.values())


def test_uncertain_gets_no_blanket_intervention(graph):
    assert blanket_factor(_uncertain_detection()) == 1.0
    p_maps = build_strategy_p_maps(graph, _uncertain_detection(), "blanket", 2)
    assert all(p_map is None for p_map in p_maps.values())


@pytest.mark.parametrize(
    "strategy",
    ["baseline", "blanket", "random", "degree", "betweenness", "pagerank"],
)
def test_every_strategy_handles_uncertain_without_action(graph, strategy):
    p_maps = build_strategy_p_maps(
        graph, _uncertain_detection(), strategy, 2
    )
    assert set(p_maps) == {0, 1}
    assert all(p_map is None for p_map in p_maps.values())


def test_graph_never_mutated(graph):
    before_edges = set(graph.edges())
    before_p = {edge: graph[edge[0]][edge[1]]["p"] for edge in graph.edges()}
    selected = select_top_k(graph, "degree", config.K_BUDGET)
    build_prob_map(graph, 0.5, throttled=selected)
    assert set(graph.edges()) == before_edges
    for edge in graph.edges():
        assert graph[edge[0]][edge[1]]["p"] == before_p[edge]


def test_dominance_blanket(graph, posts):
    # for fixed (post, run) pairs, cumulative reach under the blanket can
    # never exceed the baseline at any step
    detector = StubDetector(posts)
    detections = run_detection(detector, posts)
    checked = 0
    for post in posts:
        detection = detections[post.id]
        if detection.is_fake is not True:
            continue
        baseline_p_maps = build_strategy_p_maps(graph, detection, "baseline", 6)
        blanket_p_maps = build_strategy_p_maps(graph, detection, "blanket", 6)
        baseline_results = run_post_experiment(graph, post, baseline_p_maps, 6, "baseline")
        blanket_results = run_post_experiment(graph, post, blanket_p_maps, 6, "blanket")
        for run in range(6):
            base = baseline_results[run].newly_reached_by_step
            protected = blanket_results[run].newly_reached_by_step
            base_cum = 0
            prot_cum = 0
            for step in range(len(base)):
                base_cum += base[step]
                if step < len(protected):
                    prot_cum += protected[step]
                assert prot_cum <= base_cum
        checked += 1
        if checked >= 4:
            break
    assert checked >= 1


def test_dominance_targeted(graph, posts):
    detector = StubDetector(posts)
    detections = run_detection(detector, posts)
    checked = 0
    for post in posts:
        detection = detections[post.id]
        if detection.is_fake is not True:
            continue
        baseline_p_maps = build_strategy_p_maps(graph, detection, "baseline", 4)
        targeted_p_maps = build_strategy_p_maps(graph, detection, "degree", 4)
        baseline_results = run_post_experiment(graph, post, baseline_p_maps, 4, "baseline")
        targeted_results = run_post_experiment(graph, post, targeted_p_maps, 4, "degree")
        for run in range(4):
            base = baseline_results[run].newly_reached_by_step
            protected = targeted_results[run].newly_reached_by_step
            assert sum(protected) <= sum(base)
        checked += 1
        if checked >= 4:
            break
    assert checked >= 1


def test_select_top_k_budget(graph):
    for method in ("degree", "betweenness", "pagerank", "random"):
        selected = select_top_k(graph, method, config.K_BUDGET, seed=1)
        assert len(selected) == config.K_BUDGET
        assert len(set(selected)) == config.K_BUDGET
        assert all(node in graph.nodes for node in selected)


def test_random_k_deterministic(graph):
    first = select_top_k(graph, "random", config.K_BUDGET, seed=7)
    second = select_top_k(graph, "random", config.K_BUDGET, seed=7)
    assert first == second
