"""Opt-in bounded live integration of the simulation's real adapter.

Run with:
    RUN_VERIFACTO_LIVE=1 pytest verifacto/tests/test_live_detector_integration.py -s

Makes at most three detections. The satire case must exit before any
provider call. The other verdicts are reported, not forced: live evidence
may legitimately return REAL, FAKE, or UNCERTAIN.
"""

import os

import networkx as nx
import pytest

from verifacto.data.posts import Post
from verifacto.detector.base import DetectorUnavailableError
from verifacto.detector.real_adapter import RealDetectorAdapter
from verifacto.evaluation.metrics import build_summary_table, detector_summary
from verifacto.simulation.experiment import (
    build_strategy_p_maps,
    run_post_experiment,
)
from verifacto.simulation.intervention import run_detection

pytestmark = pytest.mark.skipif(
    os.environ.get("RUN_VERIFACTO_LIVE") != "1",
    reason="Set RUN_VERIFACTO_LIVE=1 to run live detector integration",
)


def _posts():
    return [
        Post(
            id=9001,
            text="The Colosseum is located in Rome, Italy.",
            true_label="real",
            author=0,
            language="en",
            source_url="https://www.britannica.com/topic/Colosseum",
        ),
        Post(
            id=9002,
            text="The Great Barrier Reef is located off the coast of Brazil.",
            true_label="fake",
            author=1,
            language="en",
            source_url="https://www.britannica.com/topic/Great-Barrier-Reef",
        ),
        Post(
            id=9003,
            text="The moon is made of cheese, according to a new study.",
            true_label="fake",
            author=2,
            language="en",
            source_url="https://www.theonion.com/some-article",
        ),
    ]


def _graph():
    G = nx.DiGraph()
    G.add_nodes_from(range(6))
    for u, v in [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5)]:
        G.add_edge(u, v, p=1.0)
    return G


def test_live_adapter_three_posts():
    adapter = RealDetectorAdapter(error_policy="record", timeout_seconds=300)
    try:
        provider = adapter.active_provider()
    except DetectorUnavailableError as exc:
        pytest.skip(f"real detector unavailable: {exc}")

    if provider == "mock":
        pytest.skip("LLM provider is mock; live integration NOT TESTED")

    posts = _posts()
    detections = run_detection(adapter, posts, error_policy="record")

    for post in posts:
        detection = detections[post.id]
        print(
            f"LIVE_VERIFACTO post={post.id} language={post.language} "
            f"verdict={detection.verdict} status={detection.status} "
            f"confidence={detection.confidence} "
            f"support={detection.support_score} "
            f"reason={detection.reason} "
            f"blacklisted={detection.blacklisted} "
            f"provider={detection.provider} "
            f"latency_ms={detection.latency_ms} "
            f"usage={detection.provider_usage} "
            f"provider_status={detection.provider_status} "
            f"error={detection.error}"
        )

    satire = detections[9003]
    assert satire.status == "ok"
    assert satire.verdict == "FAKE"
    assert satire.blacklisted is True
    assert satire.reason.startswith("BLACKLISTED_")
    assert satire.provider_usage["llm"] == 0
    assert satire.provider_usage["search"] == 0
    assert satire.provider_usage["factcheck"] == 0
    assert satire.provider_usage["news"] == 0
    assert satire.provider_usage["content"] == 0

    # The non-blacklisted claims must complete through the real adapter.
    for post_id in (9001, 9002):
        detection = detections[post_id]
        if detection.status in ("error", "timeout"):
            pytest.fail(
                f"live detection failed for post {post_id}: "
                f"{detection.error}"
            )
        assert detection.verdict in ("REAL", "FAKE", "UNCERTAIN")

    # Complete the loop on a tiny graph: detection verdicts -> containment
    # -> cascade -> aggregate metrics.
    G = _graph()
    n_runs = 2
    results = {"baseline": {}, "blanket": {}}
    for strategy in results:
        for post in posts:
            p_maps = build_strategy_p_maps(
                G, detections[post.id], strategy, n_runs
            )
            results[strategy][post.id] = run_post_experiment(
                G, post, p_maps, n_runs, strategy
            )

    summary = detector_summary(posts, detections)
    table = build_summary_table(
        G,
        posts,
        detections,
        results,
        n_runs,
        {"baseline": 0, "blanket": 0},
    )
    fake_baseline = [
        result.total_reached
        for post in posts
        if post.true_label == "fake"
        for result in results["baseline"][post.id]
    ]
    fake_blanket = [
        result.total_reached
        for post in posts
        if post.true_label == "fake"
        for result in results["blanket"][post.id]
    ]
    print(
        "LIVE_EXPERIMENT "
        f"fake_reach_baseline={sum(fake_baseline) / len(fake_baseline):.2f} "
        f"fake_reach_blanket={sum(fake_blanket) / len(fake_blanket):.2f} "
        f"posts_acted_on={table['blanket']['interventions']['posts_acted_on']} "
        f"verdicts={summary['verdict_counts']} "
        f"uncertain={summary['uncertain_count']} "
        f"errors={summary['error_count']} "
        f"blacklisted={summary['blacklist_count']}"
    )

    assert satirical_post_acted_on(detections, G, n_runs)
    assert table["blanket"]["interventions"]["posts_acted_on"] >= 1


def satirical_post_acted_on(detections, G, n_runs):
    p_maps = build_strategy_p_maps(
        G, detections[9003], "blanket", n_runs
    )
    return all(p_map is not None for p_map in p_maps.values())
