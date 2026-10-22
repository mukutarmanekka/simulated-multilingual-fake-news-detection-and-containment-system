"""End-to-end detection -> containment -> measurement on a tiny graph.

Uses the real adapter's response mapping with deterministic fake detector
responses. No external API calls.
"""

import json

import networkx as nx
import pytest

from verifacto.detector.base import (
    VERDICT_FAKE,
    VERDICT_REAL,
    VERDICT_UNCERTAIN,
    map_response_dict,
)
from verifacto.data.posts import Post
from verifacto.evaluation.metrics import (
    build_summary_table,
    detector_summary,
    false_positive_cost,
)
from verifacto.simulation.experiment import (
    build_strategy_p_maps,
    run_post_experiment,
)
from verifacto.simulation.intervention import (
    run_detection,
    select_top_k,
)


def _graph():
    G = nx.DiGraph()
    G.add_nodes_from(range(6))
    for u, v in [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5)]:
        G.add_edge(u, v, p=1.0)
    return G


def _posts():
    return [
        Post(id=0, text="Fake claim.", true_label="fake", author=0, language="en"),
        Post(id=1, text="Real claim.", true_label="real", author=0, language="en"),
        Post(id=2, text="Uncertain claim.", true_label="real", author=0, language="en"),
        Post(
            id=3,
            text="Satire claim.",
            true_label="fake",
            author=0,
            language="en",
            source_url="https://www.theonion.com/some-article",
        ),
    ]


def _response(verdict, confidence, reason, **extra):
    claim = {
        "claim": "claim",
        "verdict": verdict,
        "confidence": confidence,
        "verdict_reason": reason,
        "verdict_support_score": 0.9 if verdict != VERDICT_UNCERTAIN else 0.3,
        "evidence_verdict": (
            "REFUTED"
            if verdict == VERDICT_FAKE
            else "SUPPORTED"
            if verdict == VERDICT_REAL
            else "NOT_ENOUGH_EVIDENCE"
        ),
        "evidence": [],
    }
    claim.update(extra)
    return {"results": [claim]}


class _FakeAsyncAdapter:
    def __init__(self, responses):
        self.responses = responses
        self.calls = []
        self.provider = "deepseek"

    def active_provider(self):
        return "deepseek"

    async def detect_async(self, text, language, post_id, source_url=None):
        self.calls.append((text, language, post_id, source_url))
        return map_response_dict(
            self.responses[post_id],
            post_id=post_id,
            text=text,
            language=language,
            source_url=source_url,
            provider="deepseek",
            detector_mode="real",
            latency_ms=12.5,
            provider_usage={
                "llm": 4,
                "search": 4,
                "factcheck": 1,
                "news": 0,
                "content": 8,
            },
        )


def _adapter():
    return _FakeAsyncAdapter(
        {
            0: _response(VERDICT_FAKE, 0.8, "EVIDENCE_CONTRADICTS_CLAIM"),
            1: _response(VERDICT_REAL, 0.8, "EVIDENCE_CORROBORATES_CLAIM"),
            2: _response(VERDICT_UNCERTAIN, 0.5, "SUBCLAIM_CONFLICT"),
            3: _response(
                VERDICT_FAKE,
                1.0,
                "BLACKLISTED_SATIRE_SOURCE",
                matched_blacklist_domain="theonion.com",
                blacklist_category="satire",
                source_tier="BLACKLISTED",
                source_tier_number=4,
            ),
        }
    )


def test_three_verdicts_flow_through_detection_and_containment():
    G = _graph()
    posts = _posts()
    adapter = _adapter()

    detections = run_detection(adapter, posts)

    assert detections[0].verdict == VERDICT_FAKE
    assert detections[0].is_fake is True
    assert detections[1].verdict == VERDICT_REAL
    assert detections[1].is_fake is False
    assert detections[2].verdict == VERDICT_UNCERTAIN
    assert detections[2].is_fake is None
    assert detections[3].verdict == VERDICT_FAKE
    assert detections[3].blacklisted is True
    assert detections[3].reason == "BLACKLISTED_SATIRE_SOURCE"

    for post in posts:
        assert detections[post.id].post_id == post.id

    # Ground-truth labels never enter the detector call.
    assert len(adapter.calls) == len(posts)
    assert adapter.calls[0] == (
        posts[0].text,
        posts[0].language,
        posts[0].id,
        posts[0].source_url,
    )
    assert all(len(call) == 4 for call in adapter.calls)

    summary = detector_summary(posts, detections)
    assert summary["verdict_counts"]["REAL"] == 1
    assert summary["verdict_counts"]["FAKE"] == 2
    assert summary["verdict_counts"]["UNCERTAIN"] == 1
    assert summary["blacklist_count"] == 1
    assert summary["uncertain_count"] == 1
    assert summary["false_positive_rate"] == 0.0
    assert summary["dangerous_false_negative_rate"] == 0.0


def test_containment_acts_on_fake_but_not_real_or_uncertain():
    G = _graph()
    posts = _posts()
    detections = run_detection(_adapter(), posts)
    n_runs = 3

    fake_maps = build_strategy_p_maps(
        G, detections[0], "blanket", n_runs
    )
    assert all(p_map is not None for p_map in fake_maps.values())

    for post_id in (1, 2):
        maps = build_strategy_p_maps(
            G, detections[post_id], "blanket", n_runs
        )
        assert all(p_map is None for p_map in maps.values())

    selected = select_top_k(G, "degree", 2)
    targeted_fake = build_strategy_p_maps(
        G, detections[0], "degree", n_runs, k=2, selected=selected
    )
    assert all(p_map is not None for p_map in targeted_fake.values())
    targeted_real = build_strategy_p_maps(
        G, detections[1], "degree", n_runs, k=2, selected=selected
    )
    assert all(p_map is None for p_map in targeted_real.values())

    baseline_results = {}
    blanket_results = {}
    for post in posts:
        baseline_maps = build_strategy_p_maps(
            G, detections[post.id], "baseline", n_runs
        )
        blanket_maps = build_strategy_p_maps(
            G, detections[post.id], "blanket", n_runs
        )
        baseline_results[post.id] = run_post_experiment(
            G, post, baseline_maps, n_runs, "baseline"
        )
        blanket_results[post.id] = run_post_experiment(
            G, post, blanket_maps, n_runs, "blanket"
        )

    for run in range(n_runs):
        baseline = baseline_results[0][run].total_reached
        protected = blanket_results[0][run].total_reached
        assert protected <= baseline

    table = build_summary_table(
        G,
        posts,
        detections,
        {"baseline": baseline_results, "blanket": blanket_results},
        n_runs,
        {"baseline": 0, "blanket": 0},
    )
    # Both flagged fake posts (0 and 3) are acted on; real/uncertain are not.
    assert table["blanket"]["interventions"]["posts_acted_on"] == 2

    cost = false_positive_cost(
        posts,
        detections,
        baseline_results,
        blanket_results,
        n_runs,
    )
    assert cost["fp_count"] == 0
    assert cost["uncertain_count"] == 1
    assert cost["error_count"] == 0
    assert cost["blacklist_count"] == 1


def test_detection_failure_is_recorded_not_a_verdict():
    G = _graph()
    posts = _posts()[:2]

    class FailingAdapter:
        provider = "deepseek"

        def active_provider(self):
            return "deepseek"

        async def detect_async(self, text, language, post_id, source_url=None):
            if post_id == 1:
                raise RuntimeError("provider failed")
            return map_response_dict(
                _response(VERDICT_FAKE, 0.8, "EVIDENCE_CONTRADICTS_CLAIM"),
                post_id=post_id,
                text=text,
                language=language,
                source_url=source_url,
                provider="deepseek",
                detector_mode="real",
            )

    detections = run_detection(FailingAdapter(), posts, error_policy="record")

    assert detections[0].verdict == VERDICT_FAKE
    assert detections[1].status == "error"
    assert detections[1].verdict is None
    assert detections[1].is_fake is None
    assert "provider failed" in detections[1].error

    summary = detector_summary(posts, detections)
    assert summary["error_count"] == 1
    assert summary["uncertain_count"] == 0
    assert summary["verdict_counts"]["FAKE"] == 1

    # No intervention is applied for the failed post.
    maps = build_strategy_p_maps(G, detections[1], "blanket", 2)
    assert all(p_map is None for p_map in maps.values())


def test_persistence_payload_has_detections_and_mode(tmp_path):
    posts = _posts()
    detections = run_detection(_adapter(), posts)
    payload = {
        "meta": {"detector": "real", "detector_mode": "real"},
        "detections": {
            str(post.id): detections[post.id].to_dict() for post in posts
        },
    }
    path = tmp_path / "experiment.json"
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    loaded = json.loads(path.read_text(encoding="utf-8"))

    assert loaded["meta"]["detector_mode"] == "real"
    assert loaded["detections"]["0"]["verdict"] == "FAKE"
    assert loaded["detections"]["1"]["verdict"] == "REAL"
    assert loaded["detections"]["2"]["verdict"] == "UNCERTAIN"
    assert loaded["detections"]["3"]["blacklisted"] is True
    assert "evidence" in loaded["detections"]["0"]
    assert "provider_usage" in loaded["detections"]["0"]
