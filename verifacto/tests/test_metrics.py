"""Phase 5: metric definitions are exact on hand-built inputs."""

from verifacto.data.posts import Post
from verifacto.detector.base import DetectionResult
from verifacto.evaluation.metrics import (
    aggregate_over_runs,
    false_positive_cost,
    final_fraction_reached,
    interventions_used,
    peak_spread,
    time_to_containment,
)
from verifacto.simulation.experiment import PostRunResult


def test_final_fraction_reached():
    assert final_fraction_reached([1, 2, 3], 10) == 0.6


def test_peak_spread():
    assert peak_spread([1, 4, 2, 3]) == (4, 1)
    assert peak_spread([1, 2, 2]) == (2, 1)
    assert peak_spread([1]) == (0, 0)


def test_time_to_containment():
    assert time_to_containment([1, 2, 3]) == 2
    assert time_to_containment([1]) == 0


def test_aggregate_over_runs():
    stats = aggregate_over_runs([2.0, 4.0])
    assert stats["mean"] == 3.0
    assert stats["std"] == 1.0
    assert aggregate_over_runs([]) == {"mean": 0.0, "std": 0.0}


def _post(pid, label):
    return Post(id=pid, text="t", true_label=label, author=0, language="en")


def _fake():
    return DetectionResult(is_fake=True, confidence=0.8, verdict_summary="REFUTED", provider="stub")


def _real():
    return DetectionResult(is_fake=False, confidence=0.8, verdict_summary="SUPPORTED", provider="stub")


def _uncertain():
    return DetectionResult(is_fake=None, confidence=0.0, verdict_summary="UNCERTAIN", provider="stub")


def _results(total):
    return [PostRunResult(post_id=0, run=0, strategy="x", newly_reached_by_step=[total])]


def test_interventions_used_counts_flagged_posts():
    posts = [_post(0, "fake"), _post(1, "real"), _post(2, "fake")]
    detections = {0: _fake(), 1: _real(), 2: _real()}
    used = interventions_used(posts, detections, users_throttled=25)
    assert used == {"posts_acted_on": 1, "users_throttled": 25}


def test_false_positive_cost_only_real_posts():
    posts = [
        _post(0, "fake"),
        _post(1, "real"),
        _post(2, "real"),
        _post(3, "real"),
    ]
    detections = {0: _fake(), 1: _fake(), 2: _real(), 3: _uncertain()}
    baseline = {1: _results(100), 2: _results(100), 3: _results(100)}
    strategy = {1: _results(60), 2: _results(100), 3: _results(100)}
    cost = false_positive_cost(posts, detections, baseline, strategy, n_runs=1)
    assert cost["fp_count"] == 1
    assert cost["fp_fraction"] == 1 / 3
    assert cost["real_reach_lost"]["mean"] == 40.0
    assert cost["uncertain_count"] == 1
