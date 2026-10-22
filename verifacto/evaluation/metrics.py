"""Operational definitions of the evaluation metrics.

All metrics are computed from per-run series `s = newly_reached_by_step`
(the cascade's counts per time step) and the post corpus. Ground-truth
labels are used HERE for the false-positive accounting only.
"""

import numpy as np

VERDICT_BUCKETS = ("REAL", "FAKE", "UNCERTAIN", "NO_CLAIMS", "ERROR")


def _verdict_bucket(detection):
    if detection.status in ("error", "timeout"):
        return "ERROR"
    if detection.status == "no_claims":
        return "NO_CLAIMS"
    if detection.verdict in ("REAL", "FAKE", "UNCERTAIN"):
        return detection.verdict
    return "ERROR"


def _mean(values):
    if not values:
        return 0.0
    return float(np.mean(values))


def detector_summary(posts, detections):
    """Detector quality, uncertainty, failures, and provider usage.

    Ground-truth labels are used HERE for evaluation only. They never
    influence the detector or containment decisions.
    """
    verdict_counts = {bucket: 0 for bucket in VERDICT_BUCKETS}
    status_counts = {}
    confusion = {
        truth: {bucket: 0 for bucket in VERDICT_BUCKETS}
        for truth in ("real", "fake")
    }
    blacklist_count = 0
    confidence_values = []
    support_values = []
    latency_values = []
    provider_usage = {}

    for post in posts:
        detection = detections[post.id]
        bucket = _verdict_bucket(detection)
        verdict_counts[bucket] += 1

        status = detection.status or "unknown"
        status_counts[status] = status_counts.get(status, 0) + 1

        if detection.blacklisted:
            blacklist_count += 1
        if detection.status == "ok":
            confidence_values.append(detection.confidence)
            if detection.support_score is not None:
                support_values.append(detection.support_score)
        if detection.latency_ms is not None:
            latency_values.append(detection.latency_ms)
        if detection.provider_usage:
            for key, value in detection.provider_usage.items():
                if isinstance(value, (int, float)):
                    provider_usage[key] = provider_usage.get(key, 0) + value

        truth = getattr(post, "true_label", None)
        if truth in confusion:
            confusion[truth][bucket] += 1

    total = len(posts)
    n_real = sum(1 for post in posts if getattr(post, "true_label", None) == "real")
    n_fake = sum(1 for post in posts if getattr(post, "true_label", None) == "fake")
    successful = status_counts.get("ok", 0) + status_counts.get("no_claims", 0)

    real_fp = confusion["real"]["FAKE"]
    fake_real = confusion["fake"]["REAL"]
    fake_uncertain = confusion["fake"]["UNCERTAIN"]
    fake_not_flagged = sum(
        confusion["fake"][bucket]
        for bucket in ("REAL", "UNCERTAIN", "NO_CLAIMS", "ERROR")
    )

    cache_verified = sum(
        1 for detection in detections.values()
        if detection.cache_key_verified is True
    )
    cache_unverified = sum(
        1 for detection in detections.values()
        if detection.cache_key_verified is False
    )

    return {
        "n_posts": total,
        "verdict_counts": verdict_counts,
        "status_counts": status_counts,
        "blacklist_count": blacklist_count,
        "uncertain_count": verdict_counts["UNCERTAIN"],
        "no_claims_count": verdict_counts["NO_CLAIMS"],
        "error_count": verdict_counts["ERROR"],
        "uncertain_fraction": (
            verdict_counts["UNCERTAIN"] / total if total else 0.0
        ),
        "uncertain_share_of_successful": (
            verdict_counts["UNCERTAIN"] / successful if successful else 0.0
        ),
        "no_claims_fraction": (
            verdict_counts["NO_CLAIMS"] / total if total else 0.0
        ),
        "error_fraction": (
            verdict_counts["ERROR"] / total if total else 0.0
        ),
        "detection_coverage": successful / total if total else 0.0,
        "mean_confidence": _mean(confidence_values),
        "mean_support_score": _mean(support_values),
        "mean_latency_ms": _mean(latency_values),
        "provider_usage": provider_usage,
        "replay_cache_verified": cache_verified,
        "replay_cache_unverified": cache_unverified,
        "ground_truth_confusion": confusion,
        "false_positive_rate": real_fp / n_real if n_real else 0.0,
        "dangerous_false_negative_rate": (
            fake_real / n_fake if n_fake else 0.0
        ),
        "fake_uncertain_rate": (
            fake_uncertain / n_fake if n_fake else 0.0
        ),
        "fake_not_flagged_rate": (
            fake_not_flagged / n_fake if n_fake else 0.0
        ),
    }


def final_fraction_reached(series, n_users):
    """Final fraction of users reached: total reached / users."""
    return sum(series) / n_users


def peak_spread(series):
    """(peak new infections per step, first step achieving it).

    Step 0 (the author alone) is excluded — the peak describes the
    growth phase.
    """
    if len(series) <= 1:
        return (0, 0)
    peak = max(series[1:])
    step = series.index(peak, 1)
    return (peak, step)


def time_to_containment(series):
    """Cascade duration in steps: the first step where infections hit 0.

    An intervention that stops spread earlier has a smaller value than
    the baseline.
    """
    return len(series) - 1


def interventions_used(posts, detections, users_throttled, active=True):
    """Count of interventions: posts acted on and users throttled.

    The baseline strategy takes no action at all (active=False), even
    though the detector may have flagged posts.
    """
    if not active:
        return {"posts_acted_on": 0, "users_throttled": 0}
    posts_acted_on = 0
    for post in posts:
        if detections[post.id].is_fake is True:
            posts_acted_on += 1
    return {"posts_acted_on": posts_acted_on, "users_throttled": users_throttled}


def aggregate_over_runs(values):
    """Mean and standard deviation over a list of scalar values."""
    if not values:
        return {"mean": 0.0, "std": 0.0}
    return {"mean": float(np.mean(values)), "std": float(np.std(values))}


def false_positive_cost(posts, detections, baseline_results, strategy_results, n_runs):
    """Cost of wrongly suppressing real news under one strategy.

    baseline_results / strategy_results map post_id to a list of
    PostRunResult (one per run). Only real posts wrongly flagged fake
    contribute to the cost; uncertain detections are counted separately
    and never treated as false positives.
    """
    fp_count = 0
    uncertain_count = 0
    error_count = 0
    blacklist_count = 0
    reach_lost = []
    n_real = 0
    for post in posts:
        detection = detections[post.id]
        if detection.blacklisted:
            blacklist_count += 1
        if detection.status in ("error", "timeout"):
            error_count += 1
        elif detection.status == "no_claims" or detection.verdict == "UNCERTAIN":
            uncertain_count += 1
        if post.true_label != "real":
            continue
        n_real += 1
        if detection.status == "ok" and detection.verdict == "FAKE":
            fp_count += 1
            for run in range(n_runs):
                base = baseline_results[post.id][run].total_reached
                protected = strategy_results[post.id][run].total_reached
                reach_lost.append(base - protected)
    return {
        "fp_count": fp_count,
        "fp_fraction": fp_count / n_real if n_real > 0 else 0.0,
        "real_reach_lost": aggregate_over_runs(reach_lost),
        "uncertain_count": uncertain_count,
        "error_count": error_count,
        "blacklist_count": blacklist_count,
    }


def _fake_metric_values(posts, results_by_post, metric):
    """Metric over every (fake post, run) pair for one strategy."""
    values = []
    for post in posts:
        if post.true_label != "fake":
            continue
        for result in results_by_post[post.id]:
            values.append(metric(result.newly_reached_by_step))
    return values


def build_summary_table(G, posts, detections, results, n_runs, users_throttled):
    """Per-strategy summary consumed by the CLI table and metrics.json.

    `results` maps strategy name -> {post_id: [PostRunResult]}. The
    "baseline" strategy must be present: every other strategy's reduction
    is reported relative to it.
    """
    n_users = G.number_of_nodes()
    baseline_final = aggregate_over_runs(
        _fake_metric_values(
            posts, results["baseline"], lambda s: final_fraction_reached(s, n_users)
        )
    )
    table = {}
    for strategy, results_by_post in results.items():
        final = aggregate_over_runs(
            _fake_metric_values(
                posts, results_by_post, lambda s: final_fraction_reached(s, n_users)
            )
        )
        peak = aggregate_over_runs(
            _fake_metric_values(posts, results_by_post, lambda s: peak_spread(s)[0])
        )
        peak_step = aggregate_over_runs(
            _fake_metric_values(posts, results_by_post, lambda s: peak_spread(s)[1])
        )
        containment = aggregate_over_runs(
            _fake_metric_values(posts, results_by_post, time_to_containment)
        )
        relative_reduction = 0.0
        if baseline_final["mean"] > 0:
            relative_reduction = 1.0 - final["mean"] / baseline_final["mean"]
        table[strategy] = {
            "final_fraction_reached": final,
            "peak_spread": peak,
            "peak_spread_step": peak_step,
            "time_to_containment": containment,
            "relative_reduction_vs_baseline": relative_reduction,
            "interventions": interventions_used(
                posts,
                detections,
                users_throttled.get(strategy, 0),
                active=(strategy != "baseline"),
            ),
        }
    return table
