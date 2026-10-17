"""Operational definitions of the evaluation metrics.

All metrics are computed from per-run series `s = newly_reached_by_step`
(the cascade's counts per time step) and the post corpus. Ground-truth
labels are used HERE for the false-positive accounting only.
"""

import numpy as np


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


def interventions_used(posts, detections, users_throttled):
    """Count of interventions: posts acted on and users throttled."""
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
    reach_lost = []
    n_real = 0
    for post in posts:
        detection = detections[post.id]
        if detection.is_fake is None:
            uncertain_count += 1
        if post.true_label != "real":
            continue
        n_real += 1
        if detection.is_fake is True:
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
                posts, detections, users_throttled.get(strategy, 0)
            ),
        }
    return table
