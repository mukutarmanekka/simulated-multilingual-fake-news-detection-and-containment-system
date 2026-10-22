"""VeriFacto containment layer — command line driver.

Subcommands are added phase by phase:
    graph      build and save the language-block network (Phase 1)
    posts      generate and save the post corpus (Phase 2)
    baseline   Independent Cascade with no protection (Phase 3)
    blanket    detector + blanket halving of flagged fakes (Phase 4)
    precompute run the real detector over the corpus, cache verdicts (Phase 4)
    targeted   top-k targeted containment, full evaluation (Phase 5)
"""

import argparse
import json
import sys
from pathlib import Path

# The detector logs multilingual claim text. Windows consoles default to a
# legacy code page, which can turn a successful provider response into a
# UnicodeEncodeError while logging. Configure UTF-8 before importing/running
# any detector or simulation service.
if sys.platform == "win32":
    for _stream in (sys.stdout, sys.stderr):
        try:
            _stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

import numpy as np

from . import config
from .containment.bridges import detect_bridges, print_bridge_summary
from .containment.greedy import greedy_select
from .containment.heuristics import precompute_centralities, top_k_from_scores
from .containment.priority import rank_flagged_posts
from .data.posts import generate_posts, load_posts, print_posts_summary, save_posts
from .detector.replay import ReplayDetector
from .detector.real_adapter import RealDetectorAdapter
from .detector.stub import StubDetector
from .evaluation.metrics import (
    build_summary_table,
    detector_summary,
    false_positive_cost,
)
from .evaluation.plots import plot_fp_cost_bar, plot_reach_curves, plot_sweep_lines
from .network.graph_builder import (
    build_language_graph,
    load_graph,
    print_graph_summary,
    save_graph,
)
from .simulation.cascade import run_ic_cascade
from .simulation.experiment import build_strategy_p_maps, run_post_experiment
from .simulation.intervention import run_detection, select_top_k

STRATEGY_ORDER = ["baseline", "blanket", "random", "degree", "betweenness", "pagerank"]

# strategies shared by the compare and sweep commands (greedy is optional)
ALL_STRATEGIES = ["baseline", "blanket", "random", "degree", "betweenness", "pagerank"]


def _load_or_build_graph():
    if config.GRAPH_PATH.exists():
        return load_graph()
    return build_language_graph()


def _load_or_generate_posts(G):
    if config.POSTS_PATH.exists():
        return load_posts()
    return generate_posts(G)


def cmd_graph(args) -> None:
    G = build_language_graph()
    print_graph_summary(G)
    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    save_graph(G)
    print(f"graph saved to {config.GRAPH_PATH}")


def cmd_posts(args) -> None:
    G = build_language_graph()
    posts = generate_posts(G)
    print_posts_summary(posts)
    save_posts(posts)
    print(f"posts saved to {config.POSTS_PATH}")


def cmd_baseline(args) -> None:
    G = _load_or_build_graph()
    posts = _load_or_generate_posts(G)
    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # the no-protection baseline: same (post, run) seeds every strategy
    # will reuse in later phases
    all_series = {}
    for post in posts:
        post_series = []
        for run in range(config.N_RUNS):
            seed = config.cascade_seed(post.id, run)
            post_series.append(run_ic_cascade(G, post.author, seed=seed))
        all_series[post.id] = post_series

    payload = {
        "meta": {
            "n_runs": config.N_RUNS,
            "strategy": "baseline",
            "detector": "none",
            "detector_mode": "none",
        },
        "series": all_series,
    }
    with open(config.OUTPUT_DIR / "baseline_series.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    # metrics for the plot use fake posts only; true_label never touches
    # the simulation itself
    fake_series = [
        series
        for post in posts
        if post.true_label == "fake"
        for series in all_series[post.id]
    ]
    plot_reach_curves(
        [{"label": "baseline (no protection)", "series": fake_series}],
        config.OUTPUT_DIR / "baseline_reach.png",
        "Fake posts: reach over time with no protection",
    )
    mean_final = float(np.mean([sum(s) for s in fake_series])) / G.number_of_nodes()
    print(f"mean final fraction reached by fake posts (baseline): {mean_final:.3f}")
    print(f"plot saved to {config.OUTPUT_DIR / 'baseline_reach.png'}")


def cmd_precompute(args) -> None:
    from .detector import precompute

    posts_path = Path(args.posts_path) if args.posts_path else config.POSTS_PATH
    raise SystemExit(
        precompute.main(
            posts_path=posts_path,
            pacing=args.pacing,
            max_posts=args.max_posts,
            timeout_seconds=args.detector_timeout,
            recompute_stale=args.recompute_stale,
        )
    )


def cmd_blanket(args) -> None:
    G = _load_or_build_graph()
    posts = _limit_posts(args, _load_or_generate_posts(G))
    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    detector = _build_detector(args, posts)
    print(f"detector: {args.detector}")
    detections = _run_detections(args, detector, posts)
    summary = detector_summary(posts, detections)

    # run both strategies on every post with shared (post, run) seeds
    baseline_results = {}
    blanket_results = {}
    for post in posts:
        detection = detections[post.id]
        baseline_p_maps = build_strategy_p_maps(G, detection, "baseline", config.N_RUNS)
        blanket_p_maps = build_strategy_p_maps(G, detection, "blanket", config.N_RUNS)
        baseline_results[post.id] = run_post_experiment(
            G, post, baseline_p_maps, config.N_RUNS, "baseline"
        )
        blanket_results[post.id] = run_post_experiment(
            G, post, blanket_p_maps, config.N_RUNS, "blanket"
        )

    fake_baseline = [
        result.newly_reached_by_step
        for post in posts
        if post.true_label == "fake"
        for result in baseline_results[post.id]
    ]
    fake_blanket = [
        result.newly_reached_by_step
        for post in posts
        if post.true_label == "fake"
        for result in blanket_results[post.id]
    ]
    plot_reach_curves(
        [
            {"label": "baseline (no protection)", "series": fake_baseline},
            {"label": "blanket halving (flagged)", "series": fake_blanket},
        ],
        config.OUTPUT_DIR / "blanket_vs_baseline.png",
        "Fake posts: reach with and without detection",
    )

    cost = false_positive_cost(
        posts,
        detections,
        baseline_results,
        blanket_results,
        config.N_RUNS,
    )
    n_real = sum(1 for p in posts if p.true_label == "real")
    print(f"fake posts: mean final fraction (baseline) = "
          f"{np.mean([sum(r) for r in fake_baseline]) / G.number_of_nodes():.3f}")
    print(f"fake posts: mean final fraction (blanket)  = "
          f"{np.mean([sum(r) for r in fake_blanket]) / G.number_of_nodes():.3f}")
    print(f"false positives: {cost['fp_count']}/{n_real} real posts flagged fake")
    print(
        f"real-news reach lost per run (mean): "
        f"{cost['real_reach_lost']['mean']:.1f} users"
    )
    print(
        f"uncertain detections: {cost['uncertain_count']} | "
        f"detector errors: {cost['error_count']} | "
        f"blacklisted: {cost['blacklist_count']}"
    )
    _print_detector_summary(summary)

    payload = {
        "meta": _detector_meta(
            args,
            posts,
            n_runs=config.N_RUNS,
            strategy="blanket",
        ),
        "detector_summary": summary,
        "detections": _detection_payload(posts, detections),
        "fp_cost": cost,
        "fake_reach": {
            "baseline_mean_final_fraction": float(
                np.mean([sum(r) for r in fake_baseline])
            ) / G.number_of_nodes(),
            "blanket_mean_final_fraction": float(
                np.mean([sum(r) for r in fake_blanket])
            ) / G.number_of_nodes(),
        },
    }
    _write_output(
        config.OUTPUT_DIR / "blanket_metrics.json",
        payload,
        args.detector,
    )
    print(f"plot saved to {config.OUTPUT_DIR / 'blanket_vs_baseline.png'}")


def _build_detector(args, posts):
    if args.detector == "stub":
        return StubDetector(posts)
    if args.detector == "replay":
        if not config.CACHE_PATH.exists():
            print("no detector cache found — run `python -m verifacto.main precompute` first")
            raise SystemExit(1)
        return ReplayDetector(
            expected_provider=getattr(args, "cache_provider", None),
            expected_model=getattr(args, "cache_model", None),
        )
    if args.detector == "real":
        if not args.allow_live:
            print("live detection is slow and non-deterministic; pass --allow-live to proceed")
            raise SystemExit(1)
        return RealDetectorAdapter(
            timeout_seconds=getattr(
                args, "detector_timeout", config.DETECTOR_TIMEOUT_SECONDS
            ),
            error_policy=getattr(
                args, "detector_error_policy", config.DETECTOR_ERROR_POLICY
            ),
        )
    raise ValueError(f"unknown detector: {args.detector}")


def _limit_posts(args, posts):
    """Apply the live-call bound; real mode defaults to a small subset."""
    limit = getattr(args, "max_posts", None)
    if limit is None and getattr(args, "detector", "stub") == "real":
        limit = config.DEFAULT_LIVE_MAX_POSTS
    if limit is None:
        return posts
    limited = posts[:limit]
    print(
        f"posts: {len(limited)}/{len(posts)} selected "
        f"(max_posts={limit})"
    )
    return limited


def _run_detections(args, detector, posts):
    return run_detection(
        detector,
        posts,
        error_policy=getattr(
            args, "detector_error_policy", config.DETECTOR_ERROR_POLICY
        ),
    )


def _detection_payload(posts, detections):
    return {
        str(post.id): detections[post.id].to_dict()
        for post in posts
    }


def _detector_meta(args, posts, **extra):
    meta = {
        "detector": getattr(args, "detector", "stub"),
        "detector_mode": getattr(args, "detector", "stub"),
        "n_posts": len(posts),
        "detector_error_policy": getattr(
            args, "detector_error_policy", config.DETECTOR_ERROR_POLICY
        ),
        "uncertainty_policy": config.UNCERTAIN_POLICY,
        "seed": config.SEED_MASTER,
    }
    meta.update(extra)
    return meta


def _write_output(path, payload, detector_mode):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    mode_path = path.with_name(f"{path.stem}_{detector_mode}{path.suffix}")
    with open(mode_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    print(f"saved {path} and {mode_path}")


def _print_detector_summary(summary):
    print("\n=== detector summary ===")
    print(f"verdicts: {summary['verdict_counts']}")
    print(f"status: {summary['status_counts']}")
    print(
        f"blacklisted: {summary['blacklist_count']} | "
        f"uncertain: {summary['uncertain_count']} "
        f"({summary['uncertain_fraction']:.1%}) | "
        f"no claims: {summary['no_claims_count']} | "
        f"errors: {summary['error_count']} "
        f"({summary['error_fraction']:.1%})"
    )
    print(
        f"mean confidence: {summary['mean_confidence']:.3f} | "
        f"mean support: {summary['mean_support_score']:.3f} | "
        f"mean latency: {summary['mean_latency_ms']:.0f} ms"
    )
    print(
        f"false-positive rate: {summary['false_positive_rate']:.1%} | "
        f"dangerous false-negative rate: "
        f"{summary['dangerous_false_negative_rate']:.1%} | "
        f"fake not flagged: {summary['fake_not_flagged_rate']:.1%}"
    )
    if summary["provider_usage"]:
        print(f"provider usage: {summary['provider_usage']}")
    if config.UNCERTAIN_POLICY == "review":
        print(
            f"uncertain review queue: {summary['uncertain_count']} posts "
            "(no diffusion action)"
        )


def cmd_targeted(args) -> None:
    G = _load_or_build_graph()
    posts = _limit_posts(args, _load_or_generate_posts(G))
    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    n_runs = args.runs
    k = args.k

    detector = _build_detector(args, posts)
    print(f"detector: {args.detector}")
    detections = _run_detections(args, detector, posts)
    summary = detector_summary(posts, detections)
    _print_detector_summary(summary)

    # centralities computed once on the original graph, shared by all posts
    centralities = precompute_centralities(G)
    selected_by_strategy = {
        method: top_k_from_scores(centralities[method], k)
        for method in ("degree", "betweenness", "pagerank")
    }

    results = {}
    for strategy in STRATEGY_ORDER:
        results[strategy] = {}
        for post in posts:
            p_maps = build_strategy_p_maps(
                G,
                detections[post.id],
                strategy,
                n_runs,
                k=k,
                selected=selected_by_strategy.get(strategy),
            )
            results[strategy][post.id] = run_post_experiment(
                G, post, p_maps, n_runs, strategy
            )

    users_throttled = {
        strategy: (k if strategy in ("random", "degree", "betweenness", "pagerank") else 0)
        for strategy in STRATEGY_ORDER
    }
    table = build_summary_table(G, posts, detections, results, n_runs, users_throttled)

    print("\n=== strategy comparison (fake posts, mean +/- std over runs) ===")
    for strategy in STRATEGY_ORDER:
        row = table[strategy]
        print(
            f"{strategy:>11}: final fraction {row['final_fraction_reached']['mean']:.3f} "
            f"+/- {row['final_fraction_reached']['std']:.3f} | peak "
            f"{row['peak_spread']['mean']:.0f} at step {row['peak_spread_step']['mean']:.0f} | "
            f"containment {row['time_to_containment']['mean']:.1f} steps | "
            f"reduction {row['relative_reduction_vs_baseline'] * 100:.0f}%"
        )

    print("\n=== false-positive cost ===")
    costs = []
    for strategy in STRATEGY_ORDER[1:]:
        cost = false_positive_cost(posts, detections, results["baseline"], results[strategy], n_runs)
        costs.append(
            {
                "label": strategy,
                "real_reach_lost_mean": cost["real_reach_lost"]["mean"],
                "fp_count": cost["fp_count"],
            }
        )
        print(
            f"{strategy:>11}: {cost['fp_count']} real posts flagged, "
            f"{cost['real_reach_lost']['mean']:.1f} users of real-news reach lost per run, "
            f"{cost['uncertain_count']} uncertain detections"
        )

    curves = []
    for strategy in STRATEGY_ORDER:
        curves.append(
            {
                "label": strategy,
                "series": [
                    result.newly_reached_by_step
                    for post in posts
                    if post.true_label == "fake"
                    for result in results[strategy][post.id]
                ],
            }
        )
    plot_reach_curves(
        curves,
        config.OUTPUT_DIR / "strategy_comparison.png",
        "Fake posts: spread over time per strategy (mean ± std)",
    )
    plot_fp_cost_bar(costs, config.OUTPUT_DIR / "fp_cost_bar.png")

    payload = {
        "meta": _detector_meta(args, posts, n_runs=n_runs, k=k),
        "detector_summary": summary,
        "detections": _detection_payload(posts, detections),
        "summary": table,
        "fp_cost": costs,
    }
    _write_output(
        config.OUTPUT_DIR / "metrics.json",
        payload,
        args.detector,
    )

    print(f"\ncharts and metrics saved to {config.OUTPUT_DIR}")


def _run_strategy(G, posts, detections, strategy, n_runs, k, selected):
    """Run one strategy over all posts; returns {post_id: [PostRunResult]}."""
    results_by_post = {}
    for post in posts:
        p_maps = build_strategy_p_maps(
            G, detections[post.id], strategy, n_runs, k=k, selected=selected
        )
        results_by_post[post.id] = run_post_experiment(
            G, post, p_maps, n_runs, strategy
        )
    return results_by_post


def _run_budget_block(
    G, posts, detections, centralities, n_runs, k, shared,
    greedy_pool, greedy_runs, with_greedy,
):
    """Run every strategy at budget k; returns (table_rows, chart_curves)."""
    strategies = list(ALL_STRATEGIES)
    results = dict(shared)
    selected_by_strategy = {
        method: top_k_from_scores(centralities[method], k)
        for method in ("degree", "betweenness", "pagerank")
    }
    if with_greedy:
        pool = top_k_from_scores(centralities["betweenness"], greedy_pool)
        print(
            f"greedy selection for k={k} "
            f"(pool {len(pool)}, {greedy_runs} selection runs) ..."
        )
        selected_by_strategy["greedy"] = greedy_select(
            G, posts, detections, k, greedy_runs, pool
        )
        strategies.append("greedy")

    for strategy in ("random", "degree", "betweenness", "pagerank"):
        results[strategy] = _run_strategy(
            G, posts, detections, strategy, n_runs, k=k,
            selected=selected_by_strategy.get(strategy),
        )
    if "greedy" in selected_by_strategy:
        results["greedy"] = _run_strategy(
            G, posts, detections, "greedy", n_runs, k=k,
            selected=selected_by_strategy["greedy"],
        )

    users_throttled = {"baseline": 0, "blanket": 0}
    for strategy in ("random", "degree", "betweenness", "pagerank"):
        users_throttled[strategy] = k
    if with_greedy:
        users_throttled["greedy"] = len(selected_by_strategy["greedy"])

    table = build_summary_table(G, posts, detections, results, n_runs, users_throttled)
    rows = []
    for strategy in strategies:
        row = table[strategy]
        if strategy == "baseline":
            cost = {"real_reach_lost": {"mean": 0.0, "std": 0.0}}
        else:
            cost = false_positive_cost(
                posts, detections, results["baseline"], results[strategy], n_runs
            )
        rows.append(
            {
                "strategy": strategy,
                "fake_reach_mean": row["final_fraction_reached"]["mean"],
                "fake_reach_std": row["final_fraction_reached"]["std"],
                "real_reach_lost_mean": cost["real_reach_lost"]["mean"],
                "real_reach_lost_std": cost["real_reach_lost"]["std"],
                "posts_acted_on": row["interventions"]["posts_acted_on"],
                "users_throttled": row["interventions"]["users_throttled"],
            }
        )
    curves = [
        {
            "label": strategy,
            "series": [
                result.newly_reached_by_step
                for post in posts
                if post.true_label == "fake"
                for result in results[strategy][post.id]
            ],
        }
        for strategy in strategies
    ]
    return rows, curves


def cmd_compare(args) -> None:
    G = _load_or_build_graph()
    posts = _limit_posts(args, _load_or_generate_posts(G))
    detector = _build_detector(args, posts)
    detections = _run_detections(args, detector, posts)
    summary = detector_summary(posts, detections)
    n_runs = args.runs
    budgets = args.budgets
    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"compare: detector={args.detector}, runs={n_runs}, budgets={budgets}")
    _print_detector_summary(summary)

    centralities = precompute_centralities(G)

    # baseline and blanket are budget-independent: compute once
    print("running baseline and blanket ...")
    shared = {}
    for strategy in ("baseline", "blanket"):
        shared[strategy] = _run_strategy(
            G, posts, detections, strategy, n_runs, k=config.K_BUDGET, selected=None
        )

    table_rows = []
    charts_by_budget = {}
    for k in budgets:
        rows, curves = _run_budget_block(
            G, posts, detections, centralities, n_runs, k, shared,
            args.greedy_pool, args.greedy_runs, with_greedy=True,
        )
        for row in rows:
            row["budget"] = k
        table_rows.extend(rows)
        charts_by_budget[k] = curves

    print("\n=== unified comparison (mean +/- std over runs) ===")
    print(
        f"{'budget':>6}  {'strategy':>11}  {'fake reach':>20}  "
        f"{'real-news reach lost':>26}  {'posts':>5}  {'users':>5}"
    )
    for row in table_rows:
        print(
            f"k={row['budget']:<5}  {row['strategy']:>11}  "
            f"{row['fake_reach_mean']:.3f} +/- {row['fake_reach_std']:<7.3f}  "
            f"{row['real_reach_lost_mean']:>10.1f} +/- {row['real_reach_lost_std']:<7.1f}  "
            f"{row['posts_acted_on']:>5}  {row['users_throttled']:>5}"
        )

    for k, curves in charts_by_budget.items():
        plot_reach_curves(
            curves,
            config.OUTPUT_DIR / f"strategy_comparison_k{k}.png",
            f"Fake posts per strategy at k={k} (mean ± std)",
        )

    payload = {
        "meta": _detector_meta(
            args, posts, n_runs=n_runs, budgets=budgets
        ),
        "detector_summary": summary,
        "detections": _detection_payload(posts, detections),
        "rows": table_rows,
    }
    _write_output(
        config.OUTPUT_DIR / "compare_metrics.json",
        payload,
        args.detector,
    )
    print(f"\ncharts and table saved to {config.OUTPUT_DIR}")


def cmd_sweep(args) -> None:
    n_runs = args.runs
    k = args.k
    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    base_graph = _load_or_build_graph()
    base_posts = _load_or_generate_posts(base_graph)
    # centralities depend on topology only, which every sweep shares
    centralities = precompute_centralities(base_graph)

    sweep_rows = []
    rankings = []
    curves_by_value = []

    for value in args.values:
        if args.mode == "p":
            p_spread = float(value)
            # same wiring seed -> identical topology, only edge p changes
            G = build_language_graph(p_spread=p_spread)
            posts = generate_posts(G)
            detector = StubDetector(posts)
            budget = k
        elif args.mode == "k":
            G = base_graph
            posts = base_posts
            detector = StubDetector(posts)
            budget = int(value)
        elif args.mode == "accuracy":
            G = base_graph
            posts = base_posts
            detector = StubDetector(posts, sensitivity=float(value))
            budget = k
        else:
            raise ValueError(f"unknown sweep mode: {args.mode}")

        detections = run_detection(detector, posts)
        print(f"sweep {args.mode}={value} ...")
        shared = {}
        for strategy in ("baseline", "blanket"):
            shared[strategy] = _run_strategy(
                G, posts, detections, strategy, n_runs,
                k=config.K_BUDGET, selected=None,
            )
        rows, curves = _run_budget_block(
            G, posts, detections, centralities, n_runs, budget, shared,
            args.greedy_pool, args.greedy_runs, args.with_greedy,
        )
        sorted_rows = sorted(rows, key=lambda row: row["fake_reach_mean"])
        order = [row["strategy"] for row in sorted_rows]
        rankings.append((value, order))
        for row in rows:
            row["sweep_value"] = value
        sweep_rows.extend(rows)
        curves_by_value.append((value, curves))

    print(f"\n=== {args.mode} sweep: strategy ranking by mean fake reach (lower is better) ===")
    for value, order in rankings:
        print(f"{args.mode}={value}: " + " < ".join(order))
    first_order = rankings[0][1]
    stable = all(order == first_order for _, order in rankings[1:])
    print(f"ranking stable across all values: {stable}")

    # one line per strategy across the swept values, fixed color slots
    x_values = [float(value) for value in args.values]
    if args.mode == "k":
        x_values = [int(value) for value in args.values]
    series_by_strategy = {}
    for strategy in ALL_STRATEGIES + (["greedy"] if args.with_greedy else []):
        means = []
        stds = []
        for row in sweep_rows:
            if row["strategy"] != strategy:
                continue
            means.append(row["fake_reach_mean"])
            stds.append(row["fake_reach_std"])
        series_by_strategy[strategy] = {"mean": means, "std": stds}
    x_label = {"p": "spread probability p", "k": "budget k", "accuracy": "stub sensitivity"}[args.mode]
    plot_sweep_lines(
        x_values,
        x_label,
        series_by_strategy,
        config.OUTPUT_DIR / f"sweep_{args.mode}.png",
        f"Sensitivity sweep: {x_label}",
    )

    payload = {
        "meta": {
            "mode": args.mode,
            "n_runs": n_runs,
            "k": k,
            "values": args.values,
            "detector": "stub",
            "detector_mode": "stub",
        },
        "rankings": [{"value": value, "order": order} for value, order in rankings],
        "rows": sweep_rows,
    }
    with open(config.OUTPUT_DIR / f"sweep_{args.mode}.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"\nplot and table saved to {config.OUTPUT_DIR}")


def cmd_bridges(args) -> None:
    G = _load_or_build_graph()
    bridges = detect_bridges(G)
    print_bridge_summary(G, bridges)


def cmd_priority(args) -> None:
    G = _load_or_build_graph()
    posts = _limit_posts(args, _load_or_generate_posts(G))
    detector = _build_detector(args, posts)
    detections = _run_detections(args, detector, posts)
    summary = detector_summary(posts, detections)
    ranked = rank_flagged_posts(G, posts, detections, eval_runs=args.runs)
    print("=== flagged posts by expected harm (priority = confidence x expected reach) ===")
    for post_id, priority, confidence, reach in ranked[:20]:
        print(
            f"  post {post_id:>2}: priority {priority:7.1f} "
            f"(confidence {confidence:.2f} x expected reach {reach:6.1f})"
        )
    print(f"  ... {len(ranked)} flagged posts total")
    _print_detector_summary(summary)

    payload = {
        "meta": _detector_meta(args, posts, eval_runs=args.runs),
        "detector_summary": summary,
        "detections": _detection_payload(posts, detections),
        "ranked": [
            {
                "post_id": post_id,
                "priority": priority,
                "confidence": confidence,
                "expected_reach": reach,
            }
            for post_id, priority, confidence, reach in ranked
        ],
    }
    _write_output(
        config.OUTPUT_DIR / "priority.json",
        payload,
        args.detector,
    )


def cmd_greedy(args) -> None:
    G = _load_or_build_graph()
    posts = _limit_posts(args, _load_or_generate_posts(G))
    detector = _build_detector(args, posts)
    detections = _run_detections(args, detector, posts)
    summary = detector_summary(posts, detections)
    _print_detector_summary(summary)
    n_runs = args.runs
    k = args.k

    centralities = precompute_centralities(G)
    pool = top_k_from_scores(centralities["betweenness"], args.pool)
    print(f"greedy: k={k}, candidate pool={len(pool)} (top betweenness), runs={n_runs}")
    selected = greedy_select(G, posts, detections, k, n_runs, pool)

    # compare greedy against baseline and betweenness-k on the same draws
    strategies = ["baseline", "betweenness", "greedy"]
    selected_by_strategy = {
        "betweenness": top_k_from_scores(centralities["betweenness"], k),
        "greedy": selected,
    }
    results = {}
    for strategy in strategies:
        results[strategy] = {}
        for post in posts:
            p_maps = build_strategy_p_maps(
                G,
                detections[post.id],
                strategy,
                n_runs,
                k=k,
                selected=selected_by_strategy.get(strategy),
            )
            results[strategy][post.id] = run_post_experiment(
                G, post, p_maps, n_runs, strategy
            )

    users_throttled = {
        "baseline": 0,
        "betweenness": k,
        "greedy": len(selected),
    }
    table = build_summary_table(G, posts, detections, results, n_runs, users_throttled)
    print("\n=== greedy vs betweenness vs baseline (fake posts) ===")
    for strategy in strategies:
        row = table[strategy]
        print(
            f"{strategy:>11}: final fraction {row['final_fraction_reached']['mean']:.3f} | "
            f"reduction {row['relative_reduction_vs_baseline'] * 100:.0f}% | "
            f"users throttled {row['interventions']['users_throttled']}"
        )
    costs = {}
    for strategy in ("betweenness", "greedy"):
        cost = false_positive_cost(posts, detections, results["baseline"], results[strategy], n_runs)
        costs[strategy] = cost
        print(
            f"{strategy:>11}: FP cost — {cost['fp_count']} real posts flagged, "
            f"{cost['real_reach_lost']['mean']:.1f} users of real-news reach lost per run, "
            f"{cost['uncertain_count']} uncertain, {cost['error_count']} errors"
        )

    curves = [
        {
            "label": strategy,
            "series": [
                result.newly_reached_by_step
                for post in posts
                if post.true_label == "fake"
                for result in results[strategy][post.id]
            ],
        }
        for strategy in strategies
    ]
    plot_reach_curves(
        curves,
        config.OUTPUT_DIR / "greedy_comparison.png",
        "Fake posts: greedy vs betweenness vs baseline (mean ± std)",
    )

    payload = {
        "meta": _detector_meta(
            args, posts, n_runs=n_runs, k=k, pool=args.pool
        ),
        "detector_summary": summary,
        "detections": _detection_payload(posts, detections),
        "selected_users": selected,
        "summary": table,
        "fp_cost": costs,
    }
    _write_output(
        config.OUTPUT_DIR / "greedy_metrics.json",
        payload,
        args.detector,
    )
    print(f"\nchart saved to {config.OUTPUT_DIR / 'greedy_comparison.png'}")


def _add_detector_integration_args(parser):
    parser.add_argument(
        "--max-posts",
        type=int,
        default=None,
        help=(
            "limit the posts processed; real mode defaults to "
            f"{config.DEFAULT_LIVE_MAX_POSTS} unless this is set"
        ),
    )
    parser.add_argument(
        "--detector-timeout",
        type=float,
        default=config.DETECTOR_TIMEOUT_SECONDS,
        help="per-post real-detector timeout in seconds",
    )
    parser.add_argument(
        "--detector-error-policy",
        choices=["record", "raise"],
        default=config.DETECTOR_ERROR_POLICY,
        help="record a structured error (no action) or abort on detector failure",
    )
    parser.add_argument(
        "--cache-provider",
        default=None,
        help="replay only: require this provider in every cache entry",
    )
    parser.add_argument(
        "--cache-model",
        default=None,
        help="replay only: require this model in every cache entry",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="verifacto",
        description="Simulated social network where fake news detections trigger containment actions.",
    )
    subparsers = parser.add_subparsers(dest="command")

    sub_graph = subparsers.add_parser("graph", help="build and save the language-block network")
    sub_graph.set_defaults(func=cmd_graph)

    sub_posts = subparsers.add_parser("posts", help="generate and save the post corpus")
    sub_posts.set_defaults(func=cmd_posts)

    sub_baseline = subparsers.add_parser(
        "baseline", help="run the Independent Cascade baseline (no protection)"
    )
    sub_baseline.set_defaults(func=cmd_baseline)

    sub_blanket = subparsers.add_parser(
        "blanket", help="detector + blanket halving of flagged fakes"
    )
    sub_blanket.add_argument(
        "--detector", choices=["stub", "replay", "real"], default="stub"
    )
    sub_blanket.add_argument("--allow-live", action="store_true")
    _add_detector_integration_args(sub_blanket)
    sub_blanket.set_defaults(func=cmd_blanket)

    sub_precompute = subparsers.add_parser(
        "precompute",
        help="run the real detector over the corpus and cache verdicts",
    )
    sub_precompute.add_argument(
        "--posts-path", default=None,
        help="optional custom posts JSON (e.g. a corpus with source_url values)",
    )
    sub_precompute.add_argument(
        "--max-posts", type=int, default=None,
        help="process at most this many posts in this run",
    )
    sub_precompute.add_argument(
        "--pacing", type=float, default=config.PACING_SECONDS,
        help="seconds between live posts",
    )
    sub_precompute.add_argument(
        "--detector-timeout", type=float,
        default=config.DETECTOR_TIMEOUT_SECONDS,
    )
    sub_precompute.add_argument(
        "--recompute-stale", action="store_true",
        help="rebuild entries whose policy version is missing or stale",
    )
    sub_precompute.set_defaults(func=cmd_precompute)

    sub_targeted = subparsers.add_parser(
        "targeted", help="top-k targeted containment, full evaluation"
    )
    sub_targeted.add_argument(
        "--detector", choices=["stub", "replay", "real"], default="stub",
        help="verdict source (default: deterministic stub)",
    )
    sub_targeted.add_argument("--runs", type=int, default=config.N_RUNS)
    sub_targeted.add_argument("--k", type=int, default=config.K_BUDGET)
    sub_targeted.add_argument(
        "--allow-live", action="store_true",
        help="allow live real-detector calls (slow, non-deterministic)",
    )
    _add_detector_integration_args(sub_targeted)
    sub_targeted.set_defaults(func=cmd_targeted)

    sub_compare = subparsers.add_parser(
        "compare",
        help="unified comparison of all strategies at k=10 and k=25",
    )
    sub_compare.add_argument(
        "--detector", choices=["stub", "replay", "real"], default="stub"
    )
    sub_compare.add_argument("--runs", type=int, default=config.N_RUNS)
    sub_compare.add_argument("--budgets", type=int, nargs="+", default=[10, 25])
    sub_compare.add_argument("--greedy-pool", type=int, default=60)
    sub_compare.add_argument("--greedy-runs", type=int, default=5)
    sub_compare.add_argument("--allow-live", action="store_true")
    _add_detector_integration_args(sub_compare)
    sub_compare.set_defaults(func=cmd_compare)

    sub_sweep = subparsers.add_parser(
        "sweep", help="sensitivity sweeps: p (spread), k (budget), accuracy (stub)"
    )
    sub_sweep.add_argument("mode", choices=["p", "k", "accuracy"])
    sub_sweep.add_argument("--values", nargs="+", required=True)
    sub_sweep.add_argument("--runs", type=int, default=config.N_RUNS)
    sub_sweep.add_argument(
        "--k", type=int, default=config.K_BUDGET,
        help="budget used by the p and accuracy sweeps",
    )
    sub_sweep.add_argument("--with-greedy", action="store_true")
    sub_sweep.add_argument("--greedy-pool", type=int, default=60)
    sub_sweep.add_argument("--greedy-runs", type=int, default=5)
    sub_sweep.set_defaults(func=cmd_sweep)

    sub_bridges = subparsers.add_parser("bridges", help="detect cross-language bridge users")
    sub_bridges.set_defaults(func=cmd_bridges)

    sub_priority = subparsers.add_parser(
        "priority", help="rank flagged posts by confidence x expected reach"
    )
    sub_priority.add_argument(
        "--detector", choices=["stub", "replay", "real"], default="stub"
    )
    sub_priority.add_argument("--runs", type=int, default=10)
    sub_priority.add_argument("--allow-live", action="store_true")
    _add_detector_integration_args(sub_priority)
    sub_priority.set_defaults(func=cmd_priority)

    sub_greedy = subparsers.add_parser(
        "greedy", help="greedy top-k throttling (empirical heuristic)"
    )
    sub_greedy.add_argument(
        "--detector", choices=["stub", "replay", "real"], default="stub"
    )
    sub_greedy.add_argument("--runs", type=int, default=5)
    sub_greedy.add_argument("--k", type=int, default=10)
    sub_greedy.add_argument("--pool", type=int, default=60)
    sub_greedy.add_argument("--allow-live", action="store_true")
    _add_detector_integration_args(sub_greedy)
    sub_greedy.set_defaults(func=cmd_greedy)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if args.command is None:
        parser.print_help()
    else:
        args.func(args)


if __name__ == "__main__":
    main()
