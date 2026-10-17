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
from .evaluation.metrics import build_summary_table, false_positive_cost
from .evaluation.plots import plot_fp_cost_bar, plot_reach_curves
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
        "meta": {"n_runs": config.N_RUNS, "strategy": "baseline"},
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

    raise SystemExit(precompute.main())


def cmd_blanket(args) -> None:
    G = _load_or_build_graph()
    posts = _load_or_generate_posts(G)
    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    detector = StubDetector(posts)
    detections = run_detection(detector, posts)

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

    # false-positive cost: real posts wrongly flagged fake get halved too
    fp_count = 0
    reach_lost = []
    for post in posts:
        if post.true_label != "real":
            continue
        if detections[post.id].is_fake is True:
            fp_count += 1
            for run in range(config.N_RUNS):
                base = baseline_results[post.id][run].total_reached
                protected = blanket_results[post.id][run].total_reached
                reach_lost.append(base - protected)
    n_real = sum(1 for p in posts if p.true_label == "real")
    print(f"fake posts: mean final fraction (baseline) = "
          f"{np.mean([sum(r) for r in fake_baseline]) / G.number_of_nodes():.3f}")
    print(f"fake posts: mean final fraction (blanket)  = "
          f"{np.mean([sum(r) for r in fake_blanket]) / G.number_of_nodes():.3f}")
    print(f"false positives: {fp_count}/{n_real} real posts flagged fake")
    if reach_lost:
        print(f"real-news reach lost per run (mean): {np.mean(reach_lost):.1f} users")
    print(f"plot saved to {config.OUTPUT_DIR / 'blanket_vs_baseline.png'}")


def _build_detector(args, posts):
    if args.detector == "stub":
        return StubDetector(posts)
    if args.detector == "replay":
        if not config.CACHE_PATH.exists():
            print("no detector cache found — run `python -m verifacto.main precompute` first")
            raise SystemExit(1)
        return ReplayDetector()
    if args.detector == "real":
        if not args.allow_live:
            print("live detection is slow and non-deterministic; pass --allow-live to proceed")
            raise SystemExit(1)
        return RealDetectorAdapter()
    raise ValueError(f"unknown detector: {args.detector}")


def cmd_targeted(args) -> None:
    G = _load_or_build_graph()
    posts = _load_or_generate_posts(G)
    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    n_runs = args.runs
    k = args.k

    detector = _build_detector(args, posts)
    print(f"detector: {args.detector}")
    detections = run_detection(detector, posts)

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
        "meta": {"detector": args.detector, "n_runs": n_runs, "k": k},
        "summary": table,
        "fp_cost": costs,
    }
    with open(config.OUTPUT_DIR / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

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


def cmd_compare(args) -> None:
    G = _load_or_build_graph()
    posts = _load_or_generate_posts(G)
    detector = _build_detector(args, posts)
    detections = run_detection(detector, posts)
    n_runs = args.runs
    budgets = args.budgets
    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"compare: detector={args.detector}, runs={n_runs}, budgets={budgets}")

    centralities = precompute_centralities(G)

    # baseline and blanket are budget-independent: compute once
    print("running baseline and blanket ...")
    shared = {}
    for strategy in ("baseline", "blanket"):
        shared[strategy] = _run_strategy(
            G, posts, detections, strategy, n_runs, k=config.K_BUDGET, selected=None
        )

    # greedy selects per budget on its own seed space (disjoint from the
    # evaluation draws it will be scored on)
    greedy_selected = {}
    for k in budgets:
        pool = top_k_from_scores(centralities["betweenness"], args.greedy_pool)
        print(
            f"greedy selection for k={k} "
            f"(pool {len(pool)}, {args.greedy_runs} selection runs) ..."
        )
        greedy_selected[k] = greedy_select(
            G, posts, detections, k, args.greedy_runs, pool
        )

    table_rows = []
    charts_by_budget = {}
    for k in budgets:
        results = dict(shared)
        selected_by_strategy = {
            method: top_k_from_scores(centralities[method], k)
            for method in ("degree", "betweenness", "pagerank")
        }
        selected_by_strategy["greedy"] = greedy_selected[k]
        for strategy in ("random", "degree", "betweenness", "pagerank", "greedy"):
            results[strategy] = _run_strategy(
                G,
                posts,
                detections,
                strategy,
                n_runs,
                k=k,
                selected=selected_by_strategy.get(strategy),
            )

        users_throttled = {"baseline": 0, "blanket": 0}
        for strategy in ("random", "degree", "betweenness", "pagerank"):
            users_throttled[strategy] = k
        users_throttled["greedy"] = len(greedy_selected[k])

        table = build_summary_table(
            G, posts, detections, results, n_runs, users_throttled
        )
        for strategy in (
            "baseline",
            "blanket",
            "random",
            "degree",
            "betweenness",
            "pagerank",
            "greedy",
        ):
            row = table[strategy]
            if strategy == "baseline":
                cost = {"real_reach_lost": {"mean": 0.0, "std": 0.0}}
            else:
                cost = false_positive_cost(
                    posts, detections, results["baseline"], results[strategy], n_runs
                )
            table_rows.append(
                {
                    "budget": k,
                    "strategy": strategy,
                    "fake_reach_mean": row["final_fraction_reached"]["mean"],
                    "fake_reach_std": row["final_fraction_reached"]["std"],
                    "real_reach_lost_mean": cost["real_reach_lost"]["mean"],
                    "real_reach_lost_std": cost["real_reach_lost"]["std"],
                    "posts_acted_on": row["interventions"]["posts_acted_on"],
                    "users_throttled": row["interventions"]["users_throttled"],
                }
            )

        charts_by_budget[k] = [
            {
                "label": strategy,
                "series": [
                    result.newly_reached_by_step
                    for post in posts
                    if post.true_label == "fake"
                    for result in results[strategy][post.id]
                ],
            }
            for strategy in (
                "baseline",
                "blanket",
                "random",
                "degree",
                "betweenness",
                "pagerank",
                "greedy",
            )
        ]

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
        "meta": {"detector": args.detector, "n_runs": n_runs, "budgets": budgets},
        "rows": table_rows,
    }
    with open(config.OUTPUT_DIR / "compare_metrics.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"\ncharts and table saved to {config.OUTPUT_DIR}")


def cmd_bridges(args) -> None:
    G = _load_or_build_graph()
    bridges = detect_bridges(G)
    print_bridge_summary(G, bridges)


def cmd_priority(args) -> None:
    G = _load_or_build_graph()
    posts = _load_or_generate_posts(G)
    detector = _build_detector(args, posts)
    detections = run_detection(detector, posts)
    ranked = rank_flagged_posts(G, posts, detections, eval_runs=args.runs)
    print("=== flagged posts by expected harm (priority = confidence x expected reach) ===")
    for post_id, priority, confidence, reach in ranked[:20]:
        print(
            f"  post {post_id:>2}: priority {priority:7.1f} "
            f"(confidence {confidence:.2f} x expected reach {reach:6.1f})"
        )
    print(f"  ... {len(ranked)} flagged posts total")


def cmd_greedy(args) -> None:
    G = _load_or_build_graph()
    posts = _load_or_generate_posts(G)
    detector = _build_detector(args, posts)
    detections = run_detection(detector, posts)
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
    for strategy in ("betweenness", "greedy"):
        cost = false_positive_cost(posts, detections, results["baseline"], results[strategy], n_runs)
        print(
            f"{strategy:>11}: FP cost — {cost['fp_count']} real posts flagged, "
            f"{cost['real_reach_lost']['mean']:.1f} users of real-news reach lost per run"
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
    print(f"\nchart saved to {config.OUTPUT_DIR / 'greedy_comparison.png'}")


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
    sub_blanket.set_defaults(func=cmd_blanket)

    sub_precompute = subparsers.add_parser(
        "precompute",
        help="run the real detector over the corpus and cache verdicts (once, ~1h)",
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
    sub_compare.set_defaults(func=cmd_compare)

    sub_bridges = subparsers.add_parser("bridges", help="detect cross-language bridge users")
    sub_bridges.set_defaults(func=cmd_bridges)

    sub_priority = subparsers.add_parser(
        "priority", help="rank flagged posts by confidence x expected reach"
    )
    sub_priority.add_argument(
        "--detector", choices=["stub", "replay", "real"], default="stub"
    )
    sub_priority.add_argument("--runs", type=int, default=10)
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
