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
from .data.posts import generate_posts, load_posts, print_posts_summary, save_posts
from .evaluation.plots import plot_reach_curves
from .network.graph_builder import (
    build_language_graph,
    load_graph,
    print_graph_summary,
    save_graph,
)
from .simulation.cascade import run_ic_cascade


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
