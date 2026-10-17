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

from . import config
from .data.posts import generate_posts, print_posts_summary, save_posts
from .network.graph_builder import (
    build_language_graph,
    print_graph_summary,
    save_graph,
)


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
