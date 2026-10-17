"""Run one strategy's cascades over posts and runs.

Every strategy for a given (post, run) uses the same cascade seed and the
same draw table; strategies differ only through their probability maps.
That is what makes intervention reach a subset of baseline reach and lets
strategies be compared on common random numbers (D3).
"""

from dataclasses import dataclass

from ..config import K_BUDGET, cascade_seed
from .cascade import build_draw_table, run_ic_cascade
from .intervention import blanket_factor, build_prob_map, select_top_k

TARGETED_METHODS = ("random", "degree", "betweenness", "pagerank")


@dataclass
class PostRunResult:
    post_id: int
    run: int
    strategy: str
    newly_reached_by_step: list[int]

    @property
    def total_reached(self):
        return sum(self.newly_reached_by_step)


def build_strategy_p_maps(G, detection, strategy, n_runs):
    """Probability map per run for one strategy and one post's detection.

    Returns {run: p_map}; a p_map of None means "use the graph's edge p"
    (the baseline). Targeted strategies throttle only posts whose
    detection is fake; blanket halves them everywhere.
    """
    is_fake = detection is not None and detection.is_fake is True

    if strategy == "baseline":
        return {run: None for run in range(n_runs)}

    if strategy == "blanket":
        factor = blanket_factor(detection)
        if factor == 1.0:
            return {run: None for run in range(n_runs)}
        p_map = build_prob_map(G, factor)
        return {run: p_map for run in range(n_runs)}

    if strategy in TARGETED_METHODS:
        if not is_fake:
            return {run: None for run in range(n_runs)}
        if strategy == "random":
            p_maps = {}
            for run in range(n_runs):
                selected = select_top_k(G, "random", K_BUDGET, seed=cascade_seed(0, run))
                p_maps[run] = build_prob_map(G, 1.0, throttled=selected)
            return p_maps
        # deterministic selections: computed once, shared across runs
        selected = select_top_k(G, strategy, K_BUDGET)
        p_map = build_prob_map(G, 1.0, throttled=selected)
        return {run: p_map for run in range(n_runs)}

    raise ValueError(f"unknown strategy: {strategy}")


def run_post_experiment(G, post, p_maps, n_runs, strategy):
    """Run n_runs seeded cascades for one post under one strategy."""
    results = []
    for run in range(n_runs):
        seed = cascade_seed(post.id, run)
        draws = build_draw_table(G, seed)
        series = run_ic_cascade(G, post.author, p_map=p_maps[run], draws=draws)
        results.append(
            PostRunResult(
                post_id=post.id,
                run=run,
                strategy=strategy,
                newly_reached_by_step=series,
            )
        )
    return results
