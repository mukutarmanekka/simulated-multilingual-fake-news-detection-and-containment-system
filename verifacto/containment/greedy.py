"""Phase 6: greedy throttled-user selection (empirical heuristic).

NOTE: the claim that expected-spread-reduction over throttled sets is
monotone submodular is NOT relied on here — blocking objectives are known
to be non-submodular in general, so greedy carries no (1-1/e) guarantee
and CELF's lazy bound would be invalid. Greedy is used as a documented
empirical heuristic, benchmarked against the centrality methods.

Honesty rule: the objective is defined over FLAGGED posts (detector says
fake), never over ground-truth labels — an operator does not know which
flagged posts are truly fake. The false-positive cost of the resulting
selection is measured afterwards by the evaluation layer.

The objective is exact, not Monte Carlo: every candidate is evaluated on
precomputed per-(post, run) draw tables with common random numbers. Those
draws come from the greedy's OWN seed space (greedy_seed), disjoint from
the evaluation cascade seeds — selecting on the evaluation draws would
overfit the selection to the exact randomness it is scored on.
"""

from ..config import greedy_seed
from ..simulation.cascade import build_draw_table, run_ic_cascade
from ..simulation.intervention import build_prob_map


def _flagged_posts(posts, detections):
    return [post for post in posts if detections[post.id].is_fake is True]


def _total_reached_flagged(G, flagged_posts, throttled, draw_tables):
    """Total reached over flagged posts and runs with `throttled` throttled."""
    p_map = build_prob_map(G, 1.0, throttled=throttled)
    total = 0
    for post in flagged_posts:
        for run, draws in draw_tables[post.id].items():
            series = run_ic_cascade(G, post.author, p_map=p_map, draws=draws)
            total += sum(series)
    return total


def greedy_select(G, posts, detections, k, n_runs, candidates):
    """Greedy top-k throttled users with full reevaluation every round.

    `candidates` is the candidate pool (e.g. the top-100 nodes by
    betweenness — searching all 500 users is rarely worth it). Each round
    picks the candidate whose addition reduces the objective most.
    Returns the selected user ids in pick order.
    """
    flagged_posts = _flagged_posts(posts, detections)
    draw_tables = {}
    for post in flagged_posts:
        draws_by_run = {}
        for run in range(n_runs):
            seed = greedy_seed(post.id, run)
            draws_by_run[run] = build_draw_table(G, seed)
        draw_tables[post.id] = draws_by_run

    selected = []
    best_total = _total_reached_flagged(G, flagged_posts, selected, draw_tables)
    pool = [candidate for candidate in candidates if candidate not in selected]
    for round_index in range(k):
        best_candidate = None
        best_reduction = 0.0
        for candidate in pool:
            total = _total_reached_flagged(
                G, flagged_posts, selected + [candidate], draw_tables
            )
            reduction = best_total - total
            if reduction > best_reduction:
                best_reduction = reduction
                best_candidate = candidate
        if best_candidate is None:
            # no candidate strictly improves the objective; stop early
            break
        selected.append(best_candidate)
        pool.remove(best_candidate)
        best_total = best_total - best_reduction
        print(
            f"    greedy round {round_index + 1}: +{best_candidate} "
            f"(-{best_reduction:.1f} reached)"
        )
    return selected
