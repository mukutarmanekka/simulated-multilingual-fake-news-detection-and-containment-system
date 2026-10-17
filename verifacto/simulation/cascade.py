"""Step 3: Independent Cascade over the language-block graph.

Each newly reached user gets exactly one chance to infect each of their
out-neighbors, with the edge's spread probability. New infections become
the next time step's frontier; the cascade stops when a frontier is empty.

Reproducibility contract (D3): every edge has ONE fixed random draw per
(post, run), stored in a draw table built in sorted edge order. The same
table is shared by every strategy, and interventions only ever reduce
effective probabilities. Therefore any edge that succeeds under an
intervention also succeeds under the baseline, and the reached set of an
intervention is an exact subset of the baseline's per (post, run).
"""

import numpy as np


def build_draw_table(G, seed):
    """One random draw per edge, keyed by (u, v), in sorted edge order.

    Building the table in a fixed order makes it identical for every
    strategy that uses the same seed.
    """
    rng = np.random.default_rng(seed)
    draws = {}
    for u, v in sorted(G.edges()):
        draws[(u, v)] = rng.random()
    return draws


def run_ic_cascade(G, source, p_map=None, draws=None, seed=None):
    """Run one Independent Cascade and return newly-reached counts per step.

    `p_map` maps (u, v) to the effective spread probability (defaults to
    the graph's edge attribute p). `draws` is a draw table; when omitted it
    is built from `seed`, which is then required.
    """
    if draws is None:
        if seed is None:
            raise ValueError("seed is required when draws is not provided")
        draws = build_draw_table(G, seed)
    if p_map is None:
        p_map = {(u, v): G[u][v]["p"] for u, v in G.edges()}

    reached = {source}
    frontier = [source]
    newly_reached_by_step = [1]
    while frontier:
        next_frontier = []
        for u in frontier:
            for v in sorted(G.successors(u)):
                if v in reached:
                    continue
                if draws[(u, v)] < p_map[(u, v)]:
                    reached.add(v)
                    next_frontier.append(v)
        if not next_frontier:
            break
        newly_reached_by_step.append(len(next_frontier))
        frontier = next_frontier
    return newly_reached_by_step
