"""Step 1: build the ~500-user directed social graph with language clusters.

Structure: one stochastic block per language — dense edges inside a block,
sparse edges across blocks — plus a few hub users (high out-degree, same
block) and a few bilingual bridge users (out-edges into every other block).
Every edge carries the same spread probability p (fixed at 0.1).
"""

import networkx as nx
import numpy as np

from ..config import (
    BLOCK_SIZES,
    BRIDGE_EXTRA_PER_BLOCK,
    BRIDGES_PER_BLOCK,
    GRAPH_PATH,
    GRAPH_SEED,
    HUB_DEGREE_TARGET,
    HUBS_PER_BLOCK,
    LANGUAGES,
    P_IN,
    P_OUT,
    P_SPREAD,
)


def _add_random_out_edges(G, source, candidates, count, rng):
    """Add `count` outgoing edges from `source` to distinct random candidates.

    Skips self-loops and edges that already exist, so the final out-degree
    may end up lower than requested only when candidates run out.
    """
    added = 0
    for target in rng.choice(candidates, size=count, replace=False):
        if target == source:
            continue
        if G.has_edge(source, int(target)):
            continue
        G.add_edge(source, int(target), p=P_SPREAD)
        added += 1
    return added


def _add_hub_edges(G, block_nodes, hub_ids, hub_degree_target, rng):
    for hub in hub_ids:
        while G.out_degree(hub) < hub_degree_target:
            target = int(rng.choice(block_nodes))
            if target == hub:
                continue
            if G.has_edge(hub, target):
                continue
            G.add_edge(hub, target, p=P_SPREAD)


def _add_bridge_edges(G, block_nodes, bridge_ids, other_blocks, extra_per_block, rng):
    for bridge in bridge_ids:
        for other_block in other_blocks:
            _add_random_out_edges(G, bridge, other_block, extra_per_block, rng)


def build_language_graph(
    block_sizes=BLOCK_SIZES,
    languages=LANGUAGES,
    p_in=P_IN,
    p_out=P_OUT,
    hubs_per_block=HUBS_PER_BLOCK,
    hub_degree_target=HUB_DEGREE_TARGET,
    bridges_per_block=BRIDGES_PER_BLOCK,
    bridge_extra_per_block=BRIDGE_EXTRA_PER_BLOCK,
    p_spread=P_SPREAD,
    seed=GRAPH_SEED,
) -> nx.DiGraph:
    G = nx.random_partition_graph(
        block_sizes, p_in=p_in, p_out=p_out, seed=seed, directed=True
    )

    for node in G.nodes:
        G.nodes[node]["language"] = languages[G.nodes[node]["block"]]
        G.nodes[node]["is_hub"] = False
        G.nodes[node]["is_bridge"] = False

    nx.set_edge_attributes(G, p_spread, "p")

    rng = np.random.default_rng(seed)
    partition = G.graph["partition"]
    for block in range(len(block_sizes)):
        block_nodes = sorted(partition[block])
        hub_ids = [
            int(n)
            for n in rng.choice(block_nodes, size=hubs_per_block, replace=False)
        ]
        remaining = [n for n in block_nodes if n not in hub_ids]
        bridge_ids = [
            int(n)
            for n in rng.choice(remaining, size=bridges_per_block, replace=False)
        ]
        other_blocks = [
            sorted(partition[b]) for b in range(len(block_sizes)) if b != block
        ]

        for hub in hub_ids:
            G.nodes[hub]["is_hub"] = True
        for bridge in bridge_ids:
            G.nodes[bridge]["is_bridge"] = True

        _add_hub_edges(G, block_nodes, hub_ids, hub_degree_target, rng)
        _add_bridge_edges(
            G, block_nodes, bridge_ids, other_blocks, bridge_extra_per_block, rng
        )

    return G


def save_graph(G, path=GRAPH_PATH) -> None:
    """Save the graph as GraphML.

    The graph-level `partition` attribute (a list of sets) is not XML
    serializable, so it is dropped here; block membership lives on the
    node `block` attribute and can be rebuilt with `block_node_lists`.
    """
    saved = G.copy()
    if "partition" in saved.graph:
        del saved.graph["partition"]
    nx.write_graphml(saved, path)


def load_graph(path=GRAPH_PATH) -> nx.DiGraph:
    G = nx.read_graphml(path)
    # graphml stores node ids as strings; the rest of the code expects ints
    return nx.relabel_nodes(G, {n: int(n) for n in G.nodes}, copy=False)


def block_node_lists(G):
    """Node ids per language block, reconstructed from node attributes."""
    num_blocks = max(G.nodes[n]["block"] for n in G.nodes) + 1
    lists = [[] for _ in range(num_blocks)]
    for node in G.nodes:
        lists[G.nodes[node]["block"]].append(node)
    return [sorted(ids) for ids in lists]


def cross_language_edge_fraction(G) -> float:
    """Fraction of edges whose endpoints are in different language blocks."""
    if G.number_of_edges() == 0:
        return 0.0
    cross = 0
    for u, v in G.edges:
        if G.nodes[u]["language"] != G.nodes[v]["language"]:
            cross += 1
    return cross / G.number_of_edges()


def _density(G, node_pairs):
    """Edge density over a list of ordered (u, v) node pairs."""
    if not node_pairs:
        return 0.0
    present = sum(1 for u, v in node_pairs if G.has_edge(u, v))
    return present / len(node_pairs)


def in_block_density(G, block_nodes):
    pairs = [
        (u, v) for u in block_nodes for v in block_nodes if u != v
    ]
    return _density(G, pairs)


def cross_block_density(G, block_a, block_b):
    pairs = [(u, v) for u in block_a for v in block_b]
    return _density(G, pairs)


def overall_cross_block_density(G) -> float:
    """Edge density over all ordered pairs whose blocks differ."""
    partition = [sorted(block_nodes) for block_nodes in G.graph["partition"]]
    pairs = []
    for i, block_a in enumerate(partition):
        for j, block_b in enumerate(partition):
            if i == j:
                continue
            pairs.extend((u, v) for u in block_a for v in block_b)
    return _density(G, pairs)


def print_graph_summary(G) -> None:
    print("=== VeriFacto network summary ===")
    print(f"users: {G.number_of_nodes()}  edges: {G.number_of_edges()}  directed: {G.is_directed()}")

    partition = G.graph["partition"]
    for block, block_nodes in enumerate(partition):
        language = G.nodes[sorted(block_nodes)[0]]["language"]
        print(f"  block {block} ({language}): {len(block_nodes)} users")

    cross = cross_language_edge_fraction(G)
    print(f"cross-language edge fraction: {cross:.3f}")

    for block, block_nodes in enumerate(partition):
        block_nodes = sorted(block_nodes)
        language = G.nodes[block_nodes[0]]["language"]
        in_density = in_block_density(G, block_nodes)
        cross_density = sum(
            cross_block_density(G, block_nodes, sorted(other))
            for i, other in enumerate(partition)
            if i != block
        ) / max(1, len(partition) - 1)
        print(f"  {language}: in-block density {in_density:.3f}  cross-block density {cross_density:.4f}")

    hubs = [n for n, d in G.nodes(data=True) if d["is_hub"]]
    bridges = [n for n, d in G.nodes(data=True) if d["is_bridge"]]
    print(f"hubs: {len(hubs)}  (out-degrees: {sorted(G.out_degree(h) for h in hubs)})")
    print(f"bridges: {len(bridges)}")
    for b in bridges:
        neighbor_languages = sorted({G.nodes[v]["language"] for v in G.successors(b)})
        print(f"  bridge {b} ({G.nodes[b]['language']}) -> {neighbor_languages}")
