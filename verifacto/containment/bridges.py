"""Phase 6: cross-language bridge detection.

Bridge users are heavy bilingual connectors: users with many outgoing
edges into OTHER language blocks. They are the natural points where a
claim jumps from one language to another, so they are candidate
containment targets.

With p_out=0.01 nearly every user has a couple of cross-language
out-edges, so the raw "reaches more than one language" test matches
almost everyone. The useful signal is the cross-language OUT-DEGREE:
the planted bilingual bridges have 10-20 such edges against a background
mean of ~3. Louvain is run as a sanity check that the detected
communities match the planted language blocks.
"""

import networkx as nx

# planted bridges have 10+ cross-language out-edges; background p95 is 7
BRIDGE_CROSS_DEGREE_THRESHOLD = 8


def detect_bridges(G, threshold=BRIDGE_CROSS_DEGREE_THRESHOLD):
    """Users whose cross-language out-degree reaches the threshold.

    Returns {node_id: (cross_out_degree, distinct_languages_reached)}
    sorted by cross out-degree, highest first.
    """
    cross_degree = {}
    for node in G.nodes():
        languages_reached = set()
        count = 0
        for successor in G.successors(node):
            if G.nodes[successor]["language"] != G.nodes[node]["language"]:
                count += 1
                languages_reached.add(G.nodes[successor]["language"])
        if count >= threshold:
            cross_degree[node] = (count, len(languages_reached))
    return dict(sorted(cross_degree.items(), key=lambda item: item[1][0], reverse=True))


def check_louvain_matches_blocks(G):
    """True when Louvain recovers the planted language blocks exactly."""
    communities = nx.community.louvain_communities(G.to_undirected(), seed=42)
    block_languages = {}
    for community in communities:
        languages = {G.nodes[node]["language"] for node in community}
        if len(languages) != 1:
            return False
        block_languages[list(languages)[0]] = len(community)
    expected = {language: 0 for language in {"en", "hi", "fr"}}
    for node in G.nodes():
        expected[G.nodes[node]["language"]] += 1
    return block_languages == expected


def print_bridge_summary(G, bridges) -> None:
    planted = [n for n, d in G.nodes(data=True) if d["is_bridge"]]
    detected = list(bridges.keys())
    overlap = [n for n in detected if G.nodes[n]["is_bridge"]]
    print("=== bridge detection (cross-language out-degree) ===")
    print(f"Louvain matches planted language blocks: {check_louvain_matches_blocks(G)}")
    print(f"detected bridges: {len(detected)}  planted bridges: {len(planted)}")
    print(f"planted bridges among detected: {len(overlap)}/{len(planted)}")
    for node in detected[:10]:
        flag = "planted" if G.nodes[node]["is_bridge"] else "detected"
        cross_degree, languages_reached = bridges[node]
        print(
            f"  user {node} ({G.nodes[node]['language']}, {flag}): "
            f"{cross_degree} cross-language out-edges, {languages_reached} languages"
        )
