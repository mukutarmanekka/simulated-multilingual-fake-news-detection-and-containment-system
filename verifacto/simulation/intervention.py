"""Step 4: intervention actions applied to flagged posts.

Interventions never mutate the graph. They produce per-edge probability
maps (D4): effective_p(u,v) = G[u][v]["p"] * post_factor * user_factor(u).
Centralities and throttled-user selection are always computed on the
ORIGINAL graph.

Detection is sequential and label-free. Real-mode adapters are awaited
inside ONE event loop; stub/replay adapters use the synchronous method.
"""

import asyncio

import networkx as nx
import numpy as np

from ..config import (
    BLANKET_REDUCTION,
    DETECTOR_ERROR_POLICY,
    TARGETED_REDUCTION,
)
from ..detector.base import (
    DetectorUnavailableError,
    error_detection,
)


def blanket_factor(detection):
    """Post-wide spread factor: halve flagged fakes, leave the rest alone."""
    if detection is not None and detection.is_fake is True:
        return BLANKET_REDUCTION
    return 1.0


def build_prob_map(G, post_factor, throttled=None, throttle_factor=TARGETED_REDUCTION):
    """Effective per-edge probabilities for one run; G is never mutated."""
    p_map = {(u, v): G[u][v]["p"] * post_factor for u, v in G.edges()}
    if throttled:
        throttled_set = set(throttled)
        for u, v in G.edges():
            if u in throttled_set:
                p_map[(u, v)] = p_map[(u, v)] * throttle_factor
    return p_map


def _provider_name(adapter) -> str:
    try:
        return adapter.active_provider()
    except Exception:
        return getattr(adapter, "provider", "unknown")


def _detector_mode(adapter) -> str:
    if hasattr(adapter, "detect_async"):
        return "real"
    if adapter.__class__.__name__.lower().startswith("replay"):
        return "replay"
    return "stub"


def _error_record(adapter, post, exc, status="error"):
    return error_detection(
        post_id=post.id,
        text=post.text,
        language=post.language,
        source_url=getattr(post, "source_url", None),
        provider=_provider_name(adapter),
        detector_mode=_detector_mode(adapter),
        status=status,
        error=str(exc),
    )


async def _run_detection_async(adapter, posts, error_policy):
    detections = {}
    for post in posts:
        try:
            detections[post.id] = await adapter.detect_async(
                post.text,
                post.language,
                post.id,
                getattr(post, "source_url", None),
            )
        except DetectorUnavailableError:
            # Configuration failure: abort; never fall back to a stub.
            raise
        except Exception as exc:
            if error_policy == "raise":
                raise
            detections[post.id] = _error_record(adapter, post, exc)
    return detections


def run_detection(adapter, posts, error_policy=DETECTOR_ERROR_POLICY):
    """One detection per post, associated by post id.

    The adapter receives only text/language/id/source URL. The ground-truth
    label is never passed. Real adapters are awaited in one event loop so
    their global clients stay bound to a single loop; stub and replay
    adapters use the synchronous interface.
    """
    if hasattr(adapter, "detect_async"):
        return asyncio.run(_run_detection_async(adapter, posts, error_policy))

    detections = {}
    for post in posts:
        try:
            detections[post.id] = adapter.detect(
                post.text,
                post.language,
                post.id,
                getattr(post, "source_url", None),
            )
        except DetectorUnavailableError:
            raise
        except Exception as exc:
            if error_policy == "raise":
                raise
            detections[post.id] = _error_record(adapter, post, exc)
    return detections


def select_top_k(G, method, k, seed=0):
    """Top-k user ids by the given method, on the original graph.

    Methods: "degree" (out-degree), "betweenness", "pagerank", "random".
    The random control is drawn from a seeded rng so it is reproducible.
    """
    if method == "degree":
        scores = dict(G.out_degree())
    elif method == "betweenness":
        scores = nx.betweenness_centrality(G)
    elif method == "pagerank":
        scores = nx.pagerank(G)
    elif method == "random":
        rng = np.random.default_rng(seed)
        chosen = rng.choice(sorted(G.nodes()), size=k, replace=False)
        return sorted(int(node) for node in chosen)
    else:
        raise ValueError(f"unknown selection method: {method}")
    ranked = sorted(scores, key=lambda node: scores[node], reverse=True)
    return ranked[:k]
