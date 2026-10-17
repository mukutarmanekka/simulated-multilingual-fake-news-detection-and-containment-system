"""One-time run of the real detector over the corpus, cached to JSON.

Run once (roughly an hour for 80 posts against live APIs), resume-able:
posts already in the cache are skipped, and the cache is rewritten after
every post so an interrupted run keeps its progress. The cache holds no
ground-truth labels — only what the detector itself reported.
"""

import json
import os
import time
from datetime import datetime, timezone

from ..config import CACHE_PATH, CONFIDENCE_THRESHOLD, PACING_SECONDS, POSTS_PATH
from ..data.posts import load_posts
from .base import DetectorUnavailableError
from .real_adapter import RealDetectorAdapter


def load_existing_cache(cache_path):
    if not cache_path.exists():
        return {"meta": {}, "verdicts": {}}
    with open(cache_path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_cache_atomic(cache_path, payload):
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = cache_path.with_suffix(".tmp")
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    os.replace(tmp_path, cache_path)


def main(cache_path=CACHE_PATH, posts_path=POSTS_PATH, pacing=PACING_SECONDS):
    posts = load_posts(posts_path)
    cache = load_existing_cache(cache_path)
    adapter = RealDetectorAdapter()

    print("=== VeriFacto detector precompute ===")
    print(f"{len(posts)} posts in corpus")
    print(
        "the first post may take a few minutes: sentence-transformers and "
        "NLI models download on first use"
    )

    for post in posts:
        if str(post.id) in cache["verdicts"]:
            print(f"  [{post.id}] already cached, skipping")
            continue
        try:
            result = adapter.detect(post.text, post.language, post.id)
        except DetectorUnavailableError as exc:
            print(f"ABORT: detector unavailable at post {post.id}: {exc}")
            return 1
        except Exception as exc:
            print(f"ABORT: detection failed at post {post.id}: {exc}")
            return 1

        cache["verdicts"][str(post.id)] = {
            "status": "completed",
            "provider": result.provider,
            "cached_at": datetime.now(timezone.utc).isoformat(),
            "mapped": {
                "is_fake": result.is_fake,
                "confidence": result.confidence,
                "verdict_summary": result.verdict_summary,
            },
        }
        cache["meta"] = {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "adapter": "real",
            "threshold": CONFIDENCE_THRESHOLD,
            "pacing_seconds": pacing,
        }
        write_cache_atomic(cache_path, cache)
        print(
            f"  [{post.id}] ({post.language}) -> {result.verdict_summary} "
            f"conf={result.confidence:.2f} (cached)"
        )
        time.sleep(pacing)

    n_cached = len(cache["verdicts"])
    print(f"done: {n_cached}/{len(posts)} posts cached to {cache_path}")
    return 0
