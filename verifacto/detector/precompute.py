"""One-time run of the real detector over the corpus, cached to JSON.

Run once (roughly an hour for 80 posts against live APIs), resume-able:
posts already in the cache are skipped, and the cache is rewritten after
every post so an interrupted run keeps its progress. The cache holds no
ground-truth labels — only what the detector itself reported.

Degraded-verdict guard: when the pipeline hits an internal error (for
example an exhausted LLM quota) it returns a NOT_ENOUGH_EVIDENCE claim
with confidence 0.0 and a "Pipeline error" escalation reason. Those
results are NOT cached, so a later resume retries the post instead of
silently keeping garbage. Three consecutive failures abort the run.

The whole run happens inside ONE event loop: the pipeline binds its
global clients (async DB engine, HTTP clients) to the first loop it
sees, and repeated asyncio.run() calls would break them.
"""

import asyncio
import json
import os
from datetime import datetime, timezone

from ..config import CACHE_PATH, CONFIDENCE_THRESHOLD, PACING_SECONDS, POSTS_PATH
from ..data.posts import load_posts
from .base import DetectorUnavailableError, map_job_results
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


def _claims_are_degraded(claims):
    """True when every claim is a pipeline-error placeholder.

    The orchestrator catches claim-level exceptions and returns a
    NOT_ENOUGH_EVIDENCE claim with confidence 0.0, requires_human_review,
    and a "Pipeline error: ..." escalation reason. A genuine low-evidence
    verdict has confidence > 0 or a different escalation reason.
    """
    if not claims:
        # no checkable claims is a genuine outcome, not an error
        return False
    for claim in claims:
        if claim.get("confidence", 0.0) != 0.0:
            return False
        if claim.get("verdict") != "NOT_ENOUGH_EVIDENCE":
            return False
        if "Pipeline error" not in (claim.get("escalation_reason") or ""):
            return False
    return True


def _cache_entry(result, claims):
    return {
        "status": "completed",
        "total_claims": len(claims),
        "provider": result.provider,
        "cached_at": datetime.now(timezone.utc).isoformat(),
        "results": [
            {
                "claim": claim.get("claim"),
                "verdict": claim.get("verdict"),
                "confidence": claim.get("confidence"),
                "requires_human_review": claim.get("requires_human_review"),
                "escalation_reason": claim.get("escalation_reason"),
            }
            for claim in claims
        ],
        "mapped": {
            "is_fake": result.is_fake,
            "confidence": result.confidence,
            "verdict_summary": result.verdict_summary,
        },
    }


async def _run_all(adapter, posts, cache, cache_path, pacing):
    consecutive_failures = 0
    for post in posts:
        if str(post.id) in cache["verdicts"]:
            print(f"  [{post.id}] already cached, skipping")
            continue
        try:
            response = await adapter.run_job(post.text, post.language, post.id)
            claims = [claim.model_dump() for claim in response.results]
        except DetectorUnavailableError as exc:
            print(f"ABORT: detector unavailable at post {post.id}: {exc}")
            return 1
        except Exception as exc:
            print(f"ABORT: detection failed at post {post.id}: {exc}")
            return 1

        if _claims_are_degraded(claims):
            consecutive_failures += 1
            print(
                f"  [{post.id}] degraded verdicts (pipeline errors) — "
                f"not cached, will retry on resume ({consecutive_failures}/3)"
            )
            if consecutive_failures >= 3:
                print("ABORT: 3 consecutive degraded results — the detector's "
                      "LLM provider may be out of quota; fix keys and resume")
                return 1
            await asyncio.sleep(pacing)
            continue
        consecutive_failures = 0

        result = map_job_results(
            claims, CONFIDENCE_THRESHOLD, provider=adapter._llm_service.provider
        )
        cache["verdicts"][str(post.id)] = _cache_entry(result, claims)
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
        await asyncio.sleep(pacing)

    n_cached = len(cache["verdicts"])
    print(f"done: {n_cached}/{len(posts)} posts cached to {cache_path}")
    return 0


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
    return asyncio.run(_run_all(adapter, posts, cache, cache_path, pacing))
