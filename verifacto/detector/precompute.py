"""One-time run of the real detector over the corpus, cached to JSON.

Run once (roughly an hour for 80 posts against live APIs), resume-able:
posts already in the cache are skipped, and the cache is rewritten after
every post so an interrupted run keeps its progress. The cache holds no
ground-truth labels — only what the detector itself reported.

Degraded-verdict guard: when the pipeline hits an internal error (for
example an exhausted LLM quota) it returns an UNCERTAIN claim (older
responses used NOT_ENOUGH_EVIDENCE) with confidence 0.0, an internal
NOT_ENOUGH_EVIDENCE evidence verdict, and a "Pipeline error" escalation
reason. Those results are NOT cached, so a later resume retries the post
instead of silently keeping garbage. Three consecutive failures abort the
run.

The whole run happens inside ONE event loop: the pipeline binds its
global clients (async DB engine, HTTP clients) to the first loop it
sees, and repeated asyncio.run() calls would break them.
"""

import asyncio
import json
import os
import shutil
from datetime import datetime, timezone

from ..config import (
    CACHE_PATH,
    CONFIDENCE_THRESHOLD,
    DETECTOR_POLICY_VERSION,
    DETECTOR_TIMEOUT_SECONDS,
    PACING_SECONDS,
    POSTS_PATH,
)
from ..data.posts import load_posts
from .base import (
    DetectorUnavailableError,
    ensure_real_provider,
    make_cache_key,
    map_response_dict,
)
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

    The orchestrator catches claim-level exceptions and returns an
    UNCERTAIN claim with confidence 0.0, requires_human_review, a
    "Pipeline error: ..." escalation reason, and (in the three-way
    response format) evidence_verdict == NOT_ENOUGH_EVIDENCE. Older
    responses used verdict == NOT_ENOUGH_EVIDENCE. A genuine low-evidence
    verdict has confidence > 0 or a different escalation reason.
    """
    if not claims:
        # no checkable claims is a genuine outcome, not an error
        return False
    for claim in claims:
        if claim.get("confidence", 0.0) != 0.0:
            return False
        if claim.get("verdict") not in ("NOT_ENOUGH_EVIDENCE", "UNCERTAIN"):
            return False
        evidence_verdict = claim.get("evidence_verdict")
        if evidence_verdict is not None and evidence_verdict != "NOT_ENOUGH_EVIDENCE":
            return False
        if "Pipeline error" not in (claim.get("escalation_reason") or ""):
            return False
    return True


def _cache_entry(
    result,
    claims,
    provider,
    model,
    post=None,
    cache_key_value=None,
):
    return {
        "status": "completed",
        "post_id": getattr(post, "id", result.post_id),
        "cache_key": cache_key_value,
        "policy_version": DETECTOR_POLICY_VERSION,
        "claim_text": getattr(post, "text", result.claim_text),
        "language": getattr(post, "language", result.language),
        "source_url": getattr(post, "source_url", result.source_url),
        "total_claims": len(claims),
        "provider": provider,
        "model": model,
        "cached_at": datetime.now(timezone.utc).isoformat(),
        "record": result.to_dict(),
        "results": [
            {
                "claim": claim.get("claim"),
                "verdict": claim.get("verdict"),
                "evidence_verdict": claim.get("evidence_verdict"),
                "confidence": claim.get("confidence"),
                "verdict_reason": claim.get("verdict_reason"),
                "verdict_support_score": claim.get("verdict_support_score"),
                "requires_human_review": claim.get("requires_human_review"),
                "escalation_reason": claim.get("escalation_reason"),
                "matched_blacklist_domain": claim.get("matched_blacklist_domain"),
                "blacklist_category": claim.get("blacklist_category"),
                "source_domain": claim.get("source_domain"),
            }
            for claim in claims
        ],
        "mapped": {
            "is_fake": result.is_fake,
            "confidence": result.confidence,
            "verdict_summary": result.verdict_summary,
            "verdict": result.verdict,
        },
    }


def _result_has_provider_failure(result) -> bool:
    """True when a completed run recorded a search/FactCheck failure."""
    status = result.provider_status or {}
    if status.get("search_error"):
        return True
    if status.get("factcheck_error"):
        return True
    factcheck_status = status.get("factcheck_status")
    if isinstance(factcheck_status, int) and factcheck_status >= 500:
        return True
    return False


def _record_has_provider_failure(record) -> bool:
    if not record:
        return False
    status = record.get("provider_status") or {}
    if status.get("search_error"):
        return True
    if status.get("factcheck_error"):
        return True
    factcheck_status = status.get("factcheck_status")
    if isinstance(factcheck_status, int) and factcheck_status >= 500:
        return True
    return False


def _entry_status(entry, current_provider=None, current_model=None) -> str:
    """Classify a cache entry for batch-safe stale recomputation.

    Entries are stale when the detector policy version or the configured
    provider/model differs, or when the cached run recorded a provider
    failure. Legacy entries without a policy version are also stale, but
    are kept unless --recompute-stale is explicit.
    """
    if entry is None:
        return "missing"
    version = entry.get("policy_version")
    if version is None:
        return "legacy"
    if version != DETECTOR_POLICY_VERSION:
        return "stale"
    if current_provider is not None and entry.get("provider") != current_provider:
        return "stale"
    if current_model is not None and entry.get("model") != current_model:
        return "stale"
    if _record_has_provider_failure(entry.get("record")):
        return "stale"
    return "current"


def _backup_cache(cache_path):
    if not cache_path.exists():
        return None
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    backup_path = cache_path.with_name(
        f"{cache_path.stem}.backup-{timestamp}{cache_path.suffix}"
    )
    shutil.copy2(cache_path, backup_path)
    return backup_path


def _providers_meta(cache):
    """Per-post provider/model records plus a summary over the cache."""
    per_post = {}
    used = {}
    for post_id_str, entry in cache["verdicts"].items():
        provider = entry.get("provider", "unknown")
        model = entry.get("model")
        per_post[post_id_str] = {"provider": provider, "model": model}
        key = (provider, model)
        if key not in used:
            used[key] = []
        used[key].append(post_id_str)
    return {
        "per_post": per_post,
        "used": [
            {"provider": provider, "model": model, "posts": len(ids)}
            for (provider, model), ids in used.items()
        ],
    }


async def _run_all(
    adapter,
    posts,
    cache,
    cache_path,
    pacing,
    max_posts=None,
    recompute_stale=False,
    current_provider=None,
    current_model=None,
):
    if max_posts is not None:
        posts = posts[:max_posts]

    consecutive_failures = 0
    for post in posts:
        existing = cache["verdicts"].get(str(post.id))
        status = _entry_status(
            existing,
            current_provider,
            current_model,
        )

        if status == "current":
            print(f"  [{post.id}] current cache entry, skipping")
            continue
        if status in ("legacy", "stale") and not recompute_stale:
            if status == "legacy":
                print(
                    f"  [{post.id}] legacy cache entry kept "
                    f"(no policy version); skipping"
                )
            else:
                print(
                    f"  [{post.id}] stale cache entry "
                    f"(version={existing.get('policy_version')!r}, "
                    f"provider={existing.get('provider')!r}, "
                    f"model={existing.get('model')!r}); kept — "
                    f"pass --recompute-stale to rebuild"
                )
            continue

        try:
            response = await adapter.run_job(
                post.text,
                post.language,
                post.id,
                post.source_url,
            )
            claims = [claim.model_dump() for claim in response.results]
        except DetectorUnavailableError as exc:
            print(
                f"ABORT: detector unavailable at post {post.id}: "
                f"{adapter.safe_error(exc)}"
            )
            return 1
        except Exception as exc:
            print(
                f"ABORT: detection failed at post {post.id}: "
                f"{adapter.safe_error(exc)}"
            )
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

        provider = adapter.active_provider()
        model = adapter.active_model()
        result = map_response_dict(
            response.model_dump(mode="json"),
            post_id=post.id,
            text=post.text,
            language=post.language,
            source_url=post.source_url,
            provider=provider,
            detector_mode="real",
            latency_ms=adapter.last_latency_ms,
            provider_usage=adapter.last_provider_usage,
            provider_status=adapter.last_provider_status,
        )

        if _result_has_provider_failure(result):
            consecutive_failures += 1
            print(
                f"  [{post.id}] provider failure {result.provider_status} — "
                f"not cached, will retry on resume "
                f"({consecutive_failures}/3)"
            )
            if consecutive_failures >= 3:
                print("ABORT: 3 consecutive provider failures — fix provider "
                      "availability and resume")
                return 1
            await asyncio.sleep(pacing)
            continue

        consecutive_failures = 0
        cache_key_value = make_cache_key(
            post.text,
            post.language,
            post.source_url,
        )
        cache["verdicts"][str(post.id)] = _cache_entry(
            result,
            claims,
            provider,
            model,
            post=post,
            cache_key_value=cache_key_value,
        )
        cache["meta"] = {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "adapter": "real",
            "detector_mode": "real",
            "policy_version": DETECTOR_POLICY_VERSION,
            "threshold": CONFIDENCE_THRESHOLD,
            "pacing_seconds": pacing,
            "max_posts": max_posts,
            "recompute_stale": recompute_stale,
            "providers": _providers_meta(cache),
        }
        write_cache_atomic(cache_path, cache)
        print(
            f"  [{post.id}] ({post.language}) -> {result.verdict} "
            f"(legacy={result.verdict_summary}) "
            f"conf={result.confidence:.2f} "
            f"support={result.support_score} "
            f"blacklist={result.blacklisted} "
            f"usage={result.provider_usage} "
            f"{result.latency_ms or 0.0:.0f}ms (cached)"
        )
        await asyncio.sleep(pacing)

    cached_in_subset = sum(
        1 for post in posts if str(post.id) in cache["verdicts"]
    )
    print(
        f"done: {cached_in_subset}/{len(posts)} selected posts cached "
        f"({len(cache['verdicts'])} total entries) to {cache_path}"
    )
    return 0


def main(
    cache_path=CACHE_PATH,
    posts_path=POSTS_PATH,
    pacing=PACING_SECONDS,
    max_posts=None,
    timeout_seconds=DETECTOR_TIMEOUT_SECONDS,
    recompute_stale=False,
):
    posts = load_posts(posts_path)
    cache = load_existing_cache(cache_path)
    adapter = RealDetectorAdapter(
        timeout_seconds=timeout_seconds,
        error_policy="raise",
    )
    try:
        provider = adapter.active_provider()
        model = adapter.active_model()
        ensure_real_provider(provider)
    except DetectorUnavailableError as exc:
        print(f"ABORT: detector unavailable: {adapter.safe_error(exc)}")
        return 1

    print("=== VeriFacto detector precompute ===")
    print(f"provider={provider} model={model}")
    print(f"{len(posts)} posts in corpus")
    if max_posts is not None:
        print(f"limiting this run to {max_posts} posts")
    if recompute_stale:
        backup = _backup_cache(cache_path)
        if backup is not None:
            print(f"cache backup: {backup}")
        print("stale/legacy entries will be recomputed; current entries are kept")
    print(
        "the first post may take a few minutes: sentence-transformers and "
        "NLI models download on first use"
    )
    return asyncio.run(
        _run_all(
            adapter,
            posts,
            cache,
            cache_path,
            pacing,
            max_posts,
            recompute_stale,
            provider,
            model,
        )
    )
