"""Replay cached verdicts from the real-detector precompute run.

New cache entries are keyed to the claim, language, source URL, and
detector policy version. If the current post does not match that key, the
replay fails loudly instead of attaching a stale verdict to the wrong
claim. Legacy entries that predate cache keys remain loadable and are
marked ``cache_key_verified=False``.
"""

from __future__ import annotations

import json

from ..config import CACHE_PATH, DETECTOR_POLICY_VERSION
from .base import (
    DetectionResult,
    DetectorCacheMiss,
    make_cache_key,
    normalise_verdict,
)


class ReplayDetector:
    def __init__(
        self,
        cache_path=CACHE_PATH,
        expected_provider=None,
        expected_model=None,
    ):
        with open(cache_path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
        self._meta = payload.get("meta", {})
        self._entries = payload.get("verdicts", {})
        self._expected_provider = expected_provider
        self._expected_model = expected_model

    def _entry_for(self, post_id):
        entry = self._entries.get(str(post_id))
        if entry is None:
            entry = self._entries.get(post_id)
        return entry

    def _verify_key(self, entry, text, language, source_url):
        if not entry.get("cache_key"):
            return False

        stored_version = entry.get(
            "policy_version", DETECTOR_POLICY_VERSION
        )
        if stored_version != DETECTOR_POLICY_VERSION:
            raise DetectorCacheMiss(
                "cached verdict uses detector policy "
                f"{stored_version!r}, current policy is "
                f"{DETECTOR_POLICY_VERSION!r}"
            )

        expected = make_cache_key(
            text,
            language,
            source_url,
            DETECTOR_POLICY_VERSION,
        )
        if entry["cache_key"] != expected:
            raise DetectorCacheMiss(
                "cached verdict does not match the current claim, "
                "language, or source URL"
            )
        return True

    def _verify_provider(self, entry, post_id):
        if self._expected_provider is not None:
            provider = entry.get("provider")
            if provider != self._expected_provider:
                raise DetectorCacheMiss(
                    f"post {post_id} cached with provider {provider!r}, "
                    f"expected {self._expected_provider!r}"
                )
        if self._expected_model is not None:
            model = entry.get("model")
            if model != self._expected_model:
                raise DetectorCacheMiss(
                    f"post {post_id} cached with model {model!r}, "
                    f"expected {self._expected_model!r}"
                )

    @staticmethod
    def _verify_health(entry, post_id):
        record = entry.get("record") or {}
        status = record.get("provider_status") or {}
        if status.get("search_error"):
            raise DetectorCacheMiss(
                f"post {post_id} cache records a search failure "
                f"({status['search_error']}); rerun precompute"
            )
        if status.get("factcheck_error"):
            raise DetectorCacheMiss(
                f"post {post_id} cache records a fact-check failure "
                f"({status['factcheck_error']}); rerun precompute"
            )
        factcheck_status = status.get("factcheck_status")
        if isinstance(factcheck_status, int) and factcheck_status >= 500:
            raise DetectorCacheMiss(
                f"post {post_id} cache records HTTP {factcheck_status} "
                f"from Fact Check; rerun precompute"
            )

    def detect(self, text, language, post_id, source_url=None):
        entry = self._entry_for(post_id)
        if entry is None:
            raise DetectorCacheMiss(f"no cached verdict for post {post_id}")

        self._verify_provider(entry, post_id)
        self._verify_health(entry, post_id)
        verified = self._verify_key(entry, text, language, source_url)
        record = entry.get("record")
        if record:
            result = DetectionResult.from_dict(record)
            result.detector_mode = "replay"
            result.provider = entry.get("provider", result.provider)
            result.post_id = post_id
            result.claim_text = text
            result.language = language
            if source_url is not None:
                result.source_url = source_url
            result.cache_key_verified = verified
            return result

        # Legacy cache entry: mapped verdict only.
        mapped = entry.get("mapped", {})
        verdict = normalise_verdict(mapped.get("verdict"))
        is_fake = mapped.get("is_fake")
        if verdict is None:
            if is_fake is True:
                verdict = "FAKE"
            elif is_fake is False:
                verdict = "REAL"
            else:
                verdict = "UNCERTAIN"

        verdict_summary = mapped.get("verdict_summary")
        status = "ok"
        if verdict_summary == "NO_CLAIMS":
            status = "no_claims"
            verdict_summary = "NO_CLAIMS"

        return DetectionResult(
            is_fake=is_fake,
            confidence=float(mapped.get("confidence") or 0.0),
            verdict_summary=verdict_summary or "UNCERTAIN",
            provider=entry.get("provider", "replay"),
            verdict=verdict,
            detector_mode="replay",
            post_id=post_id,
            claim_text=text,
            language=language,
            source_url=source_url,
            status=status,
            cache_key_verified=verified,
        )
