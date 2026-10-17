"""Shared detector interface: one adapter surface for every backend.

The whole containment layer talks to detectors through `detect(text,
language, post_id) -> DetectionResult`. The ground-truth label never
crosses this interface — it exists only for metrics.
"""

from dataclasses import dataclass


@dataclass
class DetectionResult:
    is_fake: bool | None  # None means uncertain -> no intervention
    confidence: float
    verdict_summary: str  # "REFUTED" / "SUPPORTED" / "UNCERTAIN" / "NO_CLAIMS"
    provider: str         # "stub" / "gemini" / "ollama" / "replay"


class DetectorUnavailableError(Exception):
    """The real detector cannot produce trustworthy verdicts."""


class DetectorCacheMiss(KeyError):
    """The replay cache has no entry for a post."""


def ensure_real_provider(provider: str) -> None:
    """Raise unless the LLM backend can produce trustworthy verdicts.

    The pipeline silently falls back to a mock provider when no LLM key
    is configured, and the mock always returns canned SUPPORTED claims.
    Trusting those verdicts would silently corrupt every experiment.
    """
    if provider == "mock":
        raise DetectorUnavailableError(
            "LLM provider is 'mock': verdicts would be canned and meaningless"
        )


def map_job_results(results, confidence_threshold, provider):
    """Map a list of claim dicts to one DetectionResult.

    `results` entries look like the model_dump() output of a claim:
    {"verdict": "REFUTED", "confidence": 0.87, ...}. A post is FAKE iff
    any claim is REFUTED with confidence >= threshold; REAL iff nothing is
    refuted and at least one claim is SUPPORTED with confidence >=
    threshold; otherwise UNCERTAIN (NOT_ENOUGH_EVIDENCE, CONFLICTING, or
    no checkable claims). Uncertain posts get no intervention.
    """
    refuted = [
        claim
        for claim in results
        if claim.get("verdict") == "REFUTED"
        and claim.get("confidence", 0.0) >= confidence_threshold
    ]
    supported = [
        claim
        for claim in results
        if claim.get("verdict") == "SUPPORTED"
        and claim.get("confidence", 0.0) >= confidence_threshold
    ]

    if refuted:
        best_confidence = max(claim["confidence"] for claim in refuted)
        return DetectionResult(
            is_fake=True,
            confidence=best_confidence,
            verdict_summary="REFUTED",
            provider=provider,
        )
    if supported:
        best_confidence = max(claim["confidence"] for claim in supported)
        return DetectionResult(
            is_fake=False,
            confidence=best_confidence,
            verdict_summary="SUPPORTED",
            provider=provider,
        )
    if not results:
        return DetectionResult(
            is_fake=None, confidence=0.0, verdict_summary="NO_CLAIMS", provider=provider
        )
    return DetectionResult(
        is_fake=None, confidence=0.0, verdict_summary="UNCERTAIN", provider=provider
    )
