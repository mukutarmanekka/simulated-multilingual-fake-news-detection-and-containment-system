"""Shared detector interface: one adapter surface for every backend.

The whole containment layer talks to detectors through
`detect(text, language, post_id, source_url=None) -> DetectionResult`.
The ground-truth label never crosses this interface — it exists only for
metrics.

The detector's public verdict is one of REAL / FAKE / UNCERTAIN. Execution
failures are represented by ``status`` in {"error", "timeout"} and
``verdict=None``; they are never silently converted into a definitive
verdict.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import asdict, dataclass, field, fields
from typing import Any
from urllib.parse import urlparse

from ..config import DETECTOR_POLICY_VERSION

VERDICT_REAL = "REAL"
VERDICT_FAKE = "FAKE"
VERDICT_UNCERTAIN = "UNCERTAIN"
VERDICTS = (VERDICT_REAL, VERDICT_FAKE, VERDICT_UNCERTAIN)

# Legacy internal verdicts still appear in old cached responses and tests.
REFUTING_VERDICTS = frozenset({"REFUTED", "FAKE"})
SUPPORTING_VERDICTS = frozenset({"SUPPORTED", "REAL"})


@dataclass
class DetectionResult:
    """Structured detection record for one simulated post.

    ``is_fake`` is the containment signal: True for FAKE, False for REAL,
    None for UNCERTAIN or an execution failure. ``verdict`` is the public
    three-way verdict and is None only when the detector did not complete.
    """

    is_fake: bool | None
    confidence: float
    verdict_summary: str  # legacy: "REFUTED" / "SUPPORTED" / "UNCERTAIN" / "NO_CLAIMS" / "ERROR"
    provider: str         # LLM provider or backend: "stub" / "deepseek" / "replay" ...
    verdict: str | None = None
    detector_mode: str = "unknown"  # "stub" / "replay" / "real"
    post_id: int | None = None
    claim_text: str | None = None
    language: str | None = None
    detected_language: str | None = None
    was_translated: bool = False
    source_url: str | None = None
    source_domain: str | None = None
    support_score: float | None = None
    reason: str | None = None
    internal_verdict: str | None = None
    evidence: list[dict] = field(default_factory=list)
    claims: list[dict] = field(default_factory=list)
    matched_existing_claim: bool = False
    blacklisted: bool = False
    blacklist_reason: str | None = None
    matched_blacklist_domain: str | None = None
    blacklist_category: str | None = None
    source_tier: str | None = None
    source_tier_number: int | None = None
    status: str = "ok"  # "ok" / "no_claims" / "error" / "timeout"
    error: str | None = None
    latency_ms: float | None = None
    provider_usage: dict | None = None
    provider_status: dict | None = None
    requires_human_review: bool = False
    escalation_reason: str | None = None
    cache_key_verified: bool | None = None

    def __post_init__(self):
        if self.verdict is not None:
            self.verdict = str(self.verdict).upper()
        elif self.status == "ok":
            if self.is_fake is True:
                self.verdict = VERDICT_FAKE
            elif self.is_fake is False:
                self.verdict = VERDICT_REAL
            else:
                self.verdict = VERDICT_UNCERTAIN

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict) -> "DetectionResult":
        known = {item.name for item in fields(cls)}
        return cls(**{key: value for key, value in payload.items() if key in known})


class DetectorUnavailableError(Exception):
    """The real detector cannot produce trustworthy verdicts."""


class DetectorCacheMiss(KeyError):
    """The replay cache has no valid entry for a post/claim."""


class DetectorExecutionError(RuntimeError):
    """The detector failed and the configured policy is to abort."""


def ensure_real_provider(provider: str) -> None:
    """Raise unless the LLM backend can produce trustworthy verdicts.

    The pipeline silently falls back to a mock provider when no LLM key
    is configured, and the mock always returns canned verdicts. Trusting
    those verdicts would silently corrupt every experiment.
    """
    if provider == "mock":
        raise DetectorUnavailableError(
            "LLM provider is 'mock': verdicts would be canned and meaningless"
        )


# ── Verdict and metadata helpers ────────────────────────────────────


def normalise_verdict(value: Any) -> str | None:
    """Map current or legacy verdict strings to REAL/FAKE/UNCERTAIN."""
    if value is None:
        return None
    text = str(value).strip().upper()
    if text in ("REAL", "SUPPORTED"):
        return VERDICT_REAL
    if text in ("FAKE", "REFUTED"):
        return VERDICT_FAKE
    if text in ("UNCERTAIN", "NOT_ENOUGH_EVIDENCE", "CONFLICTING", "NO_CLAIMS"):
        return VERDICT_UNCERTAIN
    return None


def legacy_summary(verdict: str | None) -> str:
    if verdict == VERDICT_FAKE:
        return "REFUTED"
    if verdict == VERDICT_REAL:
        return "SUPPORTED"
    if verdict == VERDICT_UNCERTAIN:
        return "UNCERTAIN"
    return "ERROR"


def _domain_from_url(url: str | None) -> str | None:
    if not url:
        return None
    try:
        host = urlparse(url).hostname
    except ValueError:
        return None
    if not host:
        return None
    host = host.lower().strip(".")
    if host.startswith("www."):
        host = host[4:]
    return host or None


_SECRET_PATTERNS = (
    (re.compile(r"sk-[A-Za-z0-9_-]{8,}"), "sk-***"),
    (re.compile(r"tvly-[A-Za-z0-9_-]{8,}"), "tvly-***"),
    (re.compile(r"AIza[0-9A-Za-z_-]{10,}"), "AIza***"),
    (
        re.compile(
            r"(?i)((?:api[_-]?key|apikey|access[_-]?token|token|key)=)[^&\s\"']+"
        ),
        r"\1***",
    ),
    (
        re.compile(r"(?i)((?:Authorization|X-Subscription-Token)\s*[:=]\s*)[^\r\n]+"),
        r"\1***",
    ),
    (
        re.compile(
            r"(?i)((?:https?|postgres(?:ql)?(?:\+asyncpg)?|redis)://[^:\s/@]+:)[^@\s]+@"
        ),
        r"\1***@",
    ),
)


def redact_error(value: Any) -> str:
    text = "" if value is None else str(value)
    for pattern, replacement in _SECRET_PATTERNS:
        text = pattern.sub(replacement, text)
    return text


def make_cache_key(
    text: str,
    language: str | None,
    source_url: str | None,
    policy_version: str = DETECTOR_POLICY_VERSION,
) -> str:
    """Key a cached verdict to the claim, language, source, and policy."""
    payload = "\x1f".join(
        [
            text or "",
            language or "",
            source_url or "",
            str(policy_version),
        ]
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def error_detection(
    *,
    post_id: int | None,
    text: str | None,
    language: str | None,
    source_url: str | None,
    provider: str,
    detector_mode: str,
    status: str,
    error: Any,
    latency_ms: float | None = None,
    provider_usage: dict | None = None,
    provider_status: dict | None = None,
) -> DetectionResult:
    return DetectionResult(
        is_fake=None,
        confidence=0.0,
        verdict_summary="ERROR",
        provider=provider,
        verdict=None,
        detector_mode=detector_mode,
        post_id=post_id,
        claim_text=text,
        language=language,
        source_url=source_url,
        source_domain=_domain_from_url(source_url),
        status=status,
        error=redact_error(error),
        latency_ms=latency_ms,
        provider_usage=provider_usage,
        provider_status=provider_status,
    )


def _select_claim(claims: list[dict]) -> dict | None:
    if not claims:
        return None

    def rank(claim: dict):
        verdict = normalise_verdict(claim.get("verdict"))
        priority = {
            VERDICT_FAKE: 2,
            VERDICT_REAL: 1,
            VERDICT_UNCERTAIN: 0,
        }.get(verdict, -1)
        support = claim.get("verdict_support_score") or 0.0
        confidence = claim.get("confidence") or 0.0
        return (priority, confidence, support)

    return max(claims, key=rank)


def map_response_dict(
    response: Any,
    *,
    post_id: int,
    text: str,
    language: str,
    source_url: str | None,
    provider: str,
    detector_mode: str = "real",
    latency_ms: float | None = None,
    provider_usage: dict | None = None,
    provider_status: dict | None = None,
) -> DetectionResult:
    """Map a detector job response into the simulation's record.

    This is a pure schema mapping. It does not apply a second confidence
    threshold: the detector's REAL/FAKE/UNCERTAIN verdict is authoritative.
    """
    if hasattr(response, "model_dump"):
        response = response.model_dump(mode="json")
    if not isinstance(response, dict):
        return error_detection(
            post_id=post_id,
            text=text,
            language=language,
            source_url=source_url,
            provider=provider,
            detector_mode=detector_mode,
            status="error",
            error=f"unexpected detector response type: {type(response).__name__}",
            latency_ms=latency_ms,
            provider_usage=provider_usage,
            provider_status=provider_status,
        )

    results = [
        claim for claim in response.get("results", []) if isinstance(claim, dict)
    ]
    selected = _select_claim(results)

    if selected is None:
        return DetectionResult(
            is_fake=None,
            confidence=0.0,
            verdict_summary="NO_CLAIMS",
            provider=provider,
            verdict=VERDICT_UNCERTAIN,
            detector_mode=detector_mode,
            post_id=post_id,
            claim_text=text,
            language=language,
            detected_language=response.get("detected_language") or language,
            was_translated=bool(response.get("was_translated")),
            source_url=source_url,
            source_domain=_domain_from_url(source_url),
            reason="NO_CHECKABLE_CLAIMS",
            status="no_claims",
            latency_ms=latency_ms,
            provider_usage=provider_usage,
            provider_status=provider_status,
        )

    verdict = normalise_verdict(selected.get("verdict")) or VERDICT_UNCERTAIN
    is_fake = None
    if verdict == VERDICT_FAKE:
        is_fake = True
    elif verdict == VERDICT_REAL:
        is_fake = False

    reason = selected.get("verdict_reason")
    matched_domain = selected.get("matched_blacklist_domain")
    blacklisted = bool(matched_domain) or bool(
        reason and str(reason).startswith("BLACKLISTED_")
    )

    resolved_source_url = selected.get("source_url") or source_url
    resolved_domain = (
        selected.get("source_domain") or _domain_from_url(resolved_source_url)
    )

    return DetectionResult(
        is_fake=is_fake,
        confidence=float(selected.get("confidence") or 0.0),
        verdict_summary=legacy_summary(verdict),
        provider=provider,
        verdict=verdict,
        detector_mode=detector_mode,
        post_id=post_id,
        claim_text=selected.get("claim") or text,
        language=language,
        detected_language=response.get("detected_language") or language,
        was_translated=bool(response.get("was_translated")),
        source_url=resolved_source_url,
        source_domain=resolved_domain,
        support_score=selected.get("verdict_support_score"),
        reason=reason,
        internal_verdict=selected.get("evidence_verdict"),
        evidence=list(selected.get("evidence") or []),
        claims=results,
        matched_existing_claim=bool(selected.get("matched_existing_claim")),
        blacklisted=blacklisted,
        blacklist_reason=reason if blacklisted else None,
        matched_blacklist_domain=matched_domain,
        blacklist_category=selected.get("blacklist_category"),
        source_tier=selected.get("source_tier"),
        source_tier_number=selected.get("source_tier_number"),
        status="ok",
        latency_ms=latency_ms,
        provider_usage=provider_usage,
        provider_status=provider_status,
        requires_human_review=bool(selected.get("requires_human_review")),
        escalation_reason=selected.get("escalation_reason"),
    )


def map_job_results(results, confidence_threshold, provider):
    """Legacy threshold mapping used by old caches and unit tests.

    Real-mode detection uses ``map_response_dict`` and preserves the
    detector's verdict without a second threshold.
    """
    refuted = [
        claim
        for claim in results
        if claim.get("verdict") in REFUTING_VERDICTS
        and claim.get("confidence", 0.0) >= confidence_threshold
    ]
    supported = [
        claim
        for claim in results
        if claim.get("verdict") in SUPPORTING_VERDICTS
        and claim.get("confidence", 0.0) >= confidence_threshold
    ]

    if refuted:
        best_confidence = max(claim["confidence"] for claim in refuted)
        return DetectionResult(
            is_fake=True,
            confidence=best_confidence,
            verdict_summary="REFUTED",
            provider=provider,
            verdict=VERDICT_FAKE,
            status="ok",
        )
    if supported:
        best_confidence = max(claim["confidence"] for claim in supported)
        return DetectionResult(
            is_fake=False,
            confidence=best_confidence,
            verdict_summary="SUPPORTED",
            provider=provider,
            verdict=VERDICT_REAL,
            status="ok",
        )
    if not results:
        return DetectionResult(
            is_fake=None,
            confidence=0.0,
            verdict_summary="NO_CLAIMS",
            provider=provider,
            verdict=VERDICT_UNCERTAIN,
            status="no_claims",
        )
    return DetectionResult(
        is_fake=None,
        confidence=0.0,
        verdict_summary="UNCERTAIN",
        provider=provider,
        verdict=VERDICT_UNCERTAIN,
        status="ok",
    )
