"""User-facing verdict policy.

The verifier produces fine-grained internal verdicts (``SUPPORTED``,
``REFUTED``, ``NOT_ENOUGH_EVIDENCE``, ``CONFLICTING``). API and CLI callers
receive exactly one of three verdicts:

- ``REAL``      strong, directly relevant evidence corroborates the claim,
                or an authoritative fact-check confirms it.
- ``FAKE``      a Tier 4 blacklist matches, or reliable evidence directly
                contradicts the claim.
- ``UNCERTAIN`` evidence is insufficient, irrelevant, conflicting,
                unavailable, or below the configured confidence threshold.

This module never invents evidence and never changes the evidence the
verifier saw. It only decides whether that evidence is strong enough to
present a REAL/FAKE verdict to a user.
"""

from __future__ import annotations

from dataclasses import dataclass

import structlog

from ..config import settings
from ..models.enums import Stance, UserVerdict, Verdict
from ..models.schemas import (
    ClaimMatchResult,
    RankedEvidence,
    RerankedEvidenceResult,
    VerificationResult,
)

logger = structlog.get_logger()

REASON_CORROBORATED = "EVIDENCE_CORROBORATES_CLAIM"
REASON_CONTRADICTED = "EVIDENCE_CONTRADICTS_CLAIM"
REASON_INSUFFICIENT = "INSUFFICIENT_OR_WEAK_EVIDENCE"
REASON_CONFLICTING = "CONFLICTING_EVIDENCE"
REASON_NO_EVIDENCE = "NO_EVIDENCE"
REASON_LOW_CONFIDENCE = "LOW_CONFIDENCE"
REASON_SUBCLAIM_CONFLICT = "SUBCLAIM_CONFLICT"
REASON_STANCE_CONFLICT = "EVIDENCE_STANCE_CONFLICT"
REASON_WEAK_CACHED_VERDICT = "WEAK_CACHED_VERDICT"
REASON_MATCHED_FACT_CHECK = "MATCHED_AUTHORITATIVE_FACT_CHECK"
REASON_MATCHED_EXISTING = "MATCHED_EXISTING_VERDICT"


@dataclass(frozen=True)
class VerdictDecision:
    """Outcome of applying the user-facing verdict policy."""

    verdict: UserVerdict
    reason: str
    internal_verdict: Verdict | None = None
    support_score: float | None = None


@dataclass(frozen=True)
class EvidenceAnalysis:
    """Evidence-quality features behind an evidence-based verdict."""

    support_score: float
    usable_doc_count: int
    subclaim_alignment: float
    stance_alignment: float
    alignment_gate_passed: bool
    subclaim_conflict: bool
    opposite_stance_conflict: bool


def map_internal_to_user_verdict(value: Verdict | str | None) -> UserVerdict:
    """Map a stored internal verdict to the three user-facing values.

    Used for history rows written before the user-facing policy existed.
    ``NOT_ENOUGH_EVIDENCE`` and ``CONFLICTING`` both become ``UNCERTAIN``.
    """
    if value is None:
        return UserVerdict.UNCERTAIN

    if isinstance(value, Verdict):
        internal = value
    else:
        try:
            internal = Verdict(value.upper())
        except (AttributeError, ValueError):
            return UserVerdict.UNCERTAIN

    if internal == Verdict.SUPPORTED:
        return UserVerdict.REAL
    if internal == Verdict.REFUTED:
        return UserVerdict.FAKE
    return UserVerdict.UNCERTAIN


def decide_user_verdict(
    verification: VerificationResult,
    evidence: RerankedEvidenceResult,
    confidence: float,
    strength_threshold: float | None = None,
    confidence_floor: float | None = None,
) -> VerdictDecision:
    """Map an internal verification result to REAL/FAKE/UNCERTAIN.

    The LLM verifier is the primary arbiter. NLI stance is a secondary
    signal: it is often NEUTRAL for paraphrased or indirect evidence, so it
    must not be the sole gate. The policy requires credible/relevant
    documents plus either LLM subclaim alignment or NLI stance alignment,
    and rejects direct conflicts from either signal.
    """
    internal = verification.verdict
    threshold = (
        settings.verdict_evidence_strength_threshold
        if strength_threshold is None
        else strength_threshold
    )
    floor = (
        settings.verdict_confidence_floor
        if confidence_floor is None
        else confidence_floor
    )

    if internal == Verdict.CONFLICTING:
        return VerdictDecision(
            verdict=UserVerdict.UNCERTAIN,
            reason=REASON_CONFLICTING,
            internal_verdict=internal,
            support_score=0.0,
        )

    if internal == Verdict.NOT_ENOUGH_EVIDENCE:
        reason = REASON_NO_EVIDENCE
        if evidence.ranked_evidence:
            reason = REASON_INSUFFICIENT
        return VerdictDecision(
            verdict=UserVerdict.UNCERTAIN,
            reason=reason,
            internal_verdict=internal,
            support_score=0.0,
        )

    if internal not in (Verdict.SUPPORTED, Verdict.REFUTED):
        return VerdictDecision(
            verdict=UserVerdict.UNCERTAIN,
            reason=REASON_INSUFFICIENT,
            internal_verdict=internal,
            support_score=0.0,
        )

    target_stance = Stance.SUPPORTS
    if internal == Verdict.REFUTED:
        target_stance = Stance.REFUTES

    analysis = _analyze_evidence(verification, evidence, target_stance)

    logger.info(
        "verdict_policy.analysis",
        claim_id=verification.claim_id,
        internal_verdict=internal.value,
        confidence=confidence,
        support_score=analysis.support_score,
        usable_docs=analysis.usable_doc_count,
        subclaim_alignment=analysis.subclaim_alignment,
        stance_alignment=analysis.stance_alignment,
        alignment_gate=analysis.alignment_gate_passed,
        subclaim_conflict=analysis.subclaim_conflict,
        stance_conflict=analysis.opposite_stance_conflict,
    )

    if confidence < floor:
        return VerdictDecision(
            verdict=UserVerdict.UNCERTAIN,
            reason=REASON_LOW_CONFIDENCE,
            internal_verdict=internal,
            support_score=analysis.support_score,
        )

    if analysis.opposite_stance_conflict:
        return VerdictDecision(
            verdict=UserVerdict.UNCERTAIN,
            reason=REASON_STANCE_CONFLICT,
            internal_verdict=internal,
            support_score=analysis.support_score,
        )

    if analysis.subclaim_conflict:
        return VerdictDecision(
            verdict=UserVerdict.UNCERTAIN,
            reason=REASON_SUBCLAIM_CONFLICT,
            internal_verdict=internal,
            support_score=analysis.support_score,
        )

    if not analysis.alignment_gate_passed:
        return VerdictDecision(
            verdict=UserVerdict.UNCERTAIN,
            reason=REASON_INSUFFICIENT,
            internal_verdict=internal,
            support_score=analysis.support_score,
        )

    if analysis.support_score < threshold:
        return VerdictDecision(
            verdict=UserVerdict.UNCERTAIN,
            reason=REASON_INSUFFICIENT,
            internal_verdict=internal,
            support_score=analysis.support_score,
        )

    if internal == Verdict.SUPPORTED:
        return VerdictDecision(
            verdict=UserVerdict.REAL,
            reason=REASON_CORROBORATED,
            internal_verdict=internal,
            support_score=analysis.support_score,
        )

    return VerdictDecision(
        verdict=UserVerdict.FAKE,
        reason=REASON_CONTRADICTED,
        internal_verdict=internal,
        support_score=analysis.support_score,
    )


def decide_cached_verdict(match: ClaimMatchResult) -> VerdictDecision:
    """Map a fast-path match to a user-facing verdict.

    Fact-check API matches are authoritative only because ``match_claim``
    already required a semantic similarity at or above the configured
    threshold. Vector-cache matches reuse the previous internal verdict and
    its confidence; a low-confidence cache entry stays UNCERTAIN.
    """
    previous = match.previous_verdict
    if previous is None:
        return VerdictDecision(
            verdict=UserVerdict.UNCERTAIN,
            reason=REASON_INSUFFICIENT,
            internal_verdict=None,
        )

    if match.similarity is not None:
        if match.similarity < settings.claim_similarity_threshold:
            return VerdictDecision(
                verdict=UserVerdict.UNCERTAIN,
                reason=REASON_WEAK_CACHED_VERDICT,
                internal_verdict=previous,
            )

    if match.match_source == "factcheck_api":
        if previous == Verdict.SUPPORTED:
            return VerdictDecision(
                verdict=UserVerdict.REAL,
                reason=REASON_MATCHED_FACT_CHECK,
                internal_verdict=previous,
            )
        if previous == Verdict.REFUTED:
            return VerdictDecision(
                verdict=UserVerdict.FAKE,
                reason=REASON_MATCHED_FACT_CHECK,
                internal_verdict=previous,
            )
        return VerdictDecision(
            verdict=UserVerdict.UNCERTAIN,
            reason=REASON_INSUFFICIENT,
            internal_verdict=previous,
        )

    if match.previous_confidence is not None:
        threshold = settings.verdict_real_min_confidence
        if previous == Verdict.REFUTED:
            threshold = settings.verdict_fake_min_confidence
        if match.previous_confidence < threshold:
            return VerdictDecision(
                verdict=UserVerdict.UNCERTAIN,
                reason=REASON_WEAK_CACHED_VERDICT,
                internal_verdict=previous,
            )

    if previous == Verdict.SUPPORTED:
        return VerdictDecision(
            verdict=UserVerdict.REAL,
            reason=REASON_MATCHED_EXISTING,
            internal_verdict=previous,
        )
    if previous == Verdict.REFUTED:
        return VerdictDecision(
            verdict=UserVerdict.FAKE,
            reason=REASON_MATCHED_EXISTING,
            internal_verdict=previous,
        )

    return VerdictDecision(
        verdict=UserVerdict.UNCERTAIN,
        reason=REASON_INSUFFICIENT,
        internal_verdict=previous,
    )


def _analyze_evidence(
    verification: VerificationResult,
    evidence: RerankedEvidenceResult,
    target_stance: Stance,
) -> EvidenceAnalysis:
    """Score document quality, LLM subclaim alignment, and NLI stance.

    Source credibility is a prior on the publisher, not proof that this
    article supports the claim. The article-level signals are relevance,
    the verifier's subclaim verdicts, and NLI stance. NLI is deliberately a
    smaller component because it often returns NEUTRAL for paraphrases.
    """
    usable: list[RankedEvidence] = []
    for document in evidence.ranked_evidence:
        if document.credibility_score < settings.verdict_min_source_credibility:
            continue
        if document.relevance_score < settings.verdict_min_relevance:
            continue
        usable.append(document)

    stance_by_doc = {
        item.doc_id: item.stance for item in verification.supporting_evidence
    }
    opposite_stance = Stance.REFUTES
    if target_stance == Stance.REFUTES:
        opposite_stance = Stance.SUPPORTS

    matching_stances = 0
    opposite_stances = 0
    for document in usable:
        stance = stance_by_doc.get(document.doc_id, Stance.NEUTRAL)
        if stance == target_stance:
            matching_stances += 1
        elif stance == opposite_stance:
            opposite_stances += 1

    stance_alignment = 0.0
    if usable:
        stance_alignment = matching_stances / len(usable)

    doc_score = 0.0
    if len(usable) >= settings.verdict_min_corroborating_sources:
        doc_score = 1.0
    elif len(usable) == 1:
        document = usable[0]
        if document.is_primary_source:
            doc_score = 1.0
        elif document.credibility_score >= settings.verdict_strong_source_credibility:
            doc_score = 1.0
        else:
            doc_score = 0.4

    subclaims = verification.subclaim_results
    total_subclaims = len(subclaims)
    supported = sum(1 for item in subclaims if item.verdict == Verdict.SUPPORTED)
    refuted = sum(1 for item in subclaims if item.verdict == Verdict.REFUTED)
    conflicting = sum(1 for item in subclaims if item.verdict == Verdict.CONFLICTING)

    if total_subclaims == 0:
        subclaim_alignment = 0.0
        subclaim_score = 0.5
        subclaim_conflict = False
    elif target_stance == Stance.SUPPORTS:
        subclaim_alignment = supported / total_subclaims
        subclaim_score = subclaim_alignment
        # A refuted or conflicting subclaim means the claim is not wholly
        # corroborated, even if the overall verifier said SUPPORTED.
        subclaim_conflict = refuted > 0 or conflicting > 0
    else:
        # The verifier prompt requires the first subclaim to be the
        # central assertion. If that assertion is refuted, auxiliary true
        # facts later in the list do not make the claim partly true.
        first_is_refuted = subclaims[0].verdict == Verdict.REFUTED
        if first_is_refuted:
            subclaim_alignment = 1.0
            subclaim_conflict = conflicting > 0
        else:
            subclaim_alignment = refuted / total_subclaims
            subclaim_conflict = True
        subclaim_score = subclaim_alignment

    alignment_gate_passed = (
        subclaim_alignment >= settings.verdict_subclaim_alignment_threshold
        or stance_alignment >= settings.verdict_stance_alignment_threshold
    )
    opposite_stance_conflict = (
        opposite_stances >= settings.verdict_stance_conflict_count
        and opposite_stances >= matching_stances
    )

    support_score = (
        0.55 * doc_score
        + 0.35 * subclaim_score
        + 0.10 * stance_alignment
    )
    support_score = max(0.0, min(1.0, support_score))

    return EvidenceAnalysis(
        support_score=round(support_score, 3),
        usable_doc_count=len(usable),
        subclaim_alignment=round(subclaim_alignment, 3),
        stance_alignment=round(stance_alignment, 3),
        alignment_gate_passed=alignment_gate_passed,
        subclaim_conflict=subclaim_conflict,
        opposite_stance_conflict=opposite_stance_conflict,
    )
