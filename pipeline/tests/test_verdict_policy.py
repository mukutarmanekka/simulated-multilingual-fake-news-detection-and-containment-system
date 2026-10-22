import pytest

from src.models.enums import Stance, UserVerdict, Verdict
from src.models.schemas import (
    ClaimMatchResult,
    EvidenceStance,
    RankedEvidence,
    RerankedEvidenceResult,
    SubclaimResult,
    VerificationResult,
)
from src.pipeline.verdict_policy import (
    REASON_CONFLICTING,
    REASON_CONTRADICTED,
    REASON_CORROBORATED,
    REASON_INSUFFICIENT,
    REASON_LOW_CONFIDENCE,
    REASON_MATCHED_FACT_CHECK,
    REASON_NO_EVIDENCE,
    REASON_STANCE_CONFLICT,
    REASON_SUBCLAIM_CONFLICT,
    REASON_WEAK_CACHED_VERDICT,
    decide_cached_verdict,
    decide_user_verdict,
    map_internal_to_user_verdict,
)


def make_document(
    doc_id: str,
    credibility: float = 0.90,
    relevance: float = 0.90,
    is_primary: bool = False,
) -> RankedEvidence:
    return RankedEvidence(
        doc_id=doc_id,
        title=f"Evidence {doc_id}",
        url=f"https://{doc_id}.example/article",
        source_domain=f"{doc_id}.example",
        published_date="2026-01-01",
        credibility_score=credibility,
        relevance_score=relevance,
        freshness_score=0.80,
        final_rank_score=0.85,
        passages=["A directly relevant passage."],
        is_primary_source=is_primary,
    )


def make_verification(
    verdict: Verdict,
    stances: list[tuple[str, Stance]],
    confidence: float = 0.0,
    subclaims: list[Verdict] | None = None,
) -> VerificationResult:
    return VerificationResult(
        claim_id="claim-1",
        verdict=verdict,
        confidence=confidence,
        reasoning=["Test verification."],
        subclaim_results=[
            SubclaimResult(
                subclaim_text=f"subclaim-{index}",
                verdict=subclaim_verdict,
            )
            for index, subclaim_verdict in enumerate(subclaims or [])
        ],
        supporting_evidence=[
            EvidenceStance(
                doc_id=doc_id,
                snippet="A directly relevant passage.",
                stance=stance,
                confidence=0.90,
            )
            for doc_id, stance in stances
        ],
    )


def make_evidence(documents: list[RankedEvidence]) -> RerankedEvidenceResult:
    return RerankedEvidenceResult(
        claim_id="claim-1",
        ranked_evidence=documents,
    )


# ── REAL ─────────────────────────────────────────────────────────────


def test_strong_corroborating_evidence_produces_real():
    documents = [
        make_document("doc-1"),
        make_document("doc-2"),
    ]
    verification = make_verification(
        Verdict.SUPPORTED,
        [("doc-1", Stance.SUPPORTS), ("doc-2", Stance.SUPPORTS)],
    )

    decision = decide_user_verdict(
        verification,
        make_evidence(documents),
        confidence=0.85,
    )

    assert decision.verdict == UserVerdict.REAL
    assert decision.reason == REASON_CORROBORATED
    assert decision.internal_verdict == Verdict.SUPPORTED


def test_single_primary_source_can_produce_real():
    documents = [make_document("doc-1", credibility=0.72, is_primary=True)]
    verification = make_verification(
        Verdict.SUPPORTED,
        [("doc-1", Stance.SUPPORTS)],
    )

    decision = decide_user_verdict(
        verification,
        make_evidence(documents),
        confidence=0.85,
    )

    assert decision.verdict == UserVerdict.REAL


def test_single_arbitrary_source_is_not_enough_for_real():
    documents = [make_document("doc-1", credibility=0.62)]
    verification = make_verification(
        Verdict.SUPPORTED,
        [("doc-1", Stance.SUPPORTS)],
    )

    decision = decide_user_verdict(
        verification,
        make_evidence(documents),
        confidence=0.85,
    )

    assert decision.verdict == UserVerdict.UNCERTAIN
    assert decision.reason == REASON_INSUFFICIENT


def test_low_confidence_support_stays_uncertain():
    documents = [
        make_document("doc-1"),
        make_document("doc-2"),
    ]
    verification = make_verification(
        Verdict.SUPPORTED,
        [("doc-1", Stance.SUPPORTS), ("doc-2", Stance.SUPPORTS)],
    )

    decision = decide_user_verdict(
        verification,
        make_evidence(documents),
        confidence=0.40,
    )

    assert decision.verdict == UserVerdict.UNCERTAIN


def test_irrelevant_support_does_not_produce_real():
    documents = [
        make_document("doc-1", relevance=0.10),
        make_document("doc-2", relevance=0.10),
    ]
    verification = make_verification(
        Verdict.SUPPORTED,
        [("doc-1", Stance.SUPPORTS), ("doc-2", Stance.SUPPORTS)],
    )

    decision = decide_user_verdict(
        verification,
        make_evidence(documents),
        confidence=0.85,
    )

    assert decision.verdict == UserVerdict.UNCERTAIN


def test_blacklisted_evidence_cannot_corroborate_real():
    documents = [
        make_document("doc-1", credibility=0.0),
        make_document("doc-2", credibility=0.0),
    ]
    verification = make_verification(
        Verdict.SUPPORTED,
        [("doc-1", Stance.SUPPORTS), ("doc-2", Stance.SUPPORTS)],
    )

    decision = decide_user_verdict(
        verification,
        make_evidence(documents),
        confidence=0.85,
    )

    assert decision.verdict == UserVerdict.UNCERTAIN


# ── FAKE ─────────────────────────────────────────────────────────────


def test_strong_contradictory_evidence_produces_fake():
    documents = [
        make_document("doc-1"),
        make_document("doc-2"),
    ]
    verification = make_verification(
        Verdict.REFUTED,
        [("doc-1", Stance.REFUTES), ("doc-2", Stance.REFUTES)],
    )

    decision = decide_user_verdict(
        verification,
        make_evidence(documents),
        confidence=0.85,
    )

    assert decision.verdict == UserVerdict.FAKE
    assert decision.reason == REASON_CONTRADICTED
    assert decision.internal_verdict == Verdict.REFUTED


def test_single_strong_refutation_produces_fake():
    documents = [make_document("doc-1", credibility=0.92)]
    verification = make_verification(
        Verdict.REFUTED,
        [("doc-1", Stance.REFUTES)],
    )

    decision = decide_user_verdict(
        verification,
        make_evidence(documents),
        confidence=0.80,
    )

    assert decision.verdict == UserVerdict.FAKE


# ── UNCERTAIN ────────────────────────────────────────────────────────


def test_insufficient_evidence_stays_uncertain():
    verification = make_verification(
        Verdict.NOT_ENOUGH_EVIDENCE,
        [],
    )

    decision = decide_user_verdict(
        verification,
        make_evidence([]),
        confidence=0.15,
    )

    assert decision.verdict == UserVerdict.UNCERTAIN
    assert decision.reason == REASON_NO_EVIDENCE


def test_conflicting_evidence_stays_uncertain():
    documents = [
        make_document("doc-1"),
        make_document("doc-2"),
    ]
    verification = make_verification(
        Verdict.CONFLICTING,
        [("doc-1", Stance.SUPPORTS), ("doc-2", Stance.REFUTES)],
    )

    decision = decide_user_verdict(
        verification,
        make_evidence(documents),
        confidence=0.65,
    )

    assert decision.verdict == UserVerdict.UNCERTAIN
    assert decision.reason == REASON_CONFLICTING


def test_neutral_stances_do_not_produce_real_or_fake():
    documents = [make_document("doc-1"), make_document("doc-2")]
    verification = make_verification(
        Verdict.SUPPORTED,
        [("doc-1", Stance.NEUTRAL), ("doc-2", Stance.NEUTRAL)],
    )

    decision = decide_user_verdict(
        verification,
        make_evidence(documents),
        confidence=0.85,
    )

    assert decision.verdict == UserVerdict.UNCERTAIN


# ── Fast-path cached verdicts ────────────────────────────────────────


def test_authoritative_fact_check_can_confirm_real():
    match = ClaimMatchResult(
        matched_existing_claim=True,
        match_source="factcheck_api",
        similarity=0.95,
        previous_verdict=Verdict.SUPPORTED,
    )

    decision = decide_cached_verdict(match)

    assert decision.verdict == UserVerdict.REAL
    assert decision.reason == REASON_MATCHED_FACT_CHECK


def test_authoritative_fact_check_can_refute_fake():
    match = ClaimMatchResult(
        matched_existing_claim=True,
        match_source="factcheck_api",
        similarity=0.95,
        previous_verdict=Verdict.REFUTED,
    )

    decision = decide_cached_verdict(match)

    assert decision.verdict == UserVerdict.FAKE
    assert decision.reason == REASON_MATCHED_FACT_CHECK


def test_low_similarity_fact_check_is_not_authoritative():
    match = ClaimMatchResult(
        matched_existing_claim=True,
        match_source="factcheck_api",
        similarity=0.40,
        previous_verdict=Verdict.REFUTED,
    )

    decision = decide_cached_verdict(match)

    assert decision.verdict == UserVerdict.UNCERTAIN
    assert decision.reason == REASON_WEAK_CACHED_VERDICT


def test_low_confidence_vector_cache_stays_uncertain():
    match = ClaimMatchResult(
        matched_existing_claim=True,
        match_source="vector_db",
        similarity=0.95,
        previous_verdict=Verdict.REFUTED,
        previous_confidence=0.30,
    )

    decision = decide_cached_verdict(match)

    assert decision.verdict == UserVerdict.UNCERTAIN
    assert decision.reason == REASON_WEAK_CACHED_VERDICT


def test_high_confidence_vector_cache_reuses_verdict():
    match = ClaimMatchResult(
        matched_existing_claim=True,
        match_source="vector_db",
        similarity=0.95,
        previous_verdict=Verdict.SUPPORTED,
        previous_confidence=0.90,
    )

    decision = decide_cached_verdict(match)

    assert decision.verdict == UserVerdict.REAL


def test_stored_internal_verdicts_map_to_three_user_verdicts():
    assert map_internal_to_user_verdict("SUPPORTED") == UserVerdict.REAL
    assert map_internal_to_user_verdict(Verdict.SUPPORTED) == UserVerdict.REAL
    assert map_internal_to_user_verdict("REFUTED") == UserVerdict.FAKE
    assert map_internal_to_user_verdict(Verdict.REFUTED) == UserVerdict.FAKE
    assert map_internal_to_user_verdict("NOT_ENOUGH_EVIDENCE") == UserVerdict.UNCERTAIN
    assert map_internal_to_user_verdict("CONFLICTING") == UserVerdict.UNCERTAIN
    assert map_internal_to_user_verdict(None) == UserVerdict.UNCERTAIN
    assert map_internal_to_user_verdict("UNKNOWN_VALUE") == UserVerdict.UNCERTAIN


# ── Subclaim and evidence-strength policy ───────────────────────────


def test_subclaim_support_overrides_neutral_nli():
    documents = [
        make_document("doc-1", credibility=0.90),
        make_document("doc-2", credibility=0.70),
    ]
    verification = make_verification(
        Verdict.SUPPORTED,
        [("doc-1", Stance.NEUTRAL), ("doc-2", Stance.NEUTRAL)],
        subclaims=[Verdict.SUPPORTED, Verdict.SUPPORTED],
    )

    decision = decide_user_verdict(
        verification,
        make_evidence(documents),
        confidence=0.50,
    )

    assert decision.verdict == UserVerdict.REAL
    assert decision.reason == REASON_CORROBORATED
    assert decision.support_score == pytest.approx(0.90)


def test_subclaim_refutation_overrides_neutral_nli():
    documents = [
        make_document("doc-1", credibility=0.72),
        make_document("doc-2", credibility=0.65),
        make_document("doc-3", credibility=0.65),
    ]
    verification = make_verification(
        Verdict.REFUTED,
        [
            ("doc-1", Stance.NEUTRAL),
            ("doc-2", Stance.NEUTRAL),
            ("doc-3", Stance.NEUTRAL),
        ],
        subclaims=[Verdict.REFUTED, Verdict.REFUTED, Verdict.SUPPORTED],
    )

    decision = decide_user_verdict(
        verification,
        make_evidence(documents),
        confidence=0.48,
    )

    assert decision.verdict == UserVerdict.FAKE
    assert decision.reason == REASON_CONTRADICTED
    assert decision.support_score == pytest.approx(0.90)


def test_conflicting_subclaim_blocks_real():
    documents = [make_document("doc-1"), make_document("doc-2")]
    verification = make_verification(
        Verdict.SUPPORTED,
        [("doc-1", Stance.SUPPORTS), ("doc-2", Stance.SUPPORTS)],
        subclaims=[Verdict.SUPPORTED, Verdict.REFUTED],
    )

    decision = decide_user_verdict(
        verification,
        make_evidence(documents),
        confidence=0.80,
    )

    assert decision.verdict == UserVerdict.UNCERTAIN
    assert decision.reason == REASON_SUBCLAIM_CONFLICT


def test_low_confidence_floor_blocks_strong_evidence():
    documents = [make_document("doc-1"), make_document("doc-2")]
    verification = make_verification(
        Verdict.SUPPORTED,
        [("doc-1", Stance.NEUTRAL), ("doc-2", Stance.NEUTRAL)],
        subclaims=[Verdict.SUPPORTED],
    )

    decision = decide_user_verdict(
        verification,
        make_evidence(documents),
        confidence=0.44,
    )

    assert decision.verdict == UserVerdict.UNCERTAIN
    assert decision.reason == REASON_LOW_CONFIDENCE


def test_opposite_nli_stances_block_support():
    documents = [make_document("doc-1"), make_document("doc-2")]
    verification = make_verification(
        Verdict.SUPPORTED,
        [("doc-1", Stance.REFUTES), ("doc-2", Stance.REFUTES)],
        subclaims=[Verdict.SUPPORTED],
    )

    decision = decide_user_verdict(
        verification,
        make_evidence(documents),
        confidence=0.70,
    )

    assert decision.verdict == UserVerdict.UNCERTAIN
    assert decision.reason == REASON_STANCE_CONFLICT


def test_partial_subclaim_alignment_stays_uncertain():
    documents = [make_document("doc-1"), make_document("doc-2")]
    verification = make_verification(
        Verdict.SUPPORTED,
        [("doc-1", Stance.NEUTRAL), ("doc-2", Stance.NEUTRAL)],
        subclaims=[Verdict.SUPPORTED, Verdict.NOT_ENOUGH_EVIDENCE],
    )

    decision = decide_user_verdict(
        verification,
        make_evidence(documents),
        confidence=0.70,
    )

    assert decision.verdict == UserVerdict.UNCERTAIN
    assert decision.reason == REASON_INSUFFICIENT


def test_single_moderate_source_stays_uncertain():
    documents = [make_document("doc-1", credibility=0.79)]
    verification = make_verification(
        Verdict.SUPPORTED,
        [("doc-1", Stance.SUPPORTS)],
        subclaims=[Verdict.SUPPORTED],
    )

    decision = decide_user_verdict(
        verification,
        make_evidence(documents),
        confidence=0.70,
    )

    assert decision.verdict == UserVerdict.UNCERTAIN
    assert decision.support_score == pytest.approx(0.67)


def test_single_strong_source_produces_real():
    documents = [make_document("doc-1", credibility=0.80)]
    verification = make_verification(
        Verdict.SUPPORTED,
        [("doc-1", Stance.NEUTRAL)],
        subclaims=[Verdict.SUPPORTED],
    )

    decision = decide_user_verdict(
        verification,
        make_evidence(documents),
        confidence=0.72,
    )

    assert decision.verdict == UserVerdict.REAL


def test_primary_moderate_source_produces_real():
    documents = [
        make_document("doc-1", credibility=0.70, is_primary=True)
    ]
    verification = make_verification(
        Verdict.SUPPORTED,
        [("doc-1", Stance.NEUTRAL)],
        subclaims=[Verdict.SUPPORTED],
    )

    decision = decide_user_verdict(
        verification,
        make_evidence(documents),
        confidence=0.68,
    )

    assert decision.verdict == UserVerdict.REAL
