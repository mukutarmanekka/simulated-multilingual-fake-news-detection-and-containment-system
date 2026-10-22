import pytest
from src.pipeline.escalation import check_escalation
from src.models.schemas import VerificationResult, RerankedEvidenceResult, RankedEvidence, EvidenceStance
from src.models.enums import Verdict, Stance


@pytest.mark.asyncio
async def test_escalation_low_confidence():
    verification = VerificationResult(
        claim_id="c1",
        verdict=Verdict.NOT_ENOUGH_EVIDENCE,
        confidence=0.40,
        reasoning=["Low evidence"],
        subclaim_results=[],
        supporting_evidence=[],
    )
    evidence = RerankedEvidenceResult(claim_id="c1", ranked_evidence=[])
    decision = await check_escalation("A random company changed its logo", verification, evidence)
    assert decision.requires_human_review is True
    assert decision.reason is not None
    assert "Low confidence" in decision.reason


@pytest.mark.asyncio
async def test_escalation_high_impact_health_domain():
    verification = VerificationResult(
        claim_id="c2",
        verdict=Verdict.SUPPORTED,
        confidence=0.72,  # < 0.80 triggers high impact escalation
        reasoning=["Some evidence"],
        subclaim_results=[],
        supporting_evidence=[
            EvidenceStance(doc_id="d1", snippet="...", stance=Stance.SUPPORTS, confidence=0.7)
        ],
    )
    evidence = RerankedEvidenceResult(
        claim_id="c2",
        ranked_evidence=[
            RankedEvidence(
                doc_id="d1",
                title="Vaccine Study",
                url="https://health.gov",
                source_domain="health.gov",
                published_date="2026-01-01",
                credibility_score=0.9,
                relevance_score=0.9,
                freshness_score=0.9,
                final_rank_score=0.9,
                passages=["Vaccine data"],
            )
        ]
    )
    decision = await check_escalation("A new COVID vaccine has been approved by the FDA", verification, evidence)
    assert decision.requires_human_review is True
    assert "health" in decision.high_impact_domains


@pytest.mark.asyncio
async def test_no_escalation_high_confidence_benign_claim():
    verification = VerificationResult(
        claim_id="c3",
        verdict=Verdict.SUPPORTED,
        confidence=0.92,
        reasoning=["Strong evidence"],
        subclaim_results=[],
        supporting_evidence=[
            EvidenceStance(doc_id="d1", snippet="...", stance=Stance.SUPPORTS, confidence=0.95)
        ],
    )
    evidence = RerankedEvidenceResult(
        claim_id="c3",
        ranked_evidence=[
            RankedEvidence(
                doc_id="d1",
                title="Paris Guide",
                url="https://paris.fr",
                source_domain="paris.fr",
                published_date="2026-01-01",
                credibility_score=0.9,
                relevance_score=0.9,
                freshness_score=0.9,
                final_rank_score=0.9,
                passages=["Eiffel tower is in Paris"],
            )
        ]
    )
    decision = await check_escalation("The Eiffel Tower is located in Paris", verification, evidence)
    assert decision.requires_human_review is False
    assert decision.reason is None
