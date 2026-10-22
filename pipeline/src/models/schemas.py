"""Pydantic schemas for the fake news detection pipeline.

Defines the data contracts for every stage of the verification pipeline,
ensuring type safety and enforcing the exact JSON output formats required.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Optional

from pydantic import AliasChoices, BaseModel, Field

from .enums import InputType, ReviewStatus, Stance, UserVerdict, Verdict


# ── Step 1: Claim Extraction ─────────────────────────────────────────


class ExtractedClaim(BaseModel):
    """A single factual claim extracted from input text."""
    claim_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    text: str
    entities: list[str] = Field(default_factory=list)
    dates: list[str] = Field(default_factory=list)
    numbers: list[str] = Field(default_factory=list)
    is_checkable: bool = True


class ClaimExtractionResult(BaseModel):
    """Output of the claim extraction step."""
    claims: list[ExtractedClaim] = Field(default_factory=list)
    non_checkable_statements: list[str] = Field(default_factory=list)


# ── Step 2: Claim Normalization ──────────────────────────────────────


class NormalizedClaim(BaseModel):
    """A claim after normalization — resolved entities, dates, etc."""
    claim_id: str
    original_text: str
    normalized_claim: str
    resolved_entities: list[dict] = Field(default_factory=list)
    resolved_dates: list[str] = Field(default_factory=list)
    resolved_numbers: list[str] = Field(default_factory=list)


# ── Step 3: Claim Matching ───────────────────────────────────────────


class ClaimMatchResult(BaseModel):
    """Result of the fast-path claim matching step."""
    matched_existing_claim: bool = False
    matched_claim_id: Optional[str] = None
    match_source: Optional[str] = None  # 'vector_db', 'exact', 'factcheck_api'
    similarity: Optional[float] = None
    previous_verdict: Optional[Verdict] = None
    previous_confidence: Optional[float] = None
    matched_claim_text: Optional[str] = None
    external_url: Optional[str] = None


# ── Step 4: Evidence Retrieval ───────────────────────────────────────


class RetrievedDocument(BaseModel):
    """A single document retrieved as evidence."""
    doc_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    title: str
    url: str
    source_domain: Optional[str] = None
    published_date: Optional[str] = None
    credibility_score: float = 0.40
    passages: list[str] = Field(default_factory=list)
    retrieval_query: Optional[str] = None
    is_primary_source: bool = False


class EvidenceRetrievalResult(BaseModel):
    """Output of the evidence retrieval step."""
    claim_id: str
    search_queries: list[str] = Field(default_factory=list)
    retrieved_documents: list[RetrievedDocument] = Field(default_factory=list)


# ── Step 5: Evidence Re-ranking ──────────────────────────────────────


class RankedEvidence(BaseModel):
    """An evidence document after re-ranking."""
    doc_id: str
    title: str
    url: str
    source_domain: Optional[str] = None
    published_date: Optional[str] = None
    credibility_score: float = 0.40
    relevance_score: float = 0.0
    freshness_score: float = 0.0
    final_rank_score: float = 0.0
    passages: list[str] = Field(default_factory=list)
    is_primary_source: bool = False


class RerankedEvidenceResult(BaseModel):
    """Output of the evidence re-ranking step."""
    claim_id: str
    ranked_evidence: list[RankedEvidence] = Field(default_factory=list)


# ── Step 6: Claim Verification ───────────────────────────────────────


class EvidenceStance(BaseModel):
    """Stance of a single evidence item toward a claim."""
    doc_id: str
    snippet: str
    stance: Stance
    confidence: float = 0.0


class SubclaimResult(BaseModel):
    """Verification result for an atomic subclaim."""
    subclaim_text: str
    verdict: Verdict
    supporting_evidence: list[EvidenceStance] = Field(default_factory=list)
    reasoning: str = ""


class VerificationResult(BaseModel):
    """Full verification output for a single claim."""
    claim_id: str
    verdict: Verdict
    confidence: float
    reasoning: list[str] = Field(default_factory=list)
    subclaim_results: list[SubclaimResult] = Field(default_factory=list)
    supporting_evidence: list[EvidenceStance] = Field(default_factory=list)


# ── Step 7: Confidence Estimation ────────────────────────────────────


class ConfidenceBreakdown(BaseModel):
    """Detailed breakdown of how confidence was calculated."""
    source_agreement_score: float = 0.0
    evidence_quality_score: float = 0.0
    freshness_score: float = 0.0
    consistency_score: float = 0.0
    retrieval_certainty: float = 0.0
    contradiction_penalty: float = 0.0
    final_confidence: float = 0.0


# ── Step 8: Human Escalation ─────────────────────────────────────────


class EscalationDecision(BaseModel):
    """Decision on whether to escalate to human review."""
    requires_human_review: bool = False
    reason: Optional[str] = None
    priority: int = 5  # 1-10, 10 = highest
    high_impact_domains: list[str] = Field(default_factory=list)


# ── Final Pipeline Response ──────────────────────────────────────────


class EvidenceItem(BaseModel):
    """Evidence item in the final response."""
    source: str
    url: str
    published_date: Optional[str] = None
    snippet: str
    stance: Stance
    credibility_score: Optional[float] = None
    relevance_score: Optional[float] = None


class FinalClaimResponse(BaseModel):
    """Final response for a single claim verification."""
    claim: str
    normalized_claim: str
    verdict: UserVerdict
    confidence: float
    reasoning: list[str] = Field(default_factory=list)
    evidence: list[EvidenceItem] = Field(default_factory=list)
    matched_existing_claim: bool = False
    requires_human_review: bool = False
    escalation_reason: Optional[str] = None
    claim_id: Optional[str] = None
    pipeline_duration_ms: Optional[int] = None
    # ── Internal evidence verdict (preserved for audit/debugging) ──
    evidence_verdict: Optional[Verdict] = None
    verdict_reason: Optional[str] = None
    verdict_support_score: Optional[float] = None
    # ── Source assessment metadata ───────────────────────────
    source_url: Optional[str] = None
    source_domain: Optional[str] = None
    source_tier: Optional[str] = None
    source_tier_number: Optional[int] = None
    matched_blacklist_domain: Optional[str] = None
    blacklist_category: Optional[str] = None
    # ── Language metadata ─────────────────────────────────────
    detected_language: Optional[str] = None       # ISO 639-1 code
    language_name: Optional[str] = None           # Human-readable
    was_translated: bool = False
    original_text: Optional[str] = None           # Pre-translation text


class VerificationJobResponse(BaseModel):
    """Full response for a verification job (may contain multiple claims)."""
    job_id: str
    status: str = "completed"
    input_text: str
    input_type: InputType
    total_claims: int = 0
    results: list[FinalClaimResponse] = Field(default_factory=list)
    non_checkable_statements: list[str] = Field(default_factory=list)
    created_at: str = Field(default_factory=lambda: datetime.utcnow().isoformat())
    completed_at: Optional[str] = None
    total_duration_ms: Optional[int] = None
    source_url: Optional[str] = None
    # ── Language metadata ─────────────────────────────────────
    detected_language: Optional[str] = None
    language_name: Optional[str] = None
    was_translated: bool = False
    original_input_text: Optional[str] = None


# ── API Request Models ───────────────────────────────────────────────


class VerifyRequest(BaseModel):
    """Request to verify a piece of content."""
    text: str = Field(..., min_length=10, max_length=50000)
    input_type: InputType = InputType.ARTICLE
    job_id: Optional[str] = None
    source_url: Optional[str] = Field(
        default=None,
        validation_alias=AliasChoices("source_url", "url"),
        description=(
            "Optional URL or domain of the publisher that supplied the text. "
            "Used for the Tier 4 blacklist short-circuit. The URL is never "
            "inferred from URLs merely mentioned inside the text."
        ),
    )


class HumanReviewRequest(BaseModel):
    """Human reviewer submitting a review decision."""
    reviewer_id: str
    verdict: Verdict
    confidence: float = Field(..., ge=0, le=1)
    notes: str = ""


# Fix forward reference
SubclaimResult.model_rebuild()
