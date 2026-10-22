"""Step 5 — Evidence Re-ranking.

Ranks retrieved evidence by semantic relevance, freshness,
source quality, and claim coverage. Uses a cross-encoder model
for accurate relevance scoring.
"""

from __future__ import annotations

from typing import Any, cast

import structlog

from ..models.schemas import (
    RetrievedDocument,
    EvidenceRetrievalResult,
    RankedEvidence,
    RerankedEvidenceResult,
)
from ..utils.temporal import freshness_score, is_time_sensitive
from ..utils.redaction import redact_secrets
from ..config import settings

logger = structlog.get_logger()

# Lazy-loaded cross-encoder
_reranker = None


def _get_reranker():
    """Lazy-load the cross-encoder re-ranking model."""
    global _reranker
    if _reranker is None:
        from sentence_transformers import CrossEncoder
        logger.info("reranker.loading", model=settings.reranker_model)
        _reranker = CrossEncoder(settings.reranker_model)
        logger.info("reranker.loaded")
    return _reranker


async def rerank_evidence(
    claim_text: str,
    evidence: EvidenceRetrievalResult,
) -> RerankedEvidenceResult:
    """
    Re-rank evidence documents for a claim.

    Scoring factors:
    - Semantic relevance (cross-encoder)    weight: 0.40
    - Source credibility                     weight: 0.25
    - Freshness                              weight: 0.20
    - Primary source bonus                   weight: 0.15

    Time-sensitive claims boost freshness weight.
    """
    logger.info(
        "evidence_reranker.start",
        claim_id=evidence.claim_id,
        documents=len(evidence.retrieved_documents),
    )

    docs = evidence.retrieved_documents
    if not docs:
        return RerankedEvidenceResult(claim_id=evidence.claim_id, ranked_evidence=[])

    time_sensitive = is_time_sensitive(claim_text)

    # ── Semantic relevance via cross-encoder ──────────────────
    relevance_scores = _compute_relevance(claim_text, docs)

    # ── Build ranked evidence with composite scores ───────────
    ranked = []
    for i, doc in enumerate(docs):
        rel_score = relevance_scores[i] if i < len(relevance_scores) else 0.0
        fresh_score = freshness_score(doc.published_date)
        cred_score = doc.credibility_score
        primary_bonus = 1.0 if doc.is_primary_source else 0.0

        # Adjust weights for time-sensitive claims
        if time_sensitive:
            w_rel, w_cred, w_fresh, w_primary = 0.30, 0.20, 0.35, 0.15
        else:
            w_rel, w_cred, w_fresh, w_primary = 0.40, 0.25, 0.20, 0.15

        final_score = (
            w_rel * rel_score
            + w_cred * cred_score
            + w_fresh * fresh_score
            + w_primary * primary_bonus
        )

        ranked.append(RankedEvidence(
            doc_id=doc.doc_id,
            title=doc.title,
            url=doc.url,
            source_domain=doc.source_domain,
            published_date=doc.published_date,
            credibility_score=cred_score,
            relevance_score=round(rel_score, 3),
            freshness_score=round(fresh_score, 3),
            final_rank_score=round(final_score, 3),
            passages=doc.passages,
            is_primary_source=doc.is_primary_source,
        ))

    # Sort by final score descending
    ranked.sort(key=lambda x: x.final_rank_score, reverse=True)

    # Keep top N
    top_n = ranked[:settings.top_evidence_count]

    logger.info(
        "evidence_reranker.done",
        claim_id=evidence.claim_id,
        top_scores=[r.final_rank_score for r in top_n],
    )

    return RerankedEvidenceResult(
        claim_id=evidence.claim_id,
        ranked_evidence=top_n,
    )


def _compute_relevance(claim_text: str, docs: list[RetrievedDocument]) -> list[float]:
    """Compute semantic relevance scores using cross-encoder."""
    try:
        reranker = _get_reranker()

        # Build pairs: (claim, document_text)
        pairs = []
        for doc in docs:
            doc_text = doc.title
            if doc.passages:
                doc_text += " " + " ".join(doc.passages[:2])
            pairs.append((claim_text, doc_text[:512]))

        if not pairs:
            return []

        scores = reranker.predict(cast(Any, pairs))

        # Normalize scores to 0-1 range (cross-encoder outputs can be negative)
        scores_list = scores.tolist() if hasattr(scores, 'tolist') else list(scores)
        min_s = min(scores_list) if scores_list else 0
        max_s = max(scores_list) if scores_list else 1
        range_s = max_s - min_s if max_s != min_s else 1

        normalized = [(s - min_s) / range_s for s in scores_list]
        return normalized

    except Exception as e:
        logger.warning("reranker.model_error", error=redact_secrets(str(e)))
        # Fallback: use credibility as a proxy for relevance
        return [doc.credibility_score for doc in docs]
