"""Step 3 — Claim Matching Fast Path.

Before expensive live verification, check if this claim (or a very similar one)
has already been verified. Searches Qdrant vector DB, PostgreSQL exact match,
and Google Fact Check API.
"""

from __future__ import annotations

import structlog

from ..models.schemas import NormalizedClaim, ClaimMatchResult
from ..models.enums import Verdict
from ..services.vector_service import vector_service
from ..services.factcheck_service import factcheck_service
from ..config import settings
from .claim_equivalence import check_material_equivalence
from ..utils.redaction import redact_secrets

logger = structlog.get_logger()

# Map common fact-check ratings to our verdict system
RATING_MAP = {
    "true": Verdict.SUPPORTED,
    "mostly true": Verdict.SUPPORTED,
    "correct": Verdict.SUPPORTED,
    "accurate": Verdict.SUPPORTED,
    "false": Verdict.REFUTED,
    "mostly false": Verdict.REFUTED,
    "incorrect": Verdict.REFUTED,
    "pants on fire": Verdict.REFUTED,
    "fake": Verdict.REFUTED,
    "misleading": Verdict.REFUTED,
    "mixture": Verdict.CONFLICTING,
    "mixed": Verdict.CONFLICTING,
    "half true": Verdict.CONFLICTING,
    "partly false": Verdict.CONFLICTING,
    "unproven": Verdict.NOT_ENOUGH_EVIDENCE,
    "unverified": Verdict.NOT_ENOUGH_EVIDENCE,
}


async def match_claim(claim: NormalizedClaim) -> ClaimMatchResult:
    """
    Check if a similar claim already exists in our database or external fact-checks.

    Order of checks:
    1. Qdrant vector similarity search (fastest for semantic matches)
    2. Google Fact Check API (external ClaimReview lookups)

    Returns a ClaimMatchResult indicating whether a match was found.
    """
    logger.info("claim_matcher.start", claim_id=claim.claim_id)
    claim_text = claim.normalized_claim or claim.original_text

    # ── 1. Vector Similarity Search ───────────────────────────
    try:
        vector_matches = await vector_service.search_similar(
            claim_text=claim_text,
            top_k=3,
            score_threshold=settings.claim_similarity_threshold,
        )

        if vector_matches:
            for match in vector_matches:
                if match.get("policy_version") != settings.verdict_policy_version:
                    logger.info(
                        "claim_matcher.vector_policy_version_rejected",
                        claim_id=claim.claim_id,
                        matched_id=match.get("claim_id"),
                        cached_policy=match.get("policy_version"),
                        current_policy=settings.verdict_policy_version,
                    )
                    continue

                cached_claim_text = match.get("claim_text") or ""
                guard = check_material_equivalence(
                    claim_text,
                    cached_claim_text,
                )
                if not guard.equivalent:
                    logger.info(
                        "claim_matcher.vector_match_rejected",
                        claim_id=claim.claim_id,
                        similarity=match.get("score"),
                        matched_id=match.get("claim_id"),
                        matched_claim=cached_claim_text[:160],
                        reason=guard.reason,
                    )
                    continue

                prev_verdict = match.get("verdict")

                # Skip cached results with weak verdicts — re-verify them.
                if prev_verdict and str(prev_verdict).upper() == "NOT_ENOUGH_EVIDENCE":
                    logger.info(
                        "claim_matcher.skip_weak_cache",
                        claim_id=claim.claim_id,
                        similarity=match["score"],
                        previous_verdict=prev_verdict,
                        reason="Re-verifying stale NOT_ENOUGH_EVIDENCE result",
                    )
                    continue

                logger.info(
                    "claim_matcher.vector_match",
                    claim_id=claim.claim_id,
                    similarity=match["score"],
                    matched_id=match.get("claim_id"),
                )
                return ClaimMatchResult(
                    matched_existing_claim=True,
                    matched_claim_id=match.get("claim_id"),
                    match_source="vector_db",
                    similarity=match["score"],
                    previous_verdict=prev_verdict,
                    previous_confidence=match.get("confidence"),
                    matched_claim_text=cached_claim_text,
                )
    except Exception as e:
        logger.warning("claim_matcher.vector_error", error=redact_secrets(str(e)))

    # ── 2. Google Fact Check API ──────────────────────────────
    try:
        factchecks = await factcheck_service.search_claims(
            query=claim_text,
            max_results=5,
        )

        if factchecks:
            # Compute real semantic similarity between our claim and each
            # fact-check result.  The API returns topically-related claims,
            # NOT necessarily semantically-equivalent ones — so we must
            # verify with actual cosine similarity before accepting.
            import numpy as np

            try:
                claim_embedding = vector_service.encoder.encode(claim_text)
            except Exception:
                claim_embedding = None

            for fc in factchecks:
                fc_claim_text = fc.get("claim_text", "")
                rating = fc.get("rating", "").lower().strip()
                mapped_verdict = RATING_MAP.get(rating)

                if not mapped_verdict or not fc_claim_text:
                    continue

                # Compute cosine similarity if the encoder is available
                if claim_embedding is not None:
                    try:
                        fc_embedding = vector_service.encoder.encode(fc_claim_text)
                        cosine_sim = float(
                            np.dot(claim_embedding, fc_embedding)
                            / (np.linalg.norm(claim_embedding) * np.linalg.norm(fc_embedding) + 1e-10)
                        )
                    except Exception:
                        cosine_sim = 0.0
                else:
                    cosine_sim = 0.0

                logger.debug(
                    "claim_matcher.factcheck_candidate",
                    claim_id=claim.claim_id,
                    fc_claim=fc_claim_text[:100],
                    cosine_sim=round(cosine_sim, 3),
                    rating=rating,
                )

                if cosine_sim >= settings.claim_similarity_threshold:
                    guard = check_material_equivalence(
                        claim_text,
                        fc_claim_text,
                    )
                    if not guard.equivalent:
                        logger.info(
                            "claim_matcher.factcheck_material_mismatch",
                            claim_id=claim.claim_id,
                            similarity=round(cosine_sim, 3),
                            publisher=fc.get("publisher"),
                            matched_claim=fc_claim_text[:120],
                            reason=guard.reason,
                        )
                        continue

                    logger.info(
                        "claim_matcher.factcheck_match",
                        claim_id=claim.claim_id,
                        rating=rating,
                        verdict=mapped_verdict,
                        similarity=round(cosine_sim, 3),
                        publisher=fc.get("publisher"),
                        matched_claim=fc_claim_text[:120],
                    )
                    return ClaimMatchResult(
                        matched_existing_claim=True,
                        match_source="factcheck_api",
                        similarity=cosine_sim,
                        previous_verdict=mapped_verdict,
                        matched_claim_text=fc_claim_text,
                        external_url=fc.get("url"),
                    )

            logger.debug(
                "claim_matcher.factcheck_no_semantic_match",
                claim_id=claim.claim_id,
                candidates=len(factchecks),
            )
    except Exception as e:
        logger.warning("claim_matcher.factcheck_error", error=redact_secrets(str(e)))

    # ── No match found ────────────────────────────────────────
    logger.info("claim_matcher.no_match", claim_id=claim.claim_id)
    return ClaimMatchResult(matched_existing_claim=False)
