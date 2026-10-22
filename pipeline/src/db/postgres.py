"""PostgreSQL database operations using SQLAlchemy async.

Lazy-initializes the DB engine so the app can start even without PostgreSQL.
All operations fail gracefully when the database is unavailable.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any, Optional

import structlog

from ..utils.redaction import redact_secrets

logger = structlog.get_logger()

# Lazy-initialized engine and session
_engine = None
_async_session = None
_db_available = None  # None = not yet checked, True/False after first attempt


def _get_engine():
    """Lazy-initialize the SQLAlchemy async engine."""
    global _engine
    if _engine is None:
        from sqlalchemy.ext.asyncio import create_async_engine
        from ..config import settings
        _engine = create_async_engine(
            settings.database_url,
            echo=False,
            pool_size=5,
            max_overflow=10,
            pool_pre_ping=True,
        )
    return _engine


def _get_session_factory():
    """Lazy-initialize the async session factory."""
    global _async_session
    if _async_session is None:
        from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker
        _async_session = async_sessionmaker(
            _get_engine(), class_=AsyncSession, expire_on_commit=False
        )
    return _async_session


async def _check_db():
    """Check if the database is reachable, cache the result."""
    global _db_available
    if _db_available is not None:
        return _db_available
    try:
        from sqlalchemy import text
        factory = _get_session_factory()
        async with factory() as session:
            await session.execute(text("SELECT 1"))
        _db_available = True
        logger.info("db.connected")
    except Exception as e:
        _db_available = False
        logger.warning("db.unavailable", error=redact_secrets(str(e)),
                       msg="PostgreSQL is not available — DB operations will be skipped")
    return _db_available


async def get_session():
    """Yield an async database session."""
    factory = _get_session_factory()
    async with factory() as session:
        yield session


class ClaimDB:
    """Database operations for claims and verification results.

    All methods fail gracefully when PostgreSQL is unavailable,
    returning sensible defaults instead of crashing the pipeline.
    """

    @staticmethod
    async def create_job(job_id: str, input_text: str, input_type: str) -> str:
        """Create a verification job and return its ID."""
        if not await _check_db():
            return job_id
        try:
            from sqlalchemy import text
            factory = _get_session_factory()
            async with factory() as session:
                await session.execute(
                    text("""
                        INSERT INTO verification_jobs (id, input_text, input_type, status, created_at)
                        VALUES (:id, :input_text, :input_type, 'processing', :now)
                    """),
                    {
                        "id": job_id,
                        "input_text": input_text[:5000],
                        "input_type": input_type,
                        "now": datetime.now(timezone.utc),
                    },
                )
                await session.commit()
        except Exception as e:
            logger.warning("db.create_job.error", error=redact_secrets(str(e)))
        return job_id

    @staticmethod
    async def update_job_status(
        job_id: str,
        status: str,
        total_claims: int = 0,
        results: Any = None,
        error: str | None = None,
    ) -> None:
        """Update a verification job's status."""
        if not await _check_db():
            return
        try:
            from sqlalchemy import text
            factory = _get_session_factory()
            async with factory() as session:
                params: dict[str, Any] = {
                    "id": job_id,
                    "status": status,
                    "total_claims": total_claims,
                    "now": datetime.now(timezone.utc),
                }
                extra_sets = ""
                if results is not None:
                    import json
                    extra_sets += ", results = :results"
                    params["results"] = json.dumps(results)
                if error is not None:
                    extra_sets += ", error = :error"
                    params["error"] = error
                if status in ("completed", "failed"):
                    extra_sets += ", completed_at = :now"

                await session.execute(
                    text(f"""
                        UPDATE verification_jobs
                        SET status = :status,
                            total_claims = :total_claims
                            {extra_sets}
                        WHERE id = :id
                    """),
                    params,
                )
                await session.commit()
        except Exception as e:
            logger.warning("db.update_job.error", error=redact_secrets(str(e)))

    @staticmethod
    async def store_claim(
        claim_id: str,
        input_text: str,
        input_type: str,
        claim_text: str,
        claim_index: int,
        normalized_text: str | None = None,
        entities: list | None = None,
        dates: list | None = None,
        numbers: list | None = None,
        resolved_entities: list | None = None,
        resolved_dates: list | None = None,
        is_checkable: bool = True,
        detected_language: str | None = None,
        was_translated: bool = False,
        original_text: str | None = None,
    ) -> None:
        """Store an extracted claim."""
        if not await _check_db():
            return
        try:
            import json
            from sqlalchemy import text
            factory = _get_session_factory()
            async with factory() as session:
                await session.execute(
                    text("""
                        INSERT INTO claims (
                            id, input_text, input_type, claim_text, claim_index,
                            normalized_text, entities, dates, numbers,
                            resolved_entities, resolved_dates, is_checkable,
                            detected_language, was_translated, original_text,
                            created_at
                        ) VALUES (
                            :id, :input_text, :input_type, :claim_text, :claim_index,
                            :normalized_text, :entities, :dates, :numbers,
                            :resolved_entities, :resolved_dates, :is_checkable,
                            :detected_language, :was_translated, :original_text,
                            :now
                        )
                    """),
                    {
                        "id": claim_id,
                        "input_text": input_text[:5000],
                        "input_type": input_type,
                        "claim_text": claim_text,
                        "claim_index": claim_index,
                        "normalized_text": normalized_text,
                        "entities": json.dumps(entities or []),
                        "dates": json.dumps(dates or []),
                        "numbers": json.dumps(numbers or []),
                        "resolved_entities": json.dumps(resolved_entities or []),
                        "resolved_dates": json.dumps(resolved_dates or []),
                        "is_checkable": is_checkable,
                        "detected_language": detected_language,
                        "was_translated": was_translated,
                        "original_text": original_text[:5000] if original_text else None,
                        "now": datetime.now(timezone.utc),
                    },
                )
                await session.commit()
        except Exception as e:
            logger.warning("db.store_claim.error", error=redact_secrets(str(e)))

    @staticmethod
    async def store_verification(
        claim_id: str,
        verdict: str,
        confidence: float,
        reasoning: list,
        subclaim_results: list | None = None,
        matched_existing: bool = False,
        matched_claim_id: str | None = None,
        pipeline_duration_ms: int | None = None,
        model_used: str | None = None,
    ) -> str:
        """Store a verification result and return its ID."""
        verification_id = str(uuid.uuid4())
        if not await _check_db():
            return verification_id
        try:
            import json
            from sqlalchemy import text
            factory = _get_session_factory()
            async with factory() as session:
                await session.execute(
                    text("""
                        INSERT INTO verification_results (
                            id, claim_id, verdict, confidence, reasoning,
                            subclaim_results, matched_existing, matched_claim_id,
                            pipeline_duration_ms, model_used, created_at
                        ) VALUES (
                            :id, :claim_id, :verdict, :confidence, :reasoning,
                            :subclaim_results, :matched_existing, :matched_claim_id,
                            :pipeline_duration_ms, :model_used, :now
                        )
                    """),
                    {
                        "id": verification_id,
                        "claim_id": claim_id,
                        "verdict": verdict,
                        "confidence": confidence,
                        "reasoning": json.dumps(reasoning),
                        "subclaim_results": json.dumps(subclaim_results or []),
                        "matched_existing": matched_existing,
                        "matched_claim_id": matched_claim_id,
                        "pipeline_duration_ms": pipeline_duration_ms,
                        "model_used": model_used,
                        "now": datetime.now(timezone.utc),
                    },
                )
                await session.commit()
        except Exception as e:
            logger.warning("db.store_verification.error", error=redact_secrets(str(e)))
        return verification_id

    @staticmethod
    async def store_evidence(
        claim_id: str,
        title: str,
        url: str,
        source_domain: str | None,
        published_date: str | None,
        credibility_score: float,
        relevance_score: float,
        freshness_score: float,
        final_rank_score: float,
        passages: list,
        stance: str | None,
        is_primary_source: bool,
        retrieval_query: str | None,
    ) -> str:
        """Store a piece of evidence and return its ID."""
        evidence_id = str(uuid.uuid4())
        if not await _check_db():
            return evidence_id
        try:
            import json
            from sqlalchemy import text
            factory = _get_session_factory()
            async with factory() as session:
                await session.execute(
                    text("""
                        INSERT INTO evidence (
                            id, claim_id, title, url, source_domain,
                            published_date, credibility_score, relevance_score,
                            freshness_score, final_rank_score, passages,
                            stance, is_primary_source, retrieval_query, created_at
                        ) VALUES (
                            :id, :claim_id, :title, :url, :source_domain,
                            :published_date, :credibility_score, :relevance_score,
                            :freshness_score, :final_rank_score, :passages,
                            :stance, :is_primary_source, :retrieval_query, :now
                        )
                    """),
                    {
                        "id": evidence_id,
                        "claim_id": claim_id,
                        "title": title,
                        "url": url,
                        "source_domain": source_domain,
                        "published_date": published_date,
                        "credibility_score": credibility_score,
                        "relevance_score": relevance_score,
                        "freshness_score": freshness_score,
                        "final_rank_score": final_rank_score,
                        "passages": json.dumps(passages),
                        "stance": stance,
                        "is_primary_source": is_primary_source,
                        "retrieval_query": retrieval_query,
                        "now": datetime.now(timezone.utc),
                    },
                )
                await session.commit()
        except Exception as e:
            logger.warning("db.store_evidence.error", error=redact_secrets(str(e)))
        return evidence_id

    @staticmethod
    async def create_escalation(
        claim_id: str,
        verification_id: str,
        reason: str,
        priority: int = 5,
    ) -> str:
        """Create a human review escalation."""
        escalation_id = str(uuid.uuid4())
        if not await _check_db():
            return escalation_id
        try:
            from sqlalchemy import text
            factory = _get_session_factory()
            async with factory() as session:
                await session.execute(
                    text("""
                        INSERT INTO human_reviews (
                            id, claim_id, verification_id, status,
                            escalation_reason, priority, created_at
                        ) VALUES (
                            :id, :claim_id, :verification_id, 'PENDING',
                            :reason, :priority, :now
                        )
                    """),
                    {
                        "id": escalation_id,
                        "claim_id": claim_id,
                        "verification_id": verification_id,
                        "reason": reason,
                        "priority": priority,
                        "now": datetime.now(timezone.utc),
                    },
                )
                await session.commit()
        except Exception as e:
            logger.warning("db.create_escalation.error", error=redact_secrets(str(e)))
        return escalation_id

    @staticmethod
    async def log_audit(
        claim_id: str | None,
        step_name: str,
        step_index: int,
        input_data: Any = None,
        output_data: Any = None,
        duration_ms: int | None = None,
        error: str | None = None,
    ) -> None:
        """Write an audit log entry for a pipeline step."""
        if not await _check_db():
            return
        try:
            import json
            from sqlalchemy import text
            factory = _get_session_factory()
            async with factory() as session:
                await session.execute(
                    text("""
                        INSERT INTO audit_log (
                            id, claim_id, step_name, step_index,
                            input_data, output_data, duration_ms, error, created_at
                        ) VALUES (
                            :id, :claim_id, :step_name, :step_index,
                            :input_data, :output_data, :duration_ms, :error, :now
                        )
                    """),
                    {
                        "id": str(uuid.uuid4()),
                        "claim_id": claim_id,
                        "step_name": step_name,
                        "step_index": step_index,
                        "input_data": json.dumps(input_data) if input_data else None,
                        "output_data": json.dumps(output_data) if output_data else None,
                        "duration_ms": duration_ms,
                        "error": error,
                        "now": datetime.now(timezone.utc),
                    },
                )
                await session.commit()
        except Exception:
            pass  # Don't log audit failures

    @staticmethod
    async def get_source_credibility(domain: str) -> Optional[dict]:
        """Look up a source's credibility by domain."""
        if not await _check_db():
            return None
        try:
            from sqlalchemy import text
            factory = _get_session_factory()
            async with factory() as session:
                result = await session.execute(
                    text("SELECT domain, name, tier, credibility FROM sources WHERE domain = :domain"),
                    {"domain": domain},
                )
                row = result.fetchone()
                if row:
                    return {
                        "domain": row[0],
                        "name": row[1],
                        "tier": row[2],
                        "credibility": row[3],
                    }
        except Exception as e:
            logger.debug("db.get_source.error", error=redact_secrets(str(e)))
        return None

    @staticmethod
    async def get_job(job_id: str) -> Optional[dict]:
        """Get a verification job by ID."""
        if not await _check_db():
            return None
        try:
            from sqlalchemy import text
            factory = _get_session_factory()
            async with factory() as session:
                result = await session.execute(
                    text("SELECT * FROM verification_jobs WHERE id = :id"),
                    {"id": job_id},
                )
                row = result.fetchone()
                if row:
                    return dict(row._mapping)
        except Exception as e:
            logger.debug("db.get_job.error", error=redact_secrets(str(e)))
        return None

    @staticmethod
    async def get_pending_reviews(limit: int = 20) -> list[dict]:
        """Get pending human reviews, ordered by priority."""
        if not await _check_db():
            return []
        try:
            from sqlalchemy import text
            factory = _get_session_factory()
            async with factory() as session:
                result = await session.execute(
                    text("""
                        SELECT hr.*, c.claim_text, c.normalized_text,
                               vr.verdict, vr.confidence, vr.reasoning
                        FROM human_reviews hr
                        JOIN claims c ON c.id = hr.claim_id
                        JOIN verification_results vr ON vr.id = hr.verification_id
                        WHERE hr.status = 'PENDING'
                        ORDER BY hr.priority DESC, hr.created_at ASC
                        LIMIT :limit
                    """),
                    {"limit": limit},
                )
                return [dict(row._mapping) for row in result.fetchall()]
        except Exception as e:
            logger.debug("db.get_reviews.error", error=redact_secrets(str(e)))
        return []

    @staticmethod
    async def get_job_results(job_id: str) -> list[dict]:
        """Get verification results for a specific job ID."""
        if not await _check_db():
            return []
        try:
            from sqlalchemy import text
            factory = _get_session_factory()
            async with factory() as session:
                # We need to find claims associated with this job.
                # In this schema, claims don't have a job_id directly, 
                # but verification_jobs has a results JSONB or we can match by input_text?
                # Actually, the orchestrator stores results in verification_results table.
                # Wait, the current schema doesn't link verification_results to jobs!
                # It links to claims. And claims don't link to jobs.
                # This is a major schema gap if we want to track by job_id.
                
                # Let's check verification_jobs table again.
                # It has a 'results' column (JSONB).
                result = await session.execute(
                    text("SELECT results FROM verification_jobs WHERE id = :id"),
                    {"id": job_id},
                )
                row = result.fetchone()
                if row and row[0]:
                    import json
                    if isinstance(row[0], str):
                        return json.loads(row[0])
                    return row[0]
                
                # Fallback: if results column is empty, try to find claims with the same created_at? 
                # No, that's unreliable.
                # The orchestrator SHOULD update the results column in verification_jobs.
                # Let's check orchestrator.py again.
        except Exception as e:
            logger.debug("db.get_job_results.error", error=redact_secrets(str(e)))
        return []

    @staticmethod
    async def get_recent_claims(limit: int = 50) -> list[dict]:
        """Get recent verification results with claim data."""
        if not await _check_db():
            return []
        try:
            from sqlalchemy import text
            factory = _get_session_factory()
            async with factory() as session:
                result = await session.execute(
                    text("""
                        SELECT c.id, c.claim_text, c.normalized_text, c.created_at,
                               vr.verdict, vr.confidence, vr.reasoning,
                               vr.pipeline_duration_ms, vr.matched_existing
                        FROM claims c
                        LEFT JOIN verification_results vr ON vr.claim_id = c.id
                        WHERE c.is_checkable = TRUE
                        ORDER BY c.created_at DESC
                        LIMIT :limit
                    """),
                    {"limit": limit},
                )
                return [dict(row._mapping) for row in result.fetchall()]
        except Exception as e:
            logger.debug("db.get_claims.error", error=redact_secrets(str(e)))
        return []

