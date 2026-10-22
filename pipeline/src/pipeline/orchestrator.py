

from __future__ import annotations

import asyncio
import time
import traceback
import uuid
from datetime import datetime, timezone

import structlog


from ..models.schemas import (
    FinalClaimResponse,
    VerificationJobResponse,
    EvidenceItem,
)
from ..models.enums import InputType, PipelineStep, Stance, UserVerdict, Verdict

from .claim_extractor import extract_claims
from .claim_normalizer import normalize_claim
from .claim_matcher import match_claim
from .evidence_retriever import retrieve_evidence
from .evidence_reranker import rerank_evidence
from .claim_verifier import verify_claim
from .confidence import estimate_confidence
from .escalation import check_escalation
from .verdict_policy import decide_cached_verdict, decide_user_verdict

from ..utils.blacklist import BlacklistMatch, match_blacklisted_source, normalize_hostname
from ..utils.language import detect_and_translate, LanguageResult
from ..utils.redaction import redact_secrets

logger = structlog.get_logger()


class PipelineOrchestrator:

    def __init__(self):
        self._db = None

    @property
    def db(self):
        if self._db is None:
            from ..db.postgres import ClaimDB
            self._db = ClaimDB
        return self._db

    async def run(
        self,
        input_text: str,
        input_type: InputType = InputType.ARTICLE,
        job_id: str | None = None,
        source_url: str | None = None,
    ) -> VerificationJobResponse:
        job_id = job_id or str(uuid.uuid4())
        job_start = time.time()
        current_date = datetime.now(timezone.utc).strftime("%Y-%m-%d")

        logger.info("pipeline.start", job_id=job_id, input_type=input_type.value)

        # When the caller declares the input itself is a URL and the whole
        # input is a single URL, treat it as the source. Never scan article
        # text for mentioned URLs: that would flag articles about satire sites.
        if source_url is None and input_type == InputType.URL:
            candidate = input_text.strip()
            if candidate and not any(char.isspace() for char in candidate):
                if normalize_hostname(candidate) is not None:
                    source_url = candidate

        # Tier 4 is a pre-detection policy gate. Check it before the database,
        # language translation, claim extraction, search, or LLM calls.
        source_domain = normalize_hostname(source_url)
        blacklist_match = match_blacklisted_source(source_url)
        if blacklist_match is not None:
            logger.info(
                "pipeline.blacklist_short_circuit",
                job_id=job_id,
                matched_domain=blacklist_match.matched_domain,
                category=blacklist_match.category,
            )
            return self._build_blacklist_response(
                job_id=job_id,
                input_text=input_text,
                input_type=input_type,
                source_url=source_url,
                blacklist_match=blacklist_match,
                job_start=job_start,
            )

        try:
            try:
                await self.db.create_job(job_id, input_text, input_type.value)
            except Exception as e:
                logger.warning("pipeline.db_job_create_error", error=redact_secrets(str(e)))

            lang_result: LanguageResult = detect_and_translate(input_text)
            processing_text = lang_result.processed_text

            if lang_result.was_translated:
                logger.info(
                    "pipeline.translated",
                    job_id=job_id,
                    from_lang=lang_result.detected_language,
                    lang_name=lang_result.language_name,
                )

            step_start = time.time()
            extraction = await extract_claims(processing_text)
            await self._log_step(job_id, PipelineStep.CLAIM_EXTRACTION, 1, step_start)

            if not extraction.claims:
                logger.info("pipeline.no_claims", job_id=job_id)
                return VerificationJobResponse(
                    job_id=job_id,
                    status="completed",
                    input_text=input_text[:500],
                    input_type=input_type,
                    total_claims=0,
                    results=[],
                    non_checkable_statements=extraction.non_checkable_statements,
                    completed_at=datetime.now(timezone.utc).isoformat(),
                    total_duration_ms=int((time.time() - job_start) * 1000),
                    detected_language=lang_result.detected_language,
                    language_name=lang_result.language_name,
                    was_translated=lang_result.was_translated,
                    original_input_text=input_text if lang_result.was_translated else None,
                )

            async def _process_with_error_handling(i: int, claim):
                logger.info("pipeline.claim", job_id=job_id, claim_idx=i, claim_id=claim.claim_id)
                claim_start = time.time()
                try:
                    res = await self._process_single_claim(
                        job_id=job_id,
                        claim=claim,
                        input_text=processing_text,
                        input_type=input_type,
                        current_date=current_date,
                        claim_index=i,
                        lang_result=lang_result,
                        source_url=source_url,
                        source_domain=source_domain,
                    )
                    res.pipeline_duration_ms = int((time.time() - claim_start) * 1000)
                    return res
                except Exception as e:
                    logger.error(
                        "pipeline.claim_error",
                        claim_id=claim.claim_id,
                        error=redact_secrets(str(e)),
                        trace=redact_secrets(traceback.format_exc()),
                    )
                    return FinalClaimResponse(
                        claim=claim.text,
                        normalized_claim=claim.text,
                        verdict=UserVerdict.UNCERTAIN,
                        confidence=0.0,
                        reasoning=[f"Pipeline error: {redact_secrets(str(e))}"],
                        evidence=[],
                        requires_human_review=True,
                        escalation_reason=f"Pipeline error: {redact_secrets(str(e))}",
                        claim_id=claim.claim_id,
                        pipeline_duration_ms=int((time.time() - claim_start) * 1000),
                        evidence_verdict=Verdict.NOT_ENOUGH_EVIDENCE,
                        verdict_reason="PIPELINE_ERROR",
                        source_url=source_url,
                        source_domain=source_domain,
                        detected_language=lang_result.detected_language,
                        language_name=lang_result.language_name,
                        was_translated=lang_result.was_translated,
                        original_text=input_text if lang_result.was_translated else None,
                    )

            claim_tasks = [_process_with_error_handling(i, claim) for i, claim in enumerate(extraction.claims)]
            results: list[FinalClaimResponse] = list(await asyncio.gather(*claim_tasks))


            total_duration = int((time.time() - job_start) * 1000)

            response = VerificationJobResponse(
                job_id=job_id,
                status="completed",
                input_text=input_text[:500],
                input_type=input_type,
                total_claims=len(results),
                results=results,
                non_checkable_statements=extraction.non_checkable_statements,
                completed_at=datetime.now(timezone.utc).isoformat(),
                total_duration_ms=total_duration,
                source_url=source_url,
                detected_language=lang_result.detected_language,
                language_name=lang_result.language_name,
                was_translated=lang_result.was_translated,
                original_input_text=input_text if lang_result.was_translated else None,
            )

            try:
                serialized_results = [r.model_dump(mode="json") if hasattr(r, 'model_dump') else r for r in results]
                await self.db.update_job_status(
                    job_id, "completed", total_claims=len(results), results=serialized_results
                )
            except Exception as e:
                logger.warning("pipeline.db_update_error", error=redact_secrets(str(e)))

            logger.info(
                "pipeline.done",
                job_id=job_id,
                total_claims=len(results),
                duration_ms=total_duration,
            )

            return response

        except Exception as e:
            logger.error("pipeline.fatal", job_id=job_id, error=redact_secrets(str(e)))
            try:
                await self.db.update_job_status(job_id, "failed", error=redact_secrets(str(e)))
            except Exception:
                pass
            raise

    def _build_blacklist_response(
        self,
        job_id: str,
        input_text: str,
        input_type: InputType,
        source_url: str | None,
        blacklist_match: BlacklistMatch,
        job_start: float,
    ) -> VerificationJobResponse:
        """Build the immediate Tier 4 verdict without touching detector services."""
        reason = blacklist_match.reason
        reasoning = [
            (
                f"Source '{blacklist_match.input_host}' matches configured "
                f"Tier 4 domain '{blacklist_match.matched_domain}'."
            ),
            f"Category: {blacklist_match.category}. Reason code: {reason}.",
            (
                "This is a publisher-level policy rule, not a universal factual "
                "determination; satire and fabricated news are distinct categories."
            ),
            (
                "Claim extraction, evidence retrieval, search, and external API "
                "calls were skipped for this item."
            ),
        ]

        claim_result = FinalClaimResponse(
            claim=input_text[:500],
            normalized_claim=input_text[:500],
            verdict=UserVerdict.FAKE,
            confidence=1.0,
            reasoning=reasoning,
            evidence=[],
            matched_existing_claim=False,
            requires_human_review=False,
            escalation_reason=None,
            claim_id=str(uuid.uuid4()),
            pipeline_duration_ms=int((time.time() - job_start) * 1000),
            evidence_verdict=None,
            verdict_reason=reason,
            source_url=source_url,
            source_domain=blacklist_match.input_host,
            source_tier=blacklist_match.tier,
            source_tier_number=blacklist_match.tier_number,
            matched_blacklist_domain=blacklist_match.matched_domain,
            blacklist_category=blacklist_match.category,
        )

        return VerificationJobResponse(
            job_id=job_id,
            status="completed",
            input_text=input_text[:500],
            input_type=input_type,
            total_claims=1,
            results=[claim_result],
            non_checkable_statements=[],
            completed_at=datetime.now(timezone.utc).isoformat(),
            total_duration_ms=int((time.time() - job_start) * 1000),
            source_url=source_url,
        )

    async def _process_single_claim(
        self,
        job_id: str,
        claim,
        input_text: str,
        input_type: InputType,
        current_date: str,
        claim_index: int,
        lang_result: LanguageResult | None = None,
        source_url: str | None = None,
        source_domain: str | None = None,
    ) -> FinalClaimResponse:
        step_start = time.time()
        normalized = await normalize_claim(claim, input_text, current_date)
        await self._log_step(claim.claim_id, PipelineStep.CLAIM_NORMALIZATION, 2, step_start)

        claim_text = normalized.normalized_claim or claim.text

        try:
            await self.db.store_claim(
                claim_id=claim.claim_id,
                input_text=input_text[:1000],
                input_type=input_type.value,
                claim_text=claim.text,
                claim_index=claim_index,
                normalized_text=claim_text,
                entities=claim.entities,
                dates=claim.dates,
                numbers=claim.numbers,
                resolved_entities=[
                    e if isinstance(e, dict) else {"name": str(e)}
                    for e in (normalized.resolved_entities or [])
                ],
                resolved_dates=normalized.resolved_dates,
                detected_language=lang_result.detected_language if lang_result else None,
                was_translated=lang_result.was_translated if lang_result else False,
                original_text=lang_result.original_text if (lang_result and lang_result.was_translated) else None,
            )
        except Exception as e:
            logger.warning("pipeline.db_store_claim_error", error=redact_secrets(str(e)))

        step_start = time.time()
        match_result = await match_claim(normalized)
        await self._log_step(claim.claim_id, PipelineStep.CLAIM_MATCHING, 3, step_start)

        if match_result.matched_existing_claim and match_result.previous_verdict:
            logger.info(
                "pipeline.fast_path",
                claim_id=claim.claim_id,
                verdict=match_result.previous_verdict,
                similarity=match_result.similarity,
            )
            decision = decide_cached_verdict(match_result)
            if match_result.previous_confidence is not None:
                match_confidence = match_result.previous_confidence
            else:
                match_confidence = match_result.similarity or 0.85
            return FinalClaimResponse(
                claim=claim.text,
                normalized_claim=claim_text,
                verdict=decision.verdict,
                confidence=match_confidence,
                reasoning=[
                    f"Matched existing verified claim (similarity: {match_result.similarity:.2f})",
                    f"Source: {match_result.match_source}",
                    f"Matched claim: {match_result.matched_claim_text or 'N/A'}",
                ],
                evidence=[],
                matched_existing_claim=True,
                claim_id=claim.claim_id,
                evidence_verdict=decision.internal_verdict,
                verdict_reason=decision.reason,
                verdict_support_score=decision.support_score,
                source_url=source_url,
                source_domain=source_domain,
                detected_language=lang_result.detected_language if lang_result else None,
                language_name=lang_result.language_name if lang_result else None,
                was_translated=lang_result.was_translated if lang_result else False,
                original_text=lang_result.original_text if (lang_result and lang_result.was_translated) else None,
            )

        step_start = time.time()
        raw_evidence = await retrieve_evidence(normalized)
        await self._log_step(claim.claim_id, PipelineStep.EVIDENCE_RETRIEVAL, 4, step_start)

        step_start = time.time()
        ranked_evidence = await rerank_evidence(claim_text, raw_evidence)
        await self._log_step(claim.claim_id, PipelineStep.EVIDENCE_RERANKING, 5, step_start)

        step_start = time.time()
        verification = await verify_claim(claim.claim_id, claim_text, ranked_evidence)
        await self._log_step(claim.claim_id, PipelineStep.CLAIM_VERIFICATION, 6, step_start)

        step_start = time.time()
        confidence, breakdown = await estimate_confidence(verification, ranked_evidence)
        verification.confidence = confidence
        await self._log_step(claim.claim_id, PipelineStep.CONFIDENCE_ESTIMATION, 7, step_start)

        step_start = time.time()
        escalation = await check_escalation(claim_text, verification, ranked_evidence)
        await self._log_step(claim.claim_id, PipelineStep.HUMAN_ESCALATION, 8, step_start)

        decision = decide_user_verdict(verification, ranked_evidence, confidence)

        await self._store_results(
            claim, claim_text, verification, ranked_evidence, escalation
        )

        evidence_items = []
        for ev in ranked_evidence.ranked_evidence:
            stance_info = next(
                (s for s in verification.supporting_evidence if s.doc_id == ev.doc_id),
                None,
            )
            evidence_items.append(EvidenceItem(
                source=ev.source_domain or ev.title,
                url=ev.url,
                published_date=ev.published_date,
                snippet=ev.passages[0] if ev.passages else "",
                stance=stance_info.stance if stance_info else Stance.NEUTRAL,
                credibility_score=ev.credibility_score,
                relevance_score=ev.relevance_score,
            ))

        return FinalClaimResponse(
            claim=claim.text,
            normalized_claim=claim_text,
            verdict=decision.verdict,
            confidence=confidence,
            reasoning=verification.reasoning,
            evidence=evidence_items,
            matched_existing_claim=False,
            requires_human_review=escalation.requires_human_review,
            escalation_reason=escalation.reason,
            claim_id=claim.claim_id,
            evidence_verdict=decision.internal_verdict,
            verdict_reason=decision.reason,
            verdict_support_score=decision.support_score,
            source_url=source_url,
            source_domain=source_domain,
            detected_language=lang_result.detected_language if lang_result else None,
            language_name=lang_result.language_name if lang_result else None,
            was_translated=lang_result.was_translated if lang_result else False,
            original_text=lang_result.original_text if (lang_result and lang_result.was_translated) else None,
        )

    async def _store_results(self, claim, claim_text, verification, ranked_evidence, escalation):
        try:
            verification_id = await self.db.store_verification(
                claim_id=claim.claim_id,
                verdict=verification.verdict.value,
                confidence=verification.confidence,
                reasoning=verification.reasoning,
                subclaim_results=[
                    {"text": sc.subclaim_text, "verdict": sc.verdict.value}
                    for sc in verification.subclaim_results
                ],
                model_used=f"{settings_provider()}/{settings_model()}",
            )

            for ev in ranked_evidence.ranked_evidence:
                stance_info = next(
                    (s for s in verification.supporting_evidence if s.doc_id == ev.doc_id),
                    None,
                )
                await self.db.store_evidence(
                    claim_id=claim.claim_id,
                    title=ev.title,
                    url=ev.url,
                    source_domain=ev.source_domain,
                    published_date=ev.published_date,
                    credibility_score=ev.credibility_score,
                    relevance_score=ev.relevance_score,
                    freshness_score=ev.freshness_score,
                    final_rank_score=ev.final_rank_score,
                    passages=ev.passages,
                    stance=stance_info.stance.value if stance_info else None,
                    is_primary_source=ev.is_primary_source,
                    retrieval_query=None,
                )

            from ..services.vector_service import vector_service
            await vector_service.store_claim(
                claim_id=claim.claim_id,
                claim_text=claim_text,
                verdict=verification.verdict.value,
                confidence=verification.confidence,
            )

            if escalation.requires_human_review:
                await self.db.create_escalation(
                    claim_id=claim.claim_id,
                    verification_id=verification_id,
                    reason=escalation.reason or "Unknown",
                    priority=escalation.priority,
                )

        except Exception as e:
            logger.warning("pipeline.store_error", claim_id=claim.claim_id, error=redact_secrets(str(e)))

    async def _log_step(self, entity_id: str, step: PipelineStep, index: int, start_time: float):
        duration_ms = int((time.time() - start_time) * 1000)
        try:
            await self.db.log_audit(
                claim_id=entity_id,
                step_name=step.value,
                step_index=index,
                duration_ms=duration_ms,
            )
        except Exception:
            pass
        logger.debug("pipeline.step", step=step.value, duration_ms=duration_ms)


def settings_provider() -> str:
    try:
        from ..config import settings
        return settings.llm_provider
    except Exception:
        return "unknown"


def settings_model() -> str:
    try:
        from ..config import settings
        provider = settings.llm_provider.strip().lower().replace("_", "-")
        if provider == "gemini":
            return settings.gemini_model
        if provider == "deepseek":
            return settings.deepseek_model
        if provider == "openai-compatible":
            return settings.openai_compatible_model
        if provider == "ollama":
            return settings.ollama_model
        return settings.openai_model
    except Exception:
        return "unknown"
