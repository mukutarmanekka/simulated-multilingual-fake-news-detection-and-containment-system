"""The ONLY module allowed to import the real detection pipeline.

Wraps the async PipelineOrchestrator in the simulation's detector
interface, preserving the detector's REAL/FAKE/UNCERTAIN verdict, evidence,
reason codes, support score, Tier 4 metadata, latency, and provider usage.

The adapter does not reimplement verification policy. It passes the post
text/language/source URL into the detector's public entry point and maps
the returned job response into the simulation's DetectionResult schema.

Each detection re-checks that the LLM provider is not the silent mock
fallback, before and after the run, so canned verdicts can never enter an
experiment. A mock provider or an import failure raises
DetectorUnavailableError; the simulation must not substitute a stub.
"""

from __future__ import annotations

import asyncio
import sys
import time
import uuid
from contextlib import contextmanager

from ..config import (
    DETECTOR_ERROR_POLICY,
    DETECTOR_MAX_ATTEMPTS,
    DETECTOR_RETRY_DELAY_SECONDS,
    DETECTOR_TIMEOUT_SECONDS,
    PIPELINE_DIR,
    REPO_ROOT,
)
from .base import (
    DetectionResult,
    DetectorExecutionError,
    DetectorUnavailableError,
    ensure_real_provider,
    error_detection,
    map_response_dict,
    redact_error,
)


def model_for_provider(provider, settings):
    """Model name behind a provider, for cache attribution.

    Pure function over a settings-like object (the pipeline's Settings),
    so it can be unit-tested without importing the detector.
    """
    if provider == "gemini":
        return settings.gemini_model
    if provider == "openai":
        return settings.openai_model
    if provider == "ollama":
        return settings.ollama_model
    if provider == "deepseek":
        return settings.deepseek_model
    if provider == "openai-compatible":
        return settings.openai_compatible_model
    return provider


class RealDetectorAdapter:
    def __init__(
        self,
        pipeline_dir=PIPELINE_DIR,
        pacing_seconds=0.0,
        timeout_seconds=DETECTOR_TIMEOUT_SECONDS,
        error_policy=DETECTOR_ERROR_POLICY,
        max_attempts=DETECTOR_MAX_ATTEMPTS,
        retry_delay_seconds=DETECTOR_RETRY_DELAY_SECONDS,
    ):
        if error_policy not in ("record", "raise"):
            raise ValueError(
                "error_policy must be 'record' or 'raise', "
                f"got {error_policy!r}"
            )
        if max_attempts < 1:
            raise ValueError("max_attempts must be at least 1")

        self._pipeline_dir = pipeline_dir
        self._pacing_seconds = pacing_seconds
        self._timeout_seconds = timeout_seconds
        self._error_policy = error_policy
        self._max_attempts = max_attempts
        self._retry_delay_seconds = retry_delay_seconds

        self._orchestrator = None
        self._input_type = None
        self._llm_service = None
        self._search_service = None
        self._factcheck_service = None
        self._news_service = None
        self._redact_secrets = None
        self._settings = None

        self.last_latency_ms = None
        self.last_provider_usage = None
        self.last_provider_status = None

    def _import_pipeline(self):
        if self._orchestrator is not None:
            return
        try:
            # mirrors cli.py:21 — the pipeline's src package resolves
            # against the pipeline directory
            if str(self._pipeline_dir) not in sys.path:
                sys.path.insert(0, str(self._pipeline_dir))
            from dotenv import load_dotenv

            load_dotenv(REPO_ROOT / ".env")
            from src.config import settings
            from src.models.enums import InputType
            from src.pipeline.orchestrator import PipelineOrchestrator
            from src.services.factcheck_service import factcheck_service
            from src.services.llm_service import llm_service
            from src.services.news_service import news_service
            from src.services.search_service import search_service
            from src.utils.redaction import redact_secrets
        except Exception as exc:
            raise DetectorUnavailableError(
                "real detector import failed: " + redact_error(exc)
            ) from exc

        self._settings = settings
        self._orchestrator = PipelineOrchestrator
        self._input_type = InputType
        self._llm_service = llm_service
        self._search_service = search_service
        self._factcheck_service = factcheck_service
        self._news_service = news_service
        self._redact_secrets = redact_secrets

    def active_provider(self):
        """The LLM provider name the pipeline is currently using."""
        self._import_pipeline()
        return self._llm_service.provider

    def active_model(self):
        """The model name behind the active provider (for cache records)."""
        self._import_pipeline()
        return model_for_provider(self._llm_service.provider, self._settings)

    def safe_error(self, exc) -> str:
        """Redact credentials from detector/provider error text."""
        if self._redact_secrets is not None:
            return self._redact_secrets(str(exc))
        return redact_error(exc)

    @contextmanager
    def _count_provider_calls(self):
        """Count calls to the existing pipeline services for one post.

        This is instrumentation only: the same singleton clients and the
        same detector code are used. Detections are sequential, so the
        counters cannot mix results across posts.
        """
        self._import_pipeline()
        usage = {"llm": 0, "search": 0, "factcheck": 0, "news": 0, "content": 0}
        installed = []

        def make_wrapper(name, original):
            async def wrapper(*args, **kwargs):
                usage[name] += 1
                return await original(*args, **kwargs)

            return wrapper

        targets = (
            (self._llm_service, "generate_json", "llm"),
            (self._search_service, "search", "search"),
            (self._factcheck_service, "search_claims", "factcheck"),
            (self._news_service, "search_articles", "news"),
            (self._search_service, "extract_content", "content"),
        )
        try:
            for service, attribute, name in targets:
                original = getattr(service, attribute, None)
                if original is None:
                    continue
                installed.append((service, attribute, original))
                setattr(service, attribute, make_wrapper(name, original))
            yield usage
        finally:
            for service, attribute, original in reversed(installed):
                setattr(service, attribute, original)

    def _reset_provider_status(self):
        """Clear service status so a short-circuit cannot inherit old results."""
        for service in (self._search_service, self._factcheck_service):
            if service is None:
                continue
            for attribute in ("last_provider", "last_error", "last_status"):
                if hasattr(service, attribute):
                    setattr(service, attribute, None)

    async def run_job(self, text, language, post_id, source_url=None):
        """Run the real pipeline and return the raw job response (async)."""
        self._import_pipeline()
        ensure_real_provider(self._llm_service.provider)
        self._reset_provider_status()

        self.last_latency_ms = None
        self.last_provider_usage = None
        self.last_provider_status = None
        start = time.perf_counter()
        usage = None
        try:
            with self._count_provider_calls() as usage:
                response = await asyncio.wait_for(
                    self._orchestrator().run(
                        input_text=text,
                        input_type=self._input_type.SOCIAL_MEDIA,
                        # The pipeline's verification_jobs.id is UUID-typed.
                        # A fresh id per detection avoids duplicate-key
                        # collisions when an experiment is rerun.
                        job_id=str(uuid.uuid4()),
                        source_url=source_url,
                    ),
                    timeout=self._timeout_seconds,
                )
        finally:
            self.last_latency_ms = (time.perf_counter() - start) * 1000.0
            if usage is not None:
                self.last_provider_usage = dict(usage)
            self.last_provider_status = {
                "search_provider": getattr(
                    self._search_service, "last_provider", None
                ),
                "search_error": getattr(
                    self._search_service, "last_error", None
                ),
                "factcheck_status": getattr(
                    self._factcheck_service, "last_status", None
                ),
                "factcheck_error": getattr(
                    self._factcheck_service, "last_error", None
                ),
            }

        # the pipeline could flip to mock mid-run; check again before use
        ensure_real_provider(self._llm_service.provider)
        return response

    async def detect_async(self, text, language, post_id, source_url=None):
        """One detection, awaited inside a caller-owned event loop.

        Returns a structured DetectionResult. Provider failures become a
        ``status="error"``/``status="timeout"`` record unless the adapter
        was configured with ``error_policy="raise"``. The mock provider and
        import failures always raise DetectorUnavailableError, so the
        simulation can never silently substitute a stub.
        """
        self._import_pipeline()

        total_usage = {
            "llm": 0,
            "search": 0,
            "factcheck": 0,
            "news": 0,
            "content": 0,
        }
        start = time.perf_counter()
        last_status = "error"
        last_error = "detector failed"
        attempts = 0

        for attempt in range(1, self._max_attempts + 1):
            attempts = attempt
            try:
                response = await self.run_job(
                    text,
                    language,
                    post_id,
                    source_url,
                )
            except DetectorUnavailableError:
                # Configuration failure: never fall back to stub or mock.
                raise
            except asyncio.TimeoutError:
                last_status = "timeout"
                last_error = (
                    f"detector timeout after {self._timeout_seconds:.0f}s"
                )
            except Exception as exc:
                last_status = "error"
                last_error = self.safe_error(exc)
            else:
                attempt_usage = self.last_provider_usage or {}
                for key in total_usage:
                    total_usage[key] += int(attempt_usage.get(key, 0))
                total_usage["attempts"] = attempts
                return map_response_dict(
                    response.model_dump(mode="json"),
                    post_id=post_id,
                    text=text,
                    language=language,
                    source_url=source_url,
                    provider=self._llm_service.provider,
                    detector_mode="real",
                    latency_ms=(time.perf_counter() - start) * 1000.0,
                    provider_usage=total_usage,
                    provider_status=self.last_provider_status,
                )

            attempt_usage = self.last_provider_usage or {}
            for key in total_usage:
                total_usage[key] += int(attempt_usage.get(key, 0))
            if attempt < self._max_attempts:
                await asyncio.sleep(self._retry_delay_seconds)

        total_usage["attempts"] = attempts
        latency_ms = (time.perf_counter() - start) * 1000.0
        if self._error_policy == "raise":
            raise DetectorExecutionError(last_error)

        return error_detection(
            post_id=post_id,
            text=text,
            language=language,
            source_url=source_url,
            provider=self._llm_service.provider,
            detector_mode="real",
            status=last_status,
            error=last_error,
            latency_ms=latency_ms,
            provider_usage=total_usage,
            provider_status=self.last_provider_status,
        )

    def detect(self, text, language, post_id, source_url=None):
        """Synchronous facade for a single detection.

        Batched callers should use ``run_detection`` (which awaits all
        detections inside ONE event loop): the pipeline binds its global
        clients (DB engine, HTTP clients) to the first loop it sees, and
        repeated asyncio.run() calls break them.
        """
        return asyncio.run(
            self.detect_async(text, language, post_id, source_url)
        )
