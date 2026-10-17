"""The ONLY module allowed to import the real detection pipeline.

Wraps the async PipelineOrchestrator in a synchronous detect() facade.
Each call re-checks that the LLM provider is not the silent mock
fallback, before and after the run, so canned verdicts can never enter
an experiment.
"""

import asyncio
import sys

from ..config import CONFIDENCE_THRESHOLD, PIPELINE_DIR, REPO_ROOT
from .base import ensure_real_provider, map_job_results


class RealDetectorAdapter:
    def __init__(self, pipeline_dir=PIPELINE_DIR, pacing_seconds=0.0):
        self._pipeline_dir = pipeline_dir
        self._pacing_seconds = pacing_seconds
        self._orchestrator = None
        self._input_type = None
        self._llm_service = None

    def _import_pipeline(self):
        if self._orchestrator is not None:
            return
        # mirrors cli.py:21 — the pipeline's src package resolves against
        # the pipeline directory
        if str(self._pipeline_dir) not in sys.path:
            sys.path.insert(0, str(self._pipeline_dir))
        from dotenv import load_dotenv

        load_dotenv(REPO_ROOT / ".env")
        from src.config import settings
        from src.models.enums import InputType
        from src.pipeline.orchestrator import PipelineOrchestrator
        from src.services.llm_service import llm_service

        self._settings = settings
        self._orchestrator = PipelineOrchestrator
        self._input_type = InputType
        self._llm_service = llm_service

    def active_provider(self):
        """The LLM provider name the pipeline is currently using."""
        self._import_pipeline()
        return self._llm_service.provider

    def active_model(self):
        """The model name behind the active provider (for cache records)."""
        self._import_pipeline()
        provider = self._llm_service.provider
        if provider == "gemini":
            return self._settings.gemini_model
        if provider == "openai":
            return self._settings.openai_model
        if provider == "ollama":
            return self._settings.ollama_model
        return provider

    async def run_job(self, text, language, post_id):
        """Run the pipeline and return the raw job response (async)."""
        self._import_pipeline()
        ensure_real_provider(self._llm_service.provider)
        response = await self._orchestrator().run(
            input_text=text,
            input_type=self._input_type.SOCIAL_MEDIA,
            job_id=f"verifacto-{post_id}",
        )
        # the pipeline could flip to mock mid-run; check again before use
        ensure_real_provider(self._llm_service.provider)
        return response

    async def detect_async(self, text, language, post_id):
        """One detection, awaited inside a caller-owned event loop."""
        response = await self.run_job(text, language, post_id)
        results = [claim.model_dump() for claim in response.results]
        return map_job_results(
            results, CONFIDENCE_THRESHOLD, provider=self._llm_service.provider
        )

    def detect(self, text, language, post_id):
        """Synchronous facade for a single detection.

        Batched callers should use detect_async inside ONE event loop:
        the pipeline binds its global clients (DB engine, HTTP clients)
        to the first loop it sees, and repeated asyncio.run() calls
        break them.
        """
        return asyncio.run(self.detect_async(text, language, post_id))
