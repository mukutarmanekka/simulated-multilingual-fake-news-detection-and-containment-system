"""Real-adapter failure handling, no-fallback, and Tier 4 early exit."""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from verifacto.detector.base import (
    DetectorExecutionError,
    DetectorUnavailableError,
    VERDICT_FAKE,
)
from verifacto.detector.real_adapter import RealDetectorAdapter


class _SlowOrchestrator:
    async def run(self, **kwargs):
        await asyncio.sleep(0.2)
        return None


class _BrokenOrchestrator:
    async def run(self, **kwargs):
        raise RuntimeError("provider exploded")


class _NeverOrchestrator:
    async def run(self, **kwargs):
        raise AssertionError("orchestrator must not run")


def _bare_adapter(**kwargs):
    adapter = RealDetectorAdapter(**kwargs)
    adapter._import_pipeline = lambda: None
    adapter._input_type = SimpleNamespace(SOCIAL_MEDIA="social_media")
    adapter._llm_service = SimpleNamespace(provider="deepseek")
    adapter._search_service = None
    adapter._factcheck_service = None
    adapter._news_service = None
    adapter._redact_secrets = lambda value: str(value)
    adapter._settings = SimpleNamespace(deepseek_model="deepseek-chat")
    return adapter


@pytest.mark.asyncio
async def test_timeout_is_recorded_not_verdict():
    adapter = _bare_adapter(
        timeout_seconds=0.01,
        error_policy="record",
    )
    adapter._orchestrator = _SlowOrchestrator

    result = await adapter.detect_async(
        "x", "en", 1, "https://example.com/x"
    )

    assert result.status == "timeout"
    assert result.verdict is None
    assert result.is_fake is None
    assert result.verdict_summary == "ERROR"
    assert "timeout" in result.error.lower()
    assert result.provider_usage["attempts"] == 1


@pytest.mark.asyncio
async def test_provider_exception_is_recorded_not_verdict():
    adapter = _bare_adapter(error_policy="record")
    adapter._orchestrator = _BrokenOrchestrator

    result = await adapter.detect_async(
        "x", "en", 2, "https://example.com/x"
    )

    assert result.status == "error"
    assert result.verdict is None
    assert result.is_fake is None
    assert "provider exploded" in result.error


@pytest.mark.asyncio
async def test_error_policy_raise_aborts():
    adapter = _bare_adapter(error_policy="raise")
    adapter._orchestrator = _BrokenOrchestrator

    with pytest.raises(DetectorExecutionError):
        await adapter.detect_async("x", "en", 3, "https://example.com/x")


@pytest.mark.asyncio
async def test_mock_provider_never_falls_back_to_stub():
    adapter = _bare_adapter()
    adapter._llm_service = SimpleNamespace(provider="mock")
    adapter._orchestrator = _NeverOrchestrator

    with pytest.raises(DetectorUnavailableError):
        await adapter.detect_async("x", "en", 4, "https://example.com/x")


@pytest.mark.asyncio
async def test_blacklist_short_circuit_makes_zero_provider_calls(monkeypatch):
    adapter = RealDetectorAdapter(error_policy="record")
    adapter._import_pipeline()

    # The adapter's lazy import has now placed the detector on sys.path.
    from src.models.enums import InputType
    from src.pipeline.orchestrator import PipelineOrchestrator
    from src.services.factcheck_service import factcheck_service
    from src.services.llm_service import llm_service
    from src.services.news_service import news_service
    from src.services.search_service import search_service

    adapter._orchestrator = PipelineOrchestrator
    adapter._input_type = InputType
    monkeypatch.setattr(llm_service, "provider", "deepseek")

    llm_mock = AsyncMock(side_effect=AssertionError("LLM was called"))
    search_mock = AsyncMock(side_effect=AssertionError("search was called"))
    factcheck_mock = AsyncMock(
        side_effect=AssertionError("fact check was called")
    )
    news_mock = AsyncMock(side_effect=AssertionError("news was called"))
    content_mock = AsyncMock(
        side_effect=AssertionError("content was fetched")
    )
    monkeypatch.setattr(llm_service, "generate_json", llm_mock)
    monkeypatch.setattr(search_service, "search", search_mock)
    monkeypatch.setattr(
        factcheck_service, "search_claims", factcheck_mock
    )
    monkeypatch.setattr(news_service, "search_articles", news_mock)
    monkeypatch.setattr(
        search_service, "extract_content", content_mock
    )

    result = await adapter.detect_async(
        "The Eiffel Tower is located in Paris.",
        "en",
        5,
        "https://www.theonion.com/some-article",
    )

    assert result.verdict == VERDICT_FAKE
    assert result.is_fake is True
    assert result.blacklisted is True
    assert result.reason == "BLACKLISTED_SATIRE_SOURCE"
    assert result.matched_blacklist_domain == "theonion.com"
    assert result.blacklist_category == "satire"
    assert result.source_tier == "BLACKLISTED"
    assert result.source_tier_number == 4
    assert result.status == "ok"
    assert result.provider_usage["llm"] == 0
    assert result.provider_usage["search"] == 0
    assert result.provider_usage["factcheck"] == 0
    assert result.provider_usage["news"] == 0
    assert result.provider_usage["content"] == 0
    assert result.provider_status == {
        "search_provider": None,
        "search_error": None,
        "factcheck_status": None,
        "factcheck_error": None,
    }

    llm_mock.assert_not_awaited()
    search_mock.assert_not_awaited()
    factcheck_mock.assert_not_awaited()
    news_mock.assert_not_awaited()
    content_mock.assert_not_awaited()


@pytest.mark.asyncio
async def test_provider_usage_counts_successful_calls(monkeypatch):
    adapter = RealDetectorAdapter(error_policy="record")
    adapter._import_pipeline()

    from src.models.enums import InputType
    from src.services.llm_service import llm_service
    from src.services.search_service import search_service

    monkeypatch.setattr(llm_service, "provider", "deepseek")

    async def fake_llm(*args, **kwargs):
        return {"ok": True}

    async def fake_search(*args, **kwargs):
        return []

    monkeypatch.setattr(llm_service, "generate_json", fake_llm)
    monkeypatch.setattr(search_service, "search", fake_search)

    class FakeOrchestrator:
        async def run(self, **kwargs):
            await llm_service.generate_json("system", "user")
            await search_service.search("query")
            return SimpleNamespace(
                model_dump=lambda mode=None: {
                    "results": [
                        {
                            "claim": "x",
                            "verdict": "REAL",
                            "confidence": 0.8,
                            "verdict_reason": "EVIDENCE_CORROBORATES_CLAIM",
                            "evidence": [],
                        }
                    ]
                }
            )

    adapter._orchestrator = FakeOrchestrator
    adapter._input_type = InputType
    result = await adapter.detect_async(
        "x", "en", 11, "https://example.com/x"
    )

    assert result.verdict == "REAL"
    assert result.provider_usage["llm"] == 1
    assert result.provider_usage["search"] == 1
    assert result.provider_usage["factcheck"] == 0
    assert result.provider_usage["attempts"] == 1
