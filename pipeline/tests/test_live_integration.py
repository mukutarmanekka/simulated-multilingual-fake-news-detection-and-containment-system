"""Opt-in live integration tests.

These tests make real, bounded requests to the configured DeepSeek, Tavily,
and Google Fact Check APIs. They are skipped unless RUN_LIVE_INTEGRATION=1:

    RUN_LIVE_INTEGRATION=1 pytest tests/test_live_integration.py -m live -s

Nothing here replaces the real detector. The tests call the existing
services and PipelineOrchestrator, and they never print API keys.
"""

from __future__ import annotations

import asyncio
import os
from contextlib import contextmanager
from urllib.parse import urlparse

import pytest
import pytest_asyncio

from src.config import settings
from src.models.enums import UserVerdict
from src.pipeline.orchestrator import PipelineOrchestrator
from src.services.factcheck_service import factcheck_service
from src.services.llm_service import llm_service
from src.services.news_service import news_service
from src.services.search_service import search_service
from src.services.vector_service import vector_service
from src.utils.redaction import redact_secrets

pytestmark = [
    pytest.mark.live,
    pytest.mark.skipif(
        os.environ.get("RUN_LIVE_INTEGRATION") != "1",
        reason="Set RUN_LIVE_INTEGRATION=1 to run live API integration tests",
    ),
]

EVIDENCE_SOURCE = "https://www.britannica.com/topic/Eiffel-Tower-Paris-France"


def _configured(value: str | None) -> bool:
    return bool((value or "").strip())


def _require_key(name: str, value: str | None) -> None:
    if not _configured(value):
        pytest.skip(f"{name} is not configured; provider NOT TESTED")


def _safe_error(exc: Exception) -> str:
    return f"{type(exc).__name__}: {redact_secrets(str(exc))[:300]}"


async def _reset_http_client(service) -> None:
    """Close and drop a service client created in an earlier event loop.

    The production app runs in one long-lived loop. Pytest gives each test a
    fresh loop, so the harness must not reuse an AsyncClient bound to the
    previous one (the same constraint documented in the containment adapter).
    """
    client = getattr(service, "_client", None)
    if client is None:
        return
    try:
        await client.aclose()
    except Exception:
        pass
    service._client = None


async def _reset_db_engine() -> None:
    """Drop the loop-bound async DB engine between live tests."""
    from src.db import postgres as db_module

    engine = db_module._engine
    if engine is not None:
        try:
            await engine.dispose()
        except Exception:
            pass
    db_module._engine = None
    db_module._async_session = None
    db_module._db_available = None


async def _reset_live_clients() -> None:
    await _reset_http_client(search_service)
    await _reset_http_client(factcheck_service)
    await _reset_http_client(news_service)
    await _reset_db_engine()


@pytest_asyncio.fixture(autouse=True)
async def fresh_live_clients():
    await _reset_live_clients()
    yield
    await _reset_live_clients()


@contextmanager
def force_full_pipeline():
    """Make the vector fast path miss so the live test always runs fully.

    The production pipeline keeps its normal similarity threshold. This is
    a test-harness cache miss: without it, a second live run would reuse the
    claim cached by the first run and skip evidence retrieval and
    verification.
    """
    original = vector_service.search_similar

    async def no_cached_matches(*args, **kwargs):
        return []

    vector_service.search_similar = no_cached_matches
    try:
        yield
    finally:
        vector_service.search_similar = original


@contextmanager
def provider_spy():
    """Count calls to the existing provider services without replacing them."""
    calls = {
        "llm": 0,
        "search": 0,
        "factcheck": 0,
        "news": 0,
        "content": 0,
    }

    original_llm = llm_service.generate_json
    original_search = search_service.search
    original_factcheck = factcheck_service.search_claims
    original_news = news_service.search_articles
    original_content = search_service.extract_content

    async def llm_wrapper(*args, **kwargs):
        calls["llm"] += 1
        return await original_llm(*args, **kwargs)

    async def search_wrapper(*args, **kwargs):
        calls["search"] += 1
        return await original_search(*args, **kwargs)

    async def factcheck_wrapper(*args, **kwargs):
        calls["factcheck"] += 1
        return await original_factcheck(*args, **kwargs)

    async def news_wrapper(*args, **kwargs):
        calls["news"] += 1
        return await original_news(*args, **kwargs)

    async def content_wrapper(*args, **kwargs):
        calls["content"] += 1
        return await original_content(*args, **kwargs)

    llm_service.generate_json = llm_wrapper
    search_service.search = search_wrapper
    factcheck_service.search_claims = factcheck_wrapper
    news_service.search_articles = news_wrapper
    search_service.extract_content = content_wrapper

    try:
        yield calls
    finally:
        llm_service.generate_json = original_llm
        search_service.search = original_search
        factcheck_service.search_claims = original_factcheck
        news_service.search_articles = original_news
        search_service.extract_content = original_content


# ── Provider-by-provider live checks ────────────────────────────────


@pytest.mark.asyncio
async def test_live_deepseek_provider():
    _require_key("DEEPSEEK_API_KEY", settings.deepseek_api_key)

    if llm_service.provider != "deepseek":
        pytest.fail(
            f"DeepSeek configured but active provider is '{llm_service.provider}'"
        )

    try:
        result = await asyncio.wait_for(
            llm_service.generate_json(
                system_prompt="Return JSON only.",
                user_prompt='Return {"live_check": "ok"}.',
                temperature=0.0,
                max_tokens=64,
            ),
            timeout=90,
        )
    except Exception as exc:
        pytest.fail(f"LIVE FAILURE deepseek: {_safe_error(exc)}")

    assert isinstance(result, dict)
    assert result, "DeepSeek returned no usable JSON"

    print(
        "LIVE_SUCCESS provider=deepseek "
        f"model={settings.deepseek_model} parsed_keys={sorted(result.keys())}"
    )


@pytest.mark.asyncio
async def test_live_tavily_provider():
    _require_key("TAVILY_API_KEY", settings.tavily_api_key)

    try:
        results = await asyncio.wait_for(
            search_service.search_tavily("Eiffel Tower Paris location", count=2),
            timeout=90,
        )
    except Exception as exc:
        pytest.fail(f"LIVE FAILURE tavily: {_safe_error(exc)}")

    assert results, "Tavily returned no results for a common query"
    assert all(item.get("url") for item in results)
    assert all(item.get("title") for item in results)

    domains = [
        urlparse(item.get("url", "")).netloc
        for item in results
    ]
    print(
        "LIVE_SUCCESS provider=tavily "
        f"results={len(results)} domains={domains}"
    )


@pytest.mark.asyncio
async def test_live_factcheck_provider():
    _require_key("GOOGLE_FACTCHECK_API_KEY", settings.google_factcheck_api_key)

    query = "Barack Obama was born in Kenya"
    try:
        results = await asyncio.wait_for(
            factcheck_service.search_claims(query, max_results=2),
            timeout=90,
        )
    except Exception as exc:
        pytest.fail(f"LIVE FAILURE google_factcheck: {_safe_error(exc)}")

    if factcheck_service.last_error is not None:
        pytest.fail(
            "LIVE FAILURE google_factcheck: "
            f"{factcheck_service.last_error} "
            f"(status={factcheck_service.last_status})"
        )

    assert factcheck_service.last_status == 200
    assert results, "Fact Check API returned no claims for a known ClaimReview query"
    assert results[0].get("claim_text")
    assert results[0].get("rating")
    assert results[0].get("publisher")

    publishers = sorted({item.get("publisher", "") for item in results})
    ratings = [item.get("rating", "") for item in results]
    print(
        "LIVE_SUCCESS provider=google_factcheck "
        f"results={len(results)} publishers={publishers} ratings={ratings}"
    )


# ── Full pipeline cases ─────────────────────────────────────────────


async def _run_pipeline_case(case_name, claim, expected):
    orchestrator = PipelineOrchestrator()

    with provider_spy() as calls, force_full_pipeline():
        try:
            response = await asyncio.wait_for(
                orchestrator.run(
                    input_text=claim,
                    source_url=EVIDENCE_SOURCE,
                ),
                timeout=300,
            )
        except Exception as exc:
            pytest.fail(f"{case_name}: pipeline failed: {_safe_error(exc)}")

    assert response.total_claims >= 1, f"{case_name}: no checkable claim extracted"
    result = response.results[0]

    assert result.matched_existing_claim is False, (
        f"{case_name}: vector fast path was used; this was not a full live run"
    )
    assert calls["llm"] > 0, f"{case_name}: DeepSeek was not called"
    assert calls["search"] > 0, f"{case_name}: Tavily was not called"
    assert calls["factcheck"] > 0, f"{case_name}: Fact Check was not called"

    evidence_summary = []
    for item in result.evidence[:5]:
        evidence_summary.append(
            {
                "source": item.source,
                "stance": item.stance.value,
                "credibility": item.credibility_score,
                "relevance": item.relevance_score,
                "snippet": redact_secrets(item.snippet)[:160],
            }
        )

    expectation_matched = result.verdict == expected
    print(
        f"LIVE_PIPELINE case={case_name} "
        f"expected={expected.value} actual={result.verdict.value} "
        f"evidence_verdict={result.evidence_verdict.value if result.evidence_verdict else None} "
        f"reason={result.verdict_reason} confidence={result.confidence} "
        f"support_score={result.verdict_support_score} "
        f"matched_existing={result.matched_existing_claim} "
        f"expectation_matched={expectation_matched}"
    )
    print(
        f"LIVE_PROVIDER_USAGE case={case_name} "
        f"llm_calls={calls['llm']} search_calls={calls['search']} "
        f"factcheck_calls={calls['factcheck']} news_calls={calls['news']} "
        f"content_fetches={calls['content']} "
        f"search_provider={search_service.last_provider} "
        f"factcheck_status={factcheck_service.last_status} "
        f"factcheck_error={factcheck_service.last_error}"
    )
    print(f"LIVE_EVIDENCE case={case_name} evidence={evidence_summary}")

    return result, calls


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "case_name,claim,expected",
    [
        (
            "credible",
            "The Eiffel Tower was completed in 1889.",
            UserVerdict.REAL,
        ),
        (
            "false",
            "The Great Wall of China was built by the Romans.",
            UserVerdict.FAKE,
        ),
        (
            "ambiguous",
            "The Eiffel Tower's 2026 refurbishment budget is 42 million euros.",
            UserVerdict.UNCERTAIN,
        ),
    ],
)
async def test_live_full_pipeline(case_name, claim, expected):
    _require_key("DEEPSEEK_API_KEY", settings.deepseek_api_key)
    _require_key("TAVILY_API_KEY", settings.tavily_api_key)
    _require_key("GOOGLE_FACTCHECK_API_KEY", settings.google_factcheck_api_key)

    await _run_pipeline_case(case_name, claim, expected)


@pytest.mark.asyncio
async def test_live_blacklist_skips_all_providers():
    orchestrator = PipelineOrchestrator()

    with provider_spy() as calls:
        response = await asyncio.wait_for(
            orchestrator.run(
                input_text="The Eiffel Tower was completed in 1889.",
                source_url="https://www.theonion.com/some-article",
            ),
            timeout=30,
        )

    assert response.total_claims == 1
    result = response.results[0]
    assert result.verdict == UserVerdict.FAKE
    assert result.verdict_reason == "BLACKLISTED_SATIRE_SOURCE"
    assert result.matched_blacklist_domain == "theonion.com"
    assert calls == {
        "llm": 0,
        "search": 0,
        "factcheck": 0,
        "news": 0,
        "content": 0,
    }

    print(
        "LIVE_BLACKLIST "
        f"verdict={result.verdict.value} "
        f"reason={result.verdict_reason} "
        f"matched_domain={result.matched_blacklist_domain} "
        f"provider_calls={calls}"
    )
