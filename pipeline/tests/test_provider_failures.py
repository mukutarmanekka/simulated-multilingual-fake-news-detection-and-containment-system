"""Provider failure handling and secret redaction tests.

These tests never make network calls. They exercise the existing clients
with fake HTTP responses so timeout, rate-limit, invalid-response, and
missing-key paths are deterministic.
"""

from typing import Any, cast

import httpx
import pytest
from tenacity import wait_none

from src.services import llm_service as llm_module
from src.services.factcheck_service import FactCheckService, factcheck_service
from src.services.search_service import SearchService
from src.utils.redaction import redact_secrets


class _FakeResponse:
    def __init__(self, status_code=200, payload=None, json_error=None):
        self.status_code = status_code
        self._payload = payload if payload is not None else {}
        self._json_error = json_error
        self.request = httpx.Request(
            "GET", "https://factchecktools.googleapis.com/v1alpha1/claims:search"
        )

    def raise_for_status(self):
        if self.status_code >= 400:
            raise httpx.HTTPStatusError(
                "fake status error",
                request=self.request,
                response=httpx.Response(self.status_code, request=self.request),
            )

    def json(self):
        if self._json_error is not None:
            raise self._json_error
        return self._payload


class _FakeHTTPClient:
    def __init__(self, response=None, error=None):
        self._response = response
        self._error = error

    async def get(self, *args, **kwargs):
        if self._error is not None:
            raise self._error
        return self._response


class _SequenceHTTPClient:
    def __init__(self, outcomes):
        self._outcomes = list(outcomes)
        self.calls = 0

    async def get(self, *args, **kwargs):
        self.calls += 1
        outcome = self._outcomes.pop(0)
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome


def _valid_payload():
    return {
        "claims": [
            {
                "text": "a claim",
                "claimant": "Test",
                "claimReview": [
                    {
                        "textualRating": "False",
                        "url": "https://example.test/review",
                        "publisher": {"name": "Test Publisher"},
                        "reviewDate": "2026-01-01",
                        "title": "Test review",
                        "languageCode": "en",
                    }
                ],
            }
        ]
    }


@pytest.fixture
def fast_retries():
    """Disable tenacity's sleep so retry tests stay fast."""
    retrying = cast(Any, factcheck_service.search_claims).retry
    original_wait = retrying.wait
    retrying.wait = wait_none()
    try:
        yield
    finally:
        retrying.wait = original_wait


# ── Secret redaction ─────────────────────────────────────────────────


def test_redaction_removes_provider_keys_and_db_passwords():
    text = (
        "sk-secret123456789 tvly-secret123456789 AIzaSecret1234567890 "
        "https://example.test/?key=AIzaOther1234567890&x=1 "
        "Authorization: Bearer topsecret "
        "postgresql+asyncpg://user:example_db_password@localhost:5432/testdb"
    )
    redacted = redact_secrets(text)

    assert "sk-secret" not in redacted
    assert "tvly-secret" not in redacted
    assert "AIzaSecret" not in redacted
    assert "AIzaOther" not in redacted
    assert "topsecret" not in redacted
    assert "example_db_password" not in redacted
    assert "***" in redacted


def test_redaction_keeps_ordinary_text():
    text = "The Eiffel Tower is in Paris."
    assert redact_secrets(text) == text
    assert redact_secrets(None) == ""


# ── Fact Check API failure handling ─────────────────────────────────


@pytest.mark.asyncio
async def test_factcheck_missing_key_is_reported(monkeypatch):
    monkeypatch.setattr("src.services.factcheck_service.settings.google_factcheck_api_key", "")
    service = FactCheckService()

    results = await service.search_claims("a claim")

    assert results == []
    assert service.last_error == "missing_api_key"
    assert service.last_status is None


@pytest.mark.asyncio
async def test_factcheck_retries_429_then_succeeds(monkeypatch, fast_retries):
    monkeypatch.setattr(
        "src.services.factcheck_service.settings.google_factcheck_api_key", "test-key"
    )
    service = FactCheckService()
    client = _SequenceHTTPClient(
        [
            _FakeResponse(status_code=429),
            _FakeResponse(status_code=200, payload=_valid_payload()),
        ]
    )
    service._client = cast(Any, client)

    results = await service.search_claims("a claim")

    assert client.calls == 2
    assert results
    assert service.last_status == 200
    assert service.last_error is None


@pytest.mark.asyncio
async def test_factcheck_401_is_not_retried(monkeypatch, fast_retries):
    monkeypatch.setattr(
        "src.services.factcheck_service.settings.google_factcheck_api_key", "test-key"
    )
    service = FactCheckService()
    client = _SequenceHTTPClient([_FakeResponse(status_code=401)])
    service._client = cast(Any, client)

    results = await service.search_claims("a claim")

    assert client.calls == 1
    assert results == []
    assert service.last_status == 401
    assert service.last_error == "http_401"


@pytest.mark.asyncio
async def test_factcheck_timeout_retries_then_succeeds(monkeypatch, fast_retries):
    monkeypatch.setattr(
        "src.services.factcheck_service.settings.google_factcheck_api_key", "test-key"
    )
    service = FactCheckService()
    client = _SequenceHTTPClient(
        [
            httpx.TimeoutException("timed out"),
            _FakeResponse(status_code=200, payload={"claims": []}),
        ]
    )
    service._client = cast(Any, client)

    results = await service.search_claims("a claim")

    assert client.calls == 2
    assert results == []
    assert service.last_status == 200
    assert service.last_error is None


@pytest.mark.asyncio
async def test_factcheck_timeout_exhausts_retries(monkeypatch, fast_retries):
    monkeypatch.setattr(
        "src.services.factcheck_service.settings.google_factcheck_api_key", "test-key"
    )
    service = FactCheckService()
    client = _SequenceHTTPClient(
        [
            httpx.TimeoutException("timed out"),
            httpx.TimeoutException("timed out"),
            httpx.TimeoutException("timed out"),
        ]
    )
    service._client = cast(Any, client)

    with pytest.raises(httpx.TimeoutException):
        await service.search_claims("a claim")

    assert client.calls == 3
    assert service.last_error == "TimeoutException"


@pytest.mark.asyncio
async def test_factcheck_invalid_json_is_reported(monkeypatch):
    monkeypatch.setattr(
        "src.services.factcheck_service.settings.google_factcheck_api_key", "test-key"
    )
    service = FactCheckService()
    service._client = cast(
        Any,
        _FakeHTTPClient(
            response=_FakeResponse(
                status_code=200,
                json_error=ValueError("invalid JSON"),
            )
        ),
    )

    results = await service.search_claims("a claim")

    assert results == []
    assert service.last_error == "ValueError"


@pytest.mark.asyncio
async def test_factcheck_empty_valid_response_is_not_an_error(monkeypatch):
    monkeypatch.setattr(
        "src.services.factcheck_service.settings.google_factcheck_api_key", "test-key"
    )
    service = FactCheckService()
    service._client = cast(
        Any,
        _FakeHTTPClient(
            response=_FakeResponse(status_code=200, payload={"claims": []})
        ),
    )

    results = await service.search_claims("a claim")

    assert results == []
    assert service.last_status == 200
    assert service.last_error is None


# ── Search provider failure handling ────────────────────────────────


@pytest.mark.asyncio
async def test_tavily_missing_key_returns_empty_without_network(monkeypatch):
    monkeypatch.setattr("src.services.search_service.settings.tavily_api_key", "")
    service = SearchService()

    results = await service.search_tavily("a query", count=1)

    assert results == []


@pytest.mark.asyncio
async def test_search_reports_tavily_timeout_and_does_not_fabricate(monkeypatch):
    monkeypatch.setattr("src.services.search_service.settings.tavily_api_key", "test-key")
    monkeypatch.setattr("src.services.search_service.settings.brave_search_api_key", "")
    monkeypatch.setattr("src.services.search_service.settings.google_cse_api_key", "")
    monkeypatch.setattr("src.services.search_service.settings.google_cse_cx", "")

    service = SearchService()

    async def timed_out_tavily(query, count=10):
        raise httpx.TimeoutException("timed out")

    monkeypatch.setattr(service, "search_tavily", timed_out_tavily)

    results = await service.search("a query", count=1)

    assert results == []
    assert service.last_error == "TimeoutException"
    assert service.last_provider is None


@pytest.mark.asyncio
async def test_search_reports_no_results_without_provider_error(monkeypatch):
    monkeypatch.setattr("src.services.search_service.settings.tavily_api_key", "test-key")
    monkeypatch.setattr("src.services.search_service.settings.brave_search_api_key", "")
    monkeypatch.setattr("src.services.search_service.settings.google_cse_api_key", "")
    monkeypatch.setattr("src.services.search_service.settings.google_cse_cx", "")

    service = SearchService()

    async def empty_tavily(query, count=10):
        return []

    monkeypatch.setattr(service, "search_tavily", empty_tavily)

    results = await service.search("a query", count=1)

    assert results == []
    assert service.last_error == "no_results"
    assert service.last_provider is None


# ── LLM response parsing ─────────────────────────────────────────────


def test_llm_parse_json_handles_markdown_and_invalid_text():
    assert llm_module.LLMService._parse_json('```json\n{"ok": true}\n```') == {"ok": True}
    assert llm_module.LLMService._parse_json("not json") == {}
    assert llm_module.LLMService._parse_json("") == {}
