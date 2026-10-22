"""Transient provider errors retry; configuration errors fail fast."""

import httpx

from src.services.llm_service import _is_retryable_llm_error
from src.utils.http_retry import is_retryable_http_error


def _status_error(status_code: int) -> httpx.HTTPStatusError:
    request = httpx.Request("GET", "https://example.test/resource")
    response = httpx.Response(status_code, request=request)
    return httpx.HTTPStatusError(
        "provider status error",
        request=request,
        response=response,
    )


def test_retryable_http_statuses():
    for status in (408, 425, 429, 500, 502, 503, 504):
        assert is_retryable_http_error(_status_error(status)), status

    for status in (400, 401, 403, 404, 422):
        assert not is_retryable_http_error(_status_error(status)), status


def test_retryable_transport_errors():
    assert is_retryable_http_error(httpx.TimeoutException("timeout"))
    assert is_retryable_http_error(httpx.ConnectError("connection failed"))
    assert not is_retryable_http_error(ValueError("bad response"))


def test_llm_retry_predicate():
    assert _is_retryable_llm_error(_status_error(429))
    assert _is_retryable_llm_error(_status_error(503))
    assert not _is_retryable_llm_error(_status_error(401))
    assert not _is_retryable_llm_error(_status_error(400))
    assert _is_retryable_llm_error(TimeoutError("timed out"))
    assert not _is_retryable_llm_error(ValueError("bad request"))
