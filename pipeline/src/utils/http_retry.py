"""Shared retry predicate for transient HTTP provider failures.

Retry timeouts, connection failures, and transient HTTP statuses. Do not
retry bad credentials (401/403) or malformed requests (400/422), which would
only waste quota and hide configuration errors.
"""

from __future__ import annotations

import httpx
from tenacity import retry_if_exception

RETRYABLE_STATUS_CODES = frozenset({408, 425, 429, 500, 502, 503, 504})


def is_retryable_http_error(exc: BaseException) -> bool:
    """Return True for timeouts, transport failures, and transient statuses."""
    if isinstance(exc, httpx.HTTPStatusError):
        return exc.response.status_code in RETRYABLE_STATUS_CODES
    return isinstance(exc, (httpx.TimeoutException, httpx.TransportError))


retry_on_transient_http = retry_if_exception(is_retryable_http_error)
