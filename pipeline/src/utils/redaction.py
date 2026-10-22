"""Redact credentials from log lines, errors, and terminal output.

The pipeline talks to providers whose credentials can appear in URLs,
headers, or exception text (for example ``?key=...`` on the Fact Check and
Google CSE APIs). This module is the single place that sanitizes such text
before it is logged or shown to a user.
"""

from __future__ import annotations

import re

_SECRET_PATTERNS = (
    (re.compile(r"sk-[A-Za-z0-9_-]{8,}"), "sk-***"),
    (re.compile(r"tvly-[A-Za-z0-9_-]{8,}"), "tvly-***"),
    (re.compile(r"AIza[0-9A-Za-z_-]{10,}"), "AIza***"),
    (
        re.compile(
            r"(?i)((?:api[_-]?key|apikey|access[_-]?token|token|key)=)[^&\s\"']+"
        ),
        r"\1***",
    ),
    (
        re.compile(r"(?i)((?:Authorization|X-Subscription-Token)\s*[:=]\s*)[^\r\n]+"),
        r"\1***",
    ),
    (re.compile(r"(?i)(Bearer\s+)[A-Za-z0-9._~+/=-]+"), r"\1***"),
    (
        re.compile(
            r"(?i)((?:https?|postgres(?:ql)?(?:\+asyncpg)?|redis)://[^:\s/@]+:)[^@\s]+@"
        ),
        r"\1***@",
    ),
)


def redact_secrets(value) -> str:
    """Return ``value`` as text with obvious credentials replaced."""
    if value is None:
        return ""

    text = str(value)
    for pattern, replacement in _SECRET_PATTERNS:
        text = pattern.sub(replacement, text)
    return text
