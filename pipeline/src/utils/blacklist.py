"""Tier 4 source blacklist.

The blacklist is a configured publisher-policy registry. It is deliberately
separate from the ordinary source-credibility tiers: a match means the
publisher is designated as satire, parody, troll, or a known fake-news
outlet, and the detector can short-circuit before spending money on search
and LLM calls.

A blacklist match is not a universal truth verdict. Satire and fabricated
news are different concepts, and the registry records which category
applied so callers can report that distinction.

Registry format (JSON):

    {
      "policy": "human-readable policy note",
      "categories": {"satire": "...", "parody": "...", "troll": "...",
                     "fake_news": "..."},
      "domains": {
        "example.com": {"name": "Example", "category": "satire",
                        "notes": "optional"},
        "other.example": "fake_news"
      }
    }

The dictionary key is the canonical registrable domain. Subdomains of that
domain match by label suffix (``news.example.com`` matches ``example.com``),
but a domain that merely contains the blacklisted name as a substring
(``example.com.evil.test`` or ``notexample.com``) does not match.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse

import structlog

from .redaction import redact_secrets

logger = structlog.get_logger()

# pipeline/src/utils/blacklist.py -> pipeline/config/blacklisted_sources.json
DEFAULT_CONFIG_PATH = (
    Path(__file__).resolve().parents[2] / "config" / "blacklisted_sources.json"
)

TIER_4 = "BLACKLISTED"
TIER_4_NUMBER = 4

ALLOWED_CATEGORIES = frozenset({"satire", "parody", "troll", "fake_news"})

CATEGORY_REASONS = {
    "satire": "BLACKLISTED_SATIRE_SOURCE",
    "parody": "BLACKLISTED_PARODY_SOURCE",
    "troll": "BLACKLISTED_TROLL_SOURCE",
    "fake_news": "BLACKLISTED_FAKE_NEWS_SOURCE",
}


@dataclass(frozen=True)
class BlacklistEntry:
    """One configured publisher in the Tier 4 registry."""

    domain: str
    name: str
    category: str
    notes: str = ""


@dataclass(frozen=True)
class BlacklistMatch:
    """Result of matching a URL or hostname against the Tier 4 registry."""

    input_host: str
    matched_domain: str
    name: str
    category: str
    tier: str = TIER_4
    tier_number: int = TIER_4_NUMBER

    @property
    def reason(self) -> str:
        return CATEGORY_REASONS.get(self.category, "BLACKLISTED_SOURCE")


def normalize_hostname(value: str | None) -> str | None:
    """Normalize a URL or bare domain to a comparable hostname.

    Handles schemes, scheme-relative URLs, bare domains, userinfo, ports,
    trailing dots, ``www.``, and IDN hostnames. Returns ``None`` instead of
    raising for malformed or missing input.
    """
    if value is None:
        return None

    if not isinstance(value, str):
        try:
            value = str(value)
        except Exception:
            return None

    candidate = value.strip()
    if not candidate:
        return None

    try:
        if "://" in candidate:
            parsed = urlparse(candidate)
            if parsed.scheme.lower() not in ("http", "https"):
                return None
        elif candidate.startswith("//"):
            parsed = urlparse(candidate)
        else:
            # Bare domain, domain:port, or domain/path.
            parsed = urlparse("//" + candidate)
    except ValueError:
        return None

    try:
        host = parsed.hostname
    except ValueError:
        return None

    if not host:
        return None

    host = host.strip().strip(".").lower()
    if not host:
        return None

    if host.startswith("www."):
        host = host[4:]

    if not host:
        return None

    if any(ord(char) > 127 for char in host):
        try:
            host = host.encode("idna").decode("ascii")
        except UnicodeError:
            return None

    # A real hostname cannot contain whitespace, path separators, or userinfo.
    if " " in host or "/" in host or "@" in host:
        return None

    return host


class BlacklistRegistry:
    """Loads and matches the configurable Tier 4 domain registry."""

    def __init__(self, config_path: str | Path | None = None):
        self._config_path = config_path
        self._entries: dict[str, BlacklistEntry] = {}
        self._policy = ""
        self._loaded = False

    @property
    def config_path(self) -> Path:
        if self._config_path is not None:
            return Path(self._config_path)

        env_path = os.environ.get("BLACKLIST_CONFIG_PATH", "").strip()
        if env_path:
            return Path(env_path)

        try:
            from ..config import settings
            configured = (settings.blacklist_config_path or "").strip()
            if configured:
                return Path(configured)
        except Exception:
            pass

        return DEFAULT_CONFIG_PATH

    @property
    def policy(self) -> str:
        self.load()
        return self._policy

    def entries(self) -> dict[str, BlacklistEntry]:
        self.load()
        return dict(self._entries)

    def load(self, force: bool = False) -> None:
        if self._loaded and not force:
            return

        self._entries = {}
        self._policy = ""
        path = self.config_path

        try:
            with open(path, "r", encoding="utf-8") as handle:
                data = json.load(handle)
        except FileNotFoundError:
            logger.warning("blacklist.config_missing", path=str(path))
            self._loaded = True
            return
        except (OSError, json.JSONDecodeError) as error:
            logger.warning(
                "blacklist.config_invalid",
                path=str(path),
                error=redact_secrets(str(error)),
            )
            self._loaded = True
            return

        if not isinstance(data, dict):
            logger.warning("blacklist.config_not_object", path=str(path))
            self._loaded = True
            return

        self._policy = str(data.get("policy", ""))

        domains = data.get("domains", {})
        if not isinstance(domains, dict):
            logger.warning("blacklist.domains_not_object", path=str(path))
            self._loaded = True
            return

        for raw_domain, raw_entry in domains.items():
            entry = self._parse_entry(raw_domain, raw_entry)
            if entry is not None:
                self._entries[entry.domain] = entry

        self._loaded = True
        logger.info(
            "blacklist.loaded",
            path=str(path),
            domains=len(self._entries),
            policy=self._policy,
        )

    def reload(self, config_path: str | Path | None = None) -> None:
        """Force a reload, optionally switching to another registry file."""
        self._config_path = config_path
        self.load(force=True)

    def _parse_entry(self, raw_domain, raw_entry) -> BlacklistEntry | None:
        domain = normalize_hostname(raw_domain)
        if not domain:
            logger.warning("blacklist.invalid_domain", domain=str(raw_domain))
            return None

        # A single-label entry such as "com" would suffix-match every domain.
        # Require a fully qualified registrable domain.
        if "." not in domain:
            logger.warning("blacklist.domain_not_fully_qualified", domain=domain)
            return None

        if isinstance(raw_entry, str):
            category = raw_entry.strip().lower()
            name = domain
            notes = ""
        elif isinstance(raw_entry, dict):
            category = str(raw_entry.get("category", "")).strip().lower()
            name = str(raw_entry.get("name", domain))
            notes = str(raw_entry.get("notes", ""))
        else:
            logger.warning("blacklist.invalid_entry", domain=domain)
            return None

        if category not in ALLOWED_CATEGORIES:
            logger.warning(
                "blacklist.invalid_category",
                domain=domain,
                category=category,
                allowed=sorted(ALLOWED_CATEGORIES),
            )
            return None

        return BlacklistEntry(
            domain=domain,
            name=name,
            category=category,
            notes=notes,
        )

    def match(self, value: str | None) -> BlacklistMatch | None:
        """Return the Tier 4 match for a URL/domain, or ``None``."""
        self.load()

        host = normalize_hostname(value)
        if not host:
            return None

        entry = self._entries.get(host)
        if entry is not None:
            return self._build_match(host, entry)

        # Safe subdomain matching: only complete trailing label sequences.
        # "news.theonion.com" matches "theonion.com", while
        # "theonion.com.evil.test" and "nottheonion.com" do not.
        labels = host.split(".")
        for index in range(1, len(labels)):
            parent = ".".join(labels[index:])
            entry = self._entries.get(parent)
            if entry is not None:
                return self._build_match(host, entry)

        return None

    @staticmethod
    def _build_match(host: str, entry: BlacklistEntry) -> BlacklistMatch:
        return BlacklistMatch(
            input_host=host,
            matched_domain=entry.domain,
            name=entry.name,
            category=entry.category,
        )


_registry = BlacklistRegistry()


def get_blacklist_registry() -> BlacklistRegistry:
    return _registry


def reload_blacklist(config_path: str | Path | None = None) -> None:
    """Reload the module-level registry (used by tests and long-running apps)."""
    _registry.reload(config_path)


def match_blacklisted_source(value: str | None) -> BlacklistMatch | None:
    """Match a source URL or domain against the Tier 4 registry."""
    return _registry.match(value)


def is_blacklisted_source(value: str | None) -> bool:
    """Return True when the URL/domain is configured in Tier 4."""
    return _registry.match(value) is not None
