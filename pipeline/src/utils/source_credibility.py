"""Source Credibility Scoring.

Maintains a tiered source registry and provides credibility scores
for domains encountered during evidence retrieval.
"""

from __future__ import annotations

import structlog

from ..db.postgres import ClaimDB
from .blacklist import match_blacklisted_source, normalize_hostname
from .redaction import redact_secrets

logger = structlog.get_logger()

# ── In-memory cache of known sources (loaded on first use) ──────────

_SOURCE_CACHE: dict[str, dict] = {}

# Fallback hardcoded tier scores when DB is unavailable.
# Tier 4 is the configured blacklist (see utils/blacklist.py). It is kept
# separate from TIER_3 so satire/parody/troll policy decisions do not look
# like ordinary low-credibility sources.
TIER_SCORES = {
    "TIER_1": (0.90, 1.00),
    "TIER_2": (0.75, 0.89),
    "TIER_3": (0.50, 0.74),
    "UNKNOWN": (0.40, 0.40),
    "BLACKLISTED": (0.00, 0.00),
}

# ── Built-in known sources ──────────────────────────────────────────
# These ensure common domains get reasonable scores even without a DB.
KNOWN_SOURCES: dict[str, dict] = {
    # TIER 1 — Official / primary / global institutions
    "india.gov.in":       {"tier": "TIER_1", "credibility": 0.90},
    "mea.gov.in":         {"tier": "TIER_1", "credibility": 0.90},
    "mha.gov.in":         {"tier": "TIER_1", "credibility": 0.88},
    "rbi.org.in":         {"tier": "TIER_1", "credibility": 0.92},
    "sebi.gov.in":        {"tier": "TIER_1", "credibility": 0.90},
    "mospi.gov.in":       {"tier": "TIER_1", "credibility": 0.90},
    "imd.gov.in":         {"tier": "TIER_1", "credibility": 0.90},
    "unicef.org":         {"tier": "TIER_1", "credibility": 0.94},
    "worldbank.org":      {"tier": "TIER_1", "credibility": 0.93},
    "imf.org":            {"tier": "TIER_1", "credibility": 0.93},
    "ec.europa.eu":       {"tier": "TIER_1", "credibility": 0.93},
    "europa.eu":          {"tier": "TIER_1", "credibility": 0.92},
    "gov.uk":             {"tier": "TIER_1", "credibility": 0.92},
    "noaa.gov":           {"tier": "TIER_1", "credibility": 0.92},
    "nih.gov":            {"tier": "TIER_1", "credibility": 0.92},
    "fda.gov":            {"tier": "TIER_1", "credibility": 0.91},
    "icrc.org":           {"tier": "TIER_1", "credibility": 0.91},
    # Official publisher for a specific entity: primary for that entity's
    # history, but still only a prior on article-level support.
    "toureiffel.paris":   {"tier": "TIER_1", "credibility": 0.90},

    # TIER 2 — Major international outlets / strong regional outlets / official Indian news feeds
    "cbc.ca":             {"tier": "TIER_2", "credibility": 0.82},
    "abc.net.au":         {"tier": "TIER_2", "credibility": 0.82},
    "dw.com":             {"tier": "TIER_2", "credibility": 0.80},
    "france24.com":       {"tier": "TIER_2", "credibility": 0.78},
    "aljazeera.com":      {"tier": "TIER_2", "credibility": 0.78},
    "scmp.com":           {"tier": "TIER_2", "credibility": 0.80},
    "nikkeiasia.com":     {"tier": "TIER_2", "credibility": 0.82},
    "straitstimes.com":   {"tier": "TIER_2", "credibility": 0.80},
    "aa.com.tr":          {"tier": "TIER_2", "credibility": 0.78},
    "english.elpais.com": {"tier": "TIER_2", "credibility": 0.78},
    "news24.com":         {"tier": "TIER_2", "credibility": 0.76},

    "ptinews.com":        {"tier": "TIER_2", "credibility": 0.86},
    "ddnews.gov.in":      {"tier": "TIER_2", "credibility": 0.85},
    "newsonair.gov.in":   {"tier": "TIER_2", "credibility": 0.84},

    "moneycontrol.com":   {"tier": "TIER_2", "credibility": 0.76},
    "newindianexpress.com":{"tier": "TIER_2", "credibility": 0.78},
    "deccanherald.com":   {"tier": "TIER_2", "credibility": 0.78},
    "telegraphindia.com": {"tier": "TIER_2", "credibility": 0.76},
    "tribuneindia.com":   {"tier": "TIER_2", "credibility": 0.76},
    "theprint.in":        {"tier": "TIER_2", "credibility": 0.74},
    "firstpost.com":      {"tier": "TIER_2", "credibility": 0.72},
    "thewire.in":         {"tier": "TIER_2", "credibility": 0.70},
    "aninews.in":         {"tier": "TIER_2", "credibility": 0.74},
    "britannica.com":     {"tier": "TIER_2", "credibility": 0.85},

    # TIER 3 — Niche / aggregators / social / lower-trust / experimental
    "news18.com":         {"tier": "TIER_3", "credibility": 0.68},
    "outlookindia.com":   {"tier": "TIER_3", "credibility": 0.66},
    "scroll.in":          {"tier": "TIER_3", "credibility": 0.70},
    "thequint.com":       {"tier": "TIER_3", "credibility": 0.62},
    "indiatoday.in":      {"tier": "TIER_3", "credibility": 0.68},
    "dailyhunt.in":       {"tier": "TIER_3", "credibility": 0.45},
    "inshorts.com":       {"tier": "TIER_3", "credibility": 0.40},
    "yahoo.com":          {"tier": "TIER_3", "credibility": 0.55},

    "linkedin.com":       {"tier": "TIER_3", "credibility": 0.50},
    "medium.com":         {"tier": "TIER_3", "credibility": 0.45},
    "quora.com":          {"tier": "TIER_3", "credibility": 0.35},
    "reddit.com":         {"tier": "TIER_3", "credibility": 0.35},
    "facebook.com":       {"tier": "TIER_3", "credibility": 0.30},
    "twitter.com":        {"tier": "TIER_3", "credibility": 0.35},
    "x.com":              {"tier": "TIER_3", "credibility": 0.35},

    "xinhuanet.com":      {"tier": "TIER_3", "credibility": 0.50},
    "globaltimes.cn":     {"tier": "TIER_3", "credibility": 0.40},
    "rt.com":             {"tier": "TIER_3", "credibility": 0.35},
    "sputniknews.com":    {"tier": "TIER_3", "credibility": 0.30},

    # General reference, educational, and user-generated sources.
    # Wikipedia is crowd-edited: a moderate prior, never automatic truth.
    # Reddit/Facebook/YouTube/Stack Exchange remain below the evidence gate.
    "wikipedia.org":      {"tier": "TIER_3", "credibility": 0.65},
    "history.com":        {"tier": "TIER_3", "credibility": 0.72},
    "hnn.us":             {"tier": "TIER_3", "credibility": 0.65},
    "artsandculture.google.com": {"tier": "TIER_3", "credibility": 0.70},
    "history.stackexchange.com": {"tier": "TIER_3", "credibility": 0.45},
    "youtube.com":        {"tier": "TIER_3", "credibility": 0.35},
    "youtu.be":           {"tier": "TIER_3", "credibility": 0.35},
}


def extract_domain(url: str) -> str:
    """Extract and normalize the hostname from a URL or bare domain."""
    return normalize_hostname(url) or ""


def clear_source_cache() -> None:
    """Clear cached credibility lookups (used after registry changes/tests)."""
    _SOURCE_CACHE.clear()


async def get_credibility(url: str) -> dict:
    """
    Get the credibility score and tier for a URL's domain.

    Returns dict with: domain, name, tier, credibility. Tier 4 matches also
    include tier_number, blacklisted, category, and reason.
    """
    # Tier 4 must win over the cache, the database, and the built-in tiers.
    blacklist_match = match_blacklisted_source(url)
    if blacklist_match is not None:
        return {
            "domain": blacklist_match.input_host,
            "name": blacklist_match.name,
            "tier": blacklist_match.tier,
            "tier_number": blacklist_match.tier_number,
            "credibility": 0.00,
            "blacklisted": True,
            "category": blacklist_match.category,
            "reason": blacklist_match.reason,
        }

    domain = extract_domain(url)
    if not domain:
        return {
            "domain": "",
            "name": "Unknown",
            "tier": "UNKNOWN",
            "credibility": 0.40,
        }

    # Check in-memory cache first
    if domain in _SOURCE_CACHE:
        return _SOURCE_CACHE[domain]

    # Check database
    try:
        source = await ClaimDB.get_source_credibility(domain)
        if source:
            _SOURCE_CACHE[domain] = source
            return source
    except Exception as e:
        logger.debug(
            "credibility.db_error",
            domain=domain,
            error=redact_secrets(str(e)),
        )

    # Check built-in known sources
    if domain in KNOWN_SOURCES:
        known = KNOWN_SOURCES[domain]
        result = {
            "domain": domain,
            "name": domain,
            "tier": known["tier"],
            "credibility": known["credibility"],
        }
        _SOURCE_CACHE[domain] = result
        return result

    # Check if a parent domain matches (e.g., "news.bbc.co.uk" → "bbc.co.uk")
    parts = domain.split(".")
    for i in range(1, len(parts)):
        parent = ".".join(parts[i:])
        if parent in KNOWN_SOURCES:
            known = KNOWN_SOURCES[parent]
            result = {
                "domain": domain,
                "name": domain,
                "tier": known["tier"],
                "credibility": known["credibility"],
            }
            _SOURCE_CACHE[domain] = result
            return result

    # Default: unknown source — give a baseline of 0.50 instead of 0.40
    # to avoid overly punishing legitimate sources not in our map
    result = {
        "domain": domain,
        "name": domain,
        "tier": "UNKNOWN",
        "credibility": 0.50,
    }
    _SOURCE_CACHE[domain] = result
    return result


def is_primary_source(url: str, claim_entities: list[str]) -> bool:
    """
    Heuristic: determine if a URL is likely a primary source.

    Primary sources include:
    - Government domains (.gov)
    - Organization domains matching claim entities
    - Press release pages
    - Academic/journal sources
    """
    # A configured Tier 4 publisher can never be a primary source.
    if match_blacklisted_source(url) is not None:
        return False

    domain = extract_domain(url)
    url_lower = url.lower()

    # Government sites
    if domain.endswith(".gov") or domain.endswith(".gov.uk"):
        return True

    # International organizations
    primary_domains = {"who.int", "un.org", "europa.eu", "imf.org", "worldbank.org"}
    if domain in primary_domains:
        return True

    # Press releases
    if any(term in url_lower for term in ["press-release", "pressrelease", "media-release", "news-release"]):
        return True

    # Academic journals
    academic_domains = {"nature.com", "science.org", "thelancet.com", "nejm.org", "arxiv.org"}
    if domain in academic_domains:
        return True

    # Check if domain matches any claim entity
    for entity in claim_entities:
        entity_lower = entity.lower().replace(" ", "")
        domain_clean = domain.replace(".", "").replace("-", "")
        if entity_lower in domain_clean or domain_clean in entity_lower:
            return True

    return False
