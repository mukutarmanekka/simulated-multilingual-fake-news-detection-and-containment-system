"""Expanded source registry: credible priors, not automatic truth."""

from unittest.mock import AsyncMock

import pytest

from src.db.postgres import ClaimDB
from src.utils.source_credibility import (
    clear_source_cache,
    get_credibility,
)


@pytest.fixture(autouse=True)
def _clear_cache():
    clear_source_cache()
    yield
    clear_source_cache()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "url,tier,credibility",
    [
        ("https://www.toureiffel.paris/en/the-monument/history", "TIER_1", 0.90),
        ("https://www.britannica.com/topic/Eiffel-Tower", "TIER_2", 0.85),
        ("https://en.wikipedia.org/wiki/Great_Wall_of_China", "TIER_3", 0.65),
        ("https://www.history.com/topics/great-wall-of-china", "TIER_3", 0.72),
        ("https://hnn.us/article/123", "TIER_3", 0.65),
        ("https://artsandculture.google.com/story/eiffel", "TIER_3", 0.70),
        ("https://history.stackexchange.com/questions/1", "TIER_3", 0.45),
        ("https://www.youtube.com/watch?v=1", "TIER_3", 0.35),
        ("https://www.reddit.com/r/history", "TIER_3", 0.35),
        ("https://www.facebook.com/story", "TIER_3", 0.30),
    ],
)
async def test_builtin_source_tiers(url, tier, credibility, monkeypatch):
    monkeypatch.setattr(
        ClaimDB, "get_source_credibility", AsyncMock(return_value=None)
    )

    result = await get_credibility(url)

    assert result["tier"] == tier
    assert result["credibility"] == credibility


@pytest.mark.asyncio
async def test_unknown_domain_is_not_treated_as_fake(monkeypatch):
    monkeypatch.setattr(
        ClaimDB, "get_source_credibility", AsyncMock(return_value=None)
    )

    result = await get_credibility("https://unknown.example/article")

    assert result["tier"] == "UNKNOWN"
    assert result["credibility"] == 0.50


@pytest.mark.asyncio
async def test_wikipedia_and_social_media_are_not_high_credibility(monkeypatch):
    monkeypatch.setattr(
        ClaimDB, "get_source_credibility", AsyncMock(return_value=None)
    )

    wikipedia = await get_credibility("https://en.wikipedia.org/wiki/Test")
    reddit = await get_credibility("https://www.reddit.com/r/test")
    facebook = await get_credibility("https://www.facebook.com/test")

    assert wikipedia["credibility"] < 0.80
    assert reddit["credibility"] < 0.60
    assert facebook["credibility"] < 0.60


@pytest.mark.asyncio
async def test_tier_4_still_wins_over_registry_and_database(monkeypatch):
    # Even if a database row tried to assign a high tier, Tier 4 wins.
    monkeypatch.setattr(
        ClaimDB,
        "get_source_credibility",
        AsyncMock(
            return_value={
                "domain": "theonion.com",
                "name": "The Onion",
                "tier": "TIER_1",
                "credibility": 0.99,
            }
        ),
    )

    result = await get_credibility("https://www.theonion.com/article")

    assert result["tier"] == "BLACKLISTED"
    assert result["credibility"] == 0.0
    assert result["blacklisted"] is True
    assert result["category"] == "satire"
