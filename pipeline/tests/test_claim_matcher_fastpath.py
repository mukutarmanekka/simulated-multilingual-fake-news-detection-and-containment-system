"""Semantic fast-path must not reuse verdicts for materially different claims."""

from unittest.mock import AsyncMock

import numpy as np
import pytest

from src.config import settings
from src.models.schemas import NormalizedClaim
from src.pipeline.claim_matcher import match_claim
from src.services.factcheck_service import factcheck_service
from src.services.vector_service import vector_service


def _claim(text: str) -> NormalizedClaim:
    return NormalizedClaim(
        claim_id="current",
        original_text=text,
        normalized_claim=text,
    )


def _vector_match(
    cached_text: str,
    score=0.97,
    verdict="SUPPORTED",
    confidence=0.9,
    policy_version=None,
):
    if policy_version is None:
        policy_version = settings.verdict_policy_version
    return {
        "claim_id": "cached",
        "score": score,
        "claim_text": cached_text,
        "verdict": verdict,
        "confidence": confidence,
        "policy_version": policy_version,
    }


class _FakeEncoder:
    def encode(self, text):
        return np.array([1.0, 0.0])


@pytest.mark.asyncio
async def test_vector_paris_berlin_does_not_reuse_verdict(monkeypatch):
    current = "The Louvre Museum is located in Berlin, Germany."
    cached = "The Louvre Museum is located in Paris, France."
    monkeypatch.setattr(
        vector_service,
        "search_similar",
        AsyncMock(return_value=[_vector_match(cached, score=0.99)]),
    )
    monkeypatch.setattr(
        factcheck_service,
        "search_claims",
        AsyncMock(return_value=[]),
    )

    result = await match_claim(_claim(current))

    assert result.matched_existing_claim is False


@pytest.mark.asyncio
async def test_vector_exact_duplicate_still_reuses_verdict(monkeypatch):
    text = "The Eiffel Tower was completed in 1889."
    monkeypatch.setattr(
        vector_service,
        "search_similar",
        AsyncMock(return_value=[_vector_match(text, confidence=0.9)]),
    )
    monkeypatch.setattr(
        factcheck_service,
        "search_claims",
        AsyncMock(return_value=[]),
    )

    result = await match_claim(_claim(text))

    assert result.matched_existing_claim is True
    assert result.match_source == "vector_db"
    assert result.previous_confidence == 0.9


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "current,cached",
    [
        (
            "The Eiffel Tower was completed in 1912.",
            "The Eiffel Tower was completed in 1889.",
        ),
        (
            "The project cost 300 million euros.",
            "The project cost 42 million euros.",
        ),
        (
            "The Louvre is not in Paris.",
            "The Louvre is in Paris.",
        ),
        (
            "Tim Cook announced the product.",
            "Satya Nadella announced the product.",
        ),
        (
            "The latest report says unemployment increased sharply across "
            "the region after the long consultation last year.",
            "The latest report says unemployment decreased sharply across "
            "the region after the long consultation last year.",
        ),
        (
            "सरकार ने नई दिल्ली में नई शिक्षा नीति की घोषणा की है।",
            "सरकार ने मुंबई में नई शिक्षा नीति की घोषणा की है।",
        ),
    ],
)
async def test_vector_material_differences_are_rejected(
    monkeypatch, current, cached
):
    monkeypatch.setattr(
        vector_service,
        "search_similar",
        AsyncMock(return_value=[_vector_match(cached, score=0.99)]),
    )
    monkeypatch.setattr(
        factcheck_service,
        "search_claims",
        AsyncMock(return_value=[]),
    )

    result = await match_claim(_claim(current))

    assert result.matched_existing_claim is False


@pytest.mark.asyncio
async def test_vector_falls_through_to_a_safe_lower_match(monkeypatch):
    current = "The Louvre Museum is located in Berlin, Germany."
    unsafe = "The Louvre Museum is located in Paris, France."
    safe = "The Louvre Museum is located in Berlin, Germany."
    monkeypatch.setattr(
        vector_service,
        "search_similar",
        AsyncMock(
            return_value=[
                _vector_match(unsafe, score=0.99),
                _vector_match(safe, score=0.95),
            ]
        ),
    )
    monkeypatch.setattr(
        factcheck_service,
        "search_claims",
        AsyncMock(return_value=[]),
    )

    result = await match_claim(_claim(current))

    assert result.matched_existing_claim is True
    assert result.matched_claim_text == safe


@pytest.mark.asyncio
async def test_factcheck_paris_berlin_does_not_reuse_rating(monkeypatch):
    current = "The Louvre Museum is located in Berlin, Germany."
    monkeypatch.setattr(
        vector_service, "search_similar", AsyncMock(return_value=[])
    )
    monkeypatch.setattr(vector_service, "_encoder", _FakeEncoder())
    monkeypatch.setattr(
        factcheck_service,
        "search_claims",
        AsyncMock(
            return_value=[
                {
                    "claim_text": "The Louvre Museum is located in Paris, France.",
                    "rating": "false",
                    "url": "https://example.test/factcheck",
                    "publisher": "Example Fact Check",
                }
            ]
        ),
    )

    result = await match_claim(_claim(current))

    assert result.matched_existing_claim is False


@pytest.mark.asyncio
async def test_factcheck_matching_claim_still_reuses_rating(monkeypatch):
    current = "The Louvre Museum is located in Berlin, Germany."
    matching = "The Louvre Museum is located in Berlin, Germany."
    monkeypatch.setattr(
        vector_service, "search_similar", AsyncMock(return_value=[])
    )
    monkeypatch.setattr(vector_service, "_encoder", _FakeEncoder())
    monkeypatch.setattr(
        factcheck_service,
        "search_claims",
        AsyncMock(
            return_value=[
                {
                    "claim_text": matching,
                    "rating": "false",
                    "url": "https://example.test/factcheck",
                    "publisher": "Example Fact Check",
                }
            ]
        ),
    )

    result = await match_claim(_claim(current))

    assert result.matched_existing_claim is True
    assert result.match_source == "factcheck_api"


@pytest.mark.asyncio
async def test_old_policy_version_is_rejected(monkeypatch):
    text = "The Eiffel Tower was completed in 1889."
    monkeypatch.setattr(
        vector_service,
        "search_similar",
        AsyncMock(
            return_value=[
                _vector_match(text, policy_version="old_policy")
            ]
        ),
    )
    monkeypatch.setattr(
        factcheck_service,
        "search_claims",
        AsyncMock(return_value=[]),
    )

    result = await match_claim(_claim(text))

    assert result.matched_existing_claim is False


@pytest.mark.asyncio
async def test_empty_cached_claim_text_is_rejected(monkeypatch):
    monkeypatch.setattr(
        vector_service,
        "search_similar",
        AsyncMock(return_value=[_vector_match("", score=0.99)]),
    )
    monkeypatch.setattr(
        factcheck_service,
        "search_claims",
        AsyncMock(return_value=[]),
    )

    result = await match_claim(_claim("Some claim."))

    assert result.matched_existing_claim is False
