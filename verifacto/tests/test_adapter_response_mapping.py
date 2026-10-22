"""The real detector response maps into a rich, correct simulation record."""

from verifacto.detector.base import (
    VERDICT_FAKE,
    VERDICT_REAL,
    VERDICT_UNCERTAIN,
    DetectionResult,
    error_detection,
    map_response_dict,
)


def _claim(verdict, confidence=0.8, **extra):
    claim = {
        "claim": "A test claim.",
        "verdict": verdict,
        "confidence": confidence,
        "verdict_reason": "EVIDENCE_CORROBORATES_CLAIM",
        "verdict_support_score": 0.9,
        "evidence_verdict": "SUPPORTED",
        "requires_human_review": False,
        "escalation_reason": None,
        "evidence": [
            {
                "source": "toureiffel.paris",
                "url": "https://www.toureiffel.paris/history",
                "published_date": "2026-01-01",
                "snippet": "The tower was completed in 1889.",
                "stance": "NEUTRAL",
                "credibility_score": 0.9,
                "relevance_score": 0.9,
            }
        ],
    }
    claim.update(extra)
    return claim


def _map(response, verdict="REAL"):
    return map_response_dict(
        response,
        post_id=42,
        text="The Eiffel Tower was completed in 1889.",
        language="en",
        source_url="https://www.toureiffel.paris/history",
        provider="deepseek",
        detector_mode="real",
        latency_ms=1234.0,
        provider_usage={"llm": 4, "search": 4, "factcheck": 1},
        provider_status={
            "search_provider": "tavily",
            "search_error": None,
            "factcheck_status": 200,
            "factcheck_error": None,
        },
    )


def test_real_mapping_preserves_confidence_reason_and_evidence():
    result = _map(
        {
            "results": [_claim("REAL", confidence=0.54)],
            "detected_language": "en",
            "was_translated": False,
        }
    )

    assert result.post_id == 42
    assert result.verdict == VERDICT_REAL
    assert result.is_fake is False
    assert result.verdict_summary == "SUPPORTED"
    assert result.confidence == 0.54
    assert result.support_score == 0.9
    assert result.reason == "EVIDENCE_CORROBORATES_CLAIM"
    assert result.provider == "deepseek"
    assert result.detector_mode == "real"
    assert result.status == "ok"
    assert result.latency_ms == 1234.0
    assert result.provider_usage["llm"] == 4
    assert result.provider_status["search_provider"] == "tavily"
    assert result.provider_status["factcheck_status"] == 200
    assert result.evidence[0]["credibility_score"] == 0.9
    assert result.source_domain == "toureiffel.paris"


def test_fake_mapping_wins_over_uncertain_claim():
    result = _map(
        {
            "results": [
                _claim("UNCERTAIN", confidence=0.9),
                _claim("FAKE", confidence=0.51),
            ]
        }
    )

    assert result.verdict == VERDICT_FAKE
    assert result.is_fake is True
    assert result.verdict_summary == "REFUTED"
    assert result.confidence == 0.51


def test_uncertain_mapping_is_not_fake():
    result = _map({"results": [_claim("UNCERTAIN", confidence=0.5)]})

    assert result.verdict == VERDICT_UNCERTAIN
    assert result.is_fake is None
    assert result.verdict_summary == "UNCERTAIN"


def test_blacklist_metadata_is_preserved():
    result = _map(
        {
            "results": [
                _claim(
                    "FAKE",
                    confidence=1.0,
                    verdict_reason="BLACKLISTED_SATIRE_SOURCE",
                    matched_blacklist_domain="theonion.com",
                    blacklist_category="satire",
                    source_tier="BLACKLISTED",
                    source_tier_number=4,
                    evidence=[],
                )
            ]
        }
    )

    assert result.verdict == VERDICT_FAKE
    assert result.blacklisted is True
    assert result.blacklist_reason == "BLACKLISTED_SATIRE_SOURCE"
    assert result.matched_blacklist_domain == "theonion.com"
    assert result.blacklist_category == "satire"
    assert result.source_tier == "BLACKLISTED"


def test_no_claims_maps_to_uncertain_no_claims():
    result = _map({"results": []}, verdict="UNCERTAIN")

    assert result.verdict == VERDICT_UNCERTAIN
    assert result.is_fake is None
    assert result.verdict_summary == "NO_CLAIMS"
    assert result.status == "no_claims"
    assert result.reason == "NO_CHECKABLE_CLAIMS"


def test_error_detection_is_not_a_verdict():
    result = error_detection(
        post_id=1,
        text="x",
        language="en",
        source_url="https://example.com/x",
        provider="deepseek",
        detector_mode="real",
        status="timeout",
        error="sk-secret123456789 timed out",
        latency_ms=300000,
        provider_usage={"llm": 1},
    )

    assert isinstance(result, DetectionResult)
    assert result.verdict is None
    assert result.is_fake is None
    assert result.status == "timeout"
    assert result.verdict_summary == "ERROR"
    assert "sk-secret" not in result.error
