"""Phase 4: the replay detector returns cached verdicts or fails loudly."""

import json

import pytest

from verifacto.detector.base import (
    DetectionResult,
    DetectorCacheMiss,
    make_cache_key,
)
from verifacto.detector.replay import ReplayDetector


def _write_cache(path, verdicts):
    payload = {
        "meta": {"adapter": "real"},
        "verdicts": {
            str(post_id): {
                "provider": "gemini",
                "mapped": {
                    "is_fake": is_fake,
                    "confidence": confidence,
                    "verdict_summary": summary,
                },
            }
            for post_id, (is_fake, confidence, summary) in verdicts.items()
        },
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_legacy_roundtrip(tmp_path):
    cache_path = tmp_path / "cache.json"
    _write_cache(cache_path, {0: (True, 0.8, "REFUTED"), 3: (False, 0.6, "SUPPORTED")})
    detector = ReplayDetector(cache_path)

    result = detector.detect("x", "en", 0)
    assert result.is_fake is True
    assert result.confidence == 0.8
    assert result.verdict_summary == "REFUTED"
    assert result.verdict == "FAKE"
    assert result.provider == "gemini"
    assert result.detector_mode == "replay"
    assert result.post_id == 0
    assert result.cache_key_verified is False

    assert detector.detect("x", "en", 3).is_fake is False


def test_missing_post_raises(tmp_path):
    cache_path = tmp_path / "cache.json"
    _write_cache(cache_path, {0: (True, 0.8, "REFUTED")})
    detector = ReplayDetector(cache_path)
    with pytest.raises(DetectorCacheMiss):
        detector.detect("x", "en", 99)


def _new_cache(
    path,
    text,
    language,
    source_url,
    verdict="FAKE",
    provider="deepseek",
    model="deepseek-chat",
    policy_version="verdict_policy_v2",
):
    record = DetectionResult(
        is_fake=True,
        confidence=0.91,
        verdict_summary="REFUTED",
        provider="deepseek",
        verdict=verdict,
        detector_mode="real",
        post_id=7,
        claim_text=text,
        language=language,
        source_url=source_url,
        support_score=0.93,
        reason="EVIDENCE_CONTRADICTS_CLAIM",
        evidence=[{"source": "history.com", "stance": "NEUTRAL"}],
        status="ok",
    )
    payload = {
        "meta": {"adapter": "real", "detector_mode": "real"},
        "verdicts": {
            "7": {
                "provider": provider,
                "model": model,
                "cache_key": make_cache_key(text, language, source_url),
                "policy_version": policy_version,
                "record": record.to_dict(),
                "mapped": {
                    "is_fake": True,
                    "confidence": 0.91,
                    "verdict_summary": "REFUTED",
                    "verdict": "FAKE",
                },
            }
        },
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_cache_key_matches_claim_language_and_source(tmp_path):
    cache_path = tmp_path / "cache.json"
    _new_cache(
        cache_path,
        "The Great Wall of China was built by the Romans.",
        "en",
        "https://www.britannica.com/topic/Great-Wall-of-China",
    )
    detector = ReplayDetector(cache_path)

    result = detector.detect(
        "The Great Wall of China was built by the Romans.",
        "en",
        7,
        "https://www.britannica.com/topic/Great-Wall-of-China",
    )

    assert result.verdict == "FAKE"
    assert result.support_score == 0.93
    assert result.reason == "EVIDENCE_CONTRADICTS_CLAIM"
    assert result.evidence[0]["source"] == "history.com"
    assert result.detector_mode == "replay"
    assert result.cache_key_verified is True


def test_cache_key_mismatch_fails_loudly(tmp_path):
    cache_path = tmp_path / "cache.json"
    text = "The Great Wall of China was built by the Romans."
    source = "https://www.britannica.com/topic/Great-Wall-of-China"
    _new_cache(cache_path, text, "en", source)
    detector = ReplayDetector(cache_path)

    with pytest.raises(DetectorCacheMiss):
        detector.detect(text, "fr", 7, source)
    with pytest.raises(DetectorCacheMiss):
        detector.detect(text, "en", 7, "https://example.com/other")


def test_provider_or_model_mismatch_fails_loudly(tmp_path):
    cache_path = tmp_path / "cache.json"
    text = "The Great Wall of China was built by the Romans."
    source = "https://www.britannica.com/topic/Great-Wall-of-China"
    _new_cache(cache_path, text, "en", source)

    with pytest.raises(DetectorCacheMiss):
        ReplayDetector(
            cache_path, expected_provider="gemini"
        ).detect(text, "en", 7, source)
    with pytest.raises(DetectorCacheMiss):
        ReplayDetector(
            cache_path, expected_model="other-model"
        ).detect(text, "en", 7, source)

    result = ReplayDetector(
        cache_path,
        expected_provider="deepseek",
        expected_model="deepseek-chat",
    ).detect(text, "en", 7, source)
    assert result.verdict == "FAKE"


def test_old_policy_version_fails_loudly(tmp_path):
    cache_path = tmp_path / "cache.json"
    text = "The Great Wall of China was built by the Romans."
    source = "https://www.britannica.com/topic/Great-Wall-of-China"
    _new_cache(
        cache_path,
        text,
        "en",
        source,
        policy_version="old_policy",
    )

    with pytest.raises(DetectorCacheMiss):
        ReplayDetector(cache_path).detect(text, "en", 7, source)
