"""Phase 4: the replay detector returns cached verdicts or fails loudly."""

import json

import pytest

from verifacto.detector.base import DetectorCacheMiss, DetectionResult
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


def test_roundtrip(tmp_path):
    cache_path = tmp_path / "cache.json"
    _write_cache(cache_path, {0: (True, 0.8, "REFUTED"), 3: (False, 0.6, "SUPPORTED")})
    detector = ReplayDetector(cache_path)
    assert detector.detect("x", "en", 0) == DetectionResult(
        is_fake=True, confidence=0.8, verdict_summary="REFUTED", provider="gemini"
    )
    assert detector.detect("x", "en", 3).is_fake is False


def test_missing_post_raises(tmp_path):
    cache_path = tmp_path / "cache.json"
    _write_cache(cache_path, {0: (True, 0.8, "REFUTED")})
    detector = ReplayDetector(cache_path)
    with pytest.raises(DetectorCacheMiss):
        detector.detect("x", "en", 99)
