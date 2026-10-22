"""Phase 4: the degraded-verdict guard never caches pipeline errors."""

from verifacto.config import DETECTOR_POLICY_VERSION
from verifacto.data.posts import Post
from verifacto.detector.base import DetectionResult
from verifacto.detector.precompute import (
    _cache_entry,
    _claims_are_degraded,
    _entry_status,
    _providers_meta,
)


def _pipeline_error_claim():
    return {
        "claim": "x",
        "verdict": "NOT_ENOUGH_EVIDENCE",
        "confidence": 0.0,
        "requires_human_review": True,
        "escalation_reason": "Pipeline error: quota exhausted",
    }


def _pipeline_error_claim_three_way():
    return {
        "claim": "x",
        "verdict": "UNCERTAIN",
        "evidence_verdict": "NOT_ENOUGH_EVIDENCE",
        "verdict_reason": "PIPELINE_ERROR",
        "confidence": 0.0,
        "requires_human_review": True,
        "escalation_reason": "Pipeline error: quota exhausted",
    }


def test_pipeline_error_claims_are_degraded():
    assert _claims_are_degraded([_pipeline_error_claim()]) is True


def test_three_way_pipeline_error_claims_are_degraded():
    assert _claims_are_degraded([_pipeline_error_claim_three_way()]) is True


def test_three_way_genuine_uncertain_is_not_degraded():
    genuine = _pipeline_error_claim_three_way()
    genuine["confidence"] = 0.3
    assert _claims_are_degraded([genuine]) is False


def test_uncertain_with_real_evidence_is_not_degraded():
    claim = _pipeline_error_claim_three_way()
    claim["evidence_verdict"] = "CONFLICTING"
    assert _claims_are_degraded([claim]) is False


def test_genuine_low_evidence_is_not_degraded():
    genuine = _pipeline_error_claim()
    genuine["confidence"] = 0.3
    assert _claims_are_degraded([genuine]) is False


def test_genuine_escalation_is_not_degraded():
    genuine = _pipeline_error_claim()
    genuine["escalation_reason"] = "High-impact health claim with low confidence"
    assert _claims_are_degraded([genuine]) is False


def test_mixed_claims_are_not_degraded():
    # one error claim among healthy ones still holds information
    healthy = {
        "claim": "y",
        "verdict": "SUPPORTED",
        "confidence": 0.8,
        "requires_human_review": False,
        "escalation_reason": None,
    }
    assert _claims_are_degraded([_pipeline_error_claim(), healthy]) is False


def test_no_claims_is_not_degraded():
    # a post with no checkable claims is a genuine outcome
    assert _claims_are_degraded([]) is False


def test_entry_status_classification():
    assert _entry_status(None) == "missing"
    assert _entry_status({}) == "legacy"
    assert _entry_status({"policy_version": "old_policy"}) == "stale"
    assert (
        _entry_status({"policy_version": DETECTOR_POLICY_VERSION})
        == "current"
    )
    assert (
        _entry_status(
            {
                "policy_version": DETECTOR_POLICY_VERSION,
                "provider": "gemini",
                "model": None,
            },
            current_provider="deepseek",
            current_model="deepseek-flash",
        )
        == "stale"
    )
    assert (
        _entry_status(
            {
                "policy_version": DETECTOR_POLICY_VERSION,
                "provider": "deepseek",
                "model": "deepseek-flash",
            },
            current_provider="deepseek",
            current_model="deepseek-flash",
        )
        == "current"
    )


def test_cache_entry_records_provider_and_model():
    result = DetectionResult(
        is_fake=True, confidence=0.8, verdict_summary="REFUTED", provider="ollama"
    )
    entry = _cache_entry(result, [], "ollama", "deepseek-r1:8b")
    assert entry["provider"] == "ollama"
    assert entry["model"] == "deepseek-r1:8b"
    assert entry["mapped"]["is_fake"] is True
    assert entry["mapped"]["verdict"] == "FAKE"


def test_cache_entry_records_full_detection_and_cache_key():
    post = Post(
        id=1,
        text="t",
        true_label="fake",
        author=0,
        language="en",
        source_url="https://example.com/x",
    )
    result = DetectionResult(
        is_fake=True,
        confidence=0.8,
        verdict_summary="REFUTED",
        provider="deepseek",
        verdict="FAKE",
        detector_mode="real",
        post_id=1,
        claim_text=post.text,
        language=post.language,
        source_url=post.source_url,
        support_score=0.9,
        reason="EVIDENCE_CONTRADICTS_CLAIM",
        evidence=[{"source": "history.com", "stance": "NEUTRAL"}],
        status="ok",
        latency_ms=123.0,
        provider_usage={"llm": 4, "search": 4},
    )
    entry = _cache_entry(
        result,
        [],
        "deepseek",
        "deepseek-chat",
        post=post,
        cache_key_value="abc123",
    )

    assert entry["cache_key"] == "abc123"
    assert entry["policy_version"] == DETECTOR_POLICY_VERSION
    assert entry["post_id"] == 1
    assert entry["record"]["verdict"] == "FAKE"
    assert entry["record"]["support_score"] == 0.9
    assert entry["record"]["evidence"][0]["source"] == "history.com"
    assert entry["record"]["provider_usage"]["llm"] == 4
    assert entry["mapped"]["verdict"] == "FAKE"


def test_providers_meta_summarizes_per_post():
    cache = {
        "verdicts": {
            "0": {"provider": "gemini", "model": "gemini-2.5-flash-lite"},
            "1": {"provider": "ollama", "model": "deepseek-r1:8b"},
            "2": {"provider": "ollama", "model": "deepseek-r1:8b"},
        }
    }
    meta = _providers_meta(cache)
    assert meta["per_post"]["0"] == {"provider": "gemini", "model": "gemini-2.5-flash-lite"}
    assert meta["per_post"]["2"] == {"provider": "ollama", "model": "deepseek-r1:8b"}
    used = {(item["provider"], item["model"]): item["posts"] for item in meta["used"]}
    assert used[("gemini", "gemini-2.5-flash-lite")] == 1
    assert used[("ollama", "deepseek-r1:8b")] == 2
