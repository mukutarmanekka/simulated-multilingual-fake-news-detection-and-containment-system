"""Phase 4: the degraded-verdict guard never caches pipeline errors."""

from verifacto.detector.precompute import _claims_are_degraded


def _pipeline_error_claim():
    return {
        "claim": "x",
        "verdict": "NOT_ENOUGH_EVIDENCE",
        "confidence": 0.0,
        "requires_human_review": True,
        "escalation_reason": "Pipeline error: quota exhausted",
    }


def test_pipeline_error_claims_are_degraded():
    assert _claims_are_degraded([_pipeline_error_claim()]) is True


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
