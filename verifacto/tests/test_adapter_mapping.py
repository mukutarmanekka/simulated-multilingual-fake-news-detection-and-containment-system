"""Phase 4: the D5 verdict mapping is a pure function over claim dicts."""

from verifacto.detector.base import map_job_results

THRESHOLD = 0.5


def _claim(verdict, confidence):
    return {"verdict": verdict, "confidence": confidence}


def test_fake_on_any_refuted_above_threshold():
    result = map_job_results(
        [_claim("SUPPORTED", 0.9), _claim("REFUTED", 0.87)], THRESHOLD, provider="real"
    )
    assert result.is_fake is True
    assert result.confidence == 0.87
    assert result.verdict_summary == "REFUTED"


def test_real_on_supported_only():
    result = map_job_results(
        [_claim("SUPPORTED", 0.91), _claim("SUPPORTED", 0.6)], THRESHOLD, provider="real"
    )
    assert result.is_fake is False
    assert result.confidence == 0.91
    assert result.verdict_summary == "SUPPORTED"


def test_uncertain_on_not_enough_evidence():
    result = map_job_results(
        [_claim("NOT_ENOUGH_EVIDENCE", 0.4)], THRESHOLD, provider="real"
    )
    assert result.is_fake is None
    assert result.verdict_summary == "UNCERTAIN"


def test_uncertain_on_conflicting():
    result = map_job_results([_claim("CONFLICTING", 0.6)], THRESHOLD, provider="real")
    assert result.is_fake is None
    assert result.verdict_summary == "UNCERTAIN"


def test_uncertain_on_below_threshold_refuted():
    result = map_job_results([_claim("REFUTED", 0.49)], THRESHOLD, provider="real")
    assert result.is_fake is None


def test_no_claims_on_empty_results():
    result = map_job_results([], THRESHOLD, provider="real")
    assert result.is_fake is None
    assert result.verdict_summary == "NO_CLAIMS"


def test_multi_claim_aggregation_any_refuted_wins():
    # one refuted claim among supported ones makes the post fake
    result = map_job_results(
        [_claim("SUPPORTED", 0.8), _claim("REFUTED", 0.55), _claim("SUPPORTED", 0.7)],
        THRESHOLD,
        provider="real",
    )
    assert result.is_fake is True
