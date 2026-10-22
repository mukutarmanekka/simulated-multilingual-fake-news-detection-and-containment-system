"""Regression guard for the labelled verdict-policy validation set.

These assertions are policy-level, not an end-to-end accuracy claim. The
live integration suite separately exercises real retrieval and APIs.
"""

from scripts.validate_verdict_policy import evaluate, load_cases


def _evaluate_validation_set():
    cases = load_cases()
    rows, result = evaluate(cases)
    return rows, result


def test_validation_set_macro_quality():
    _, result = _evaluate_validation_set()

    assert result["macro_f1"] >= 0.95
    assert result["recall"]["REAL"] >= 0.95
    assert result["recall"]["FAKE"] >= 0.95
    assert result["real_false_positive_rate"] == 0.0
    assert result["fake_dangerous_false_negative_rate"] == 0.0
    assert result["uncertain_rate"] <= 0.60


def test_live_examples_get_evidence_supported_verdicts():
    rows, _ = _evaluate_validation_set()
    by_id = {row["id"]: row for row in rows}

    assert by_id["real_official_eiffel"]["predicted"] == "REAL"
    assert by_id["fake_great_wall"]["predicted"] == "FAKE"


def test_ambiguous_and_weak_cases_remain_uncertain():
    rows, _ = _evaluate_validation_set()
    by_id = {row["id"]: row for row in rows}

    for case_id in (
        "ambiguous_budget",
        "ambiguous_conflicting",
        "ambiguous_no_evidence",
        "ambiguous_refuted_weak",
        "real_single_unknown",
        "fake_single_weak",
        "real_low_confidence",
        "fake_low_confidence",
        "real_irrelevant",
        "fake_irrelevant",
        "real_conflicting_subclaims",
        "fake_conflicting_subclaims",
        "real_nli_contradicts",
        "fake_nli_supports",
    ):
        assert by_id[case_id]["predicted"] == "UNCERTAIN", case_id
