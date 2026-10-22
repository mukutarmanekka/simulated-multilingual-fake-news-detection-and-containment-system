"""Evaluate the user-facing verdict policy on a labelled validation set.

This is an offline, deterministic policy-level evaluation. It does not call
DeepSeek, Tavily, or Fact Check. Each validation case supplies the verifier
verdict, subclaim verdicts, evidence credibility/relevance, and NLI stances.

Usage:
    python scripts/validate_verdict_policy.py
    python scripts/validate_verdict_policy.py --sweep
"""

from __future__ import annotations

import argparse
import inspect
import json
import os
import sys
from pathlib import Path

PIPELINE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PIPELINE_DIR))

from src.config import settings  # noqa: E402
from src.models.enums import Stance, UserVerdict, Verdict  # noqa: E402
from src.models.schemas import (  # noqa: E402
    EvidenceStance,
    RankedEvidence,
    RerankedEvidenceResult,
    SubclaimResult,
    VerificationResult,
)
from src.pipeline.verdict_policy import (  # noqa: E402
    VerdictDecision,
    decide_user_verdict,
)
from src.utils.blacklist import match_blacklisted_source  # noqa: E402

VALIDATION_PATH = PIPELINE_DIR / "data" / "validation_claims.json"
LABELS = ("REAL", "FAKE", "UNCERTAIN")


def _legacy_has_strong_evidence(
    verification: VerificationResult,
    evidence: RerankedEvidenceResult,
    stance: Stance,
    confidence: float,
    min_confidence: float,
) -> bool:
    """The pre-change confidence/stance gate, kept for before/after metrics."""
    if confidence < min_confidence:
        return False

    stance_doc_ids = {
        item.doc_id
        for item in verification.supporting_evidence
        if item.stance == stance
    }
    if not stance_doc_ids:
        return False

    usable = []
    for document in evidence.ranked_evidence:
        if document.doc_id not in stance_doc_ids:
            continue
        if document.credibility_score < settings.verdict_min_source_credibility:
            continue
        if document.relevance_score < settings.verdict_min_relevance:
            continue
        usable.append(document)

    if not usable:
        return False

    if len(usable) >= settings.verdict_min_corroborating_sources:
        return True

    for document in usable:
        if document.is_primary_source:
            return True
        if document.credibility_score >= settings.verdict_strong_source_credibility:
            return True

    return False


def legacy_decide_user_verdict(
    verification: VerificationResult,
    evidence: RerankedEvidenceResult,
    confidence: float,
) -> VerdictDecision:
    """Measurement-only copy of the pre-change policy."""
    internal = verification.verdict

    if internal == Verdict.SUPPORTED:
        strong = _legacy_has_strong_evidence(
            verification,
            evidence,
            Stance.SUPPORTS,
            confidence,
            settings.verdict_real_min_confidence,
        )
        if strong:
            return VerdictDecision(
                UserVerdict.REAL, "EVIDENCE_CORROBORATES_CLAIM", internal
            )
        return VerdictDecision(
            UserVerdict.UNCERTAIN,
            "INSUFFICIENT_OR_WEAK_EVIDENCE",
            internal,
        )

    if internal == Verdict.REFUTED:
        strong = _legacy_has_strong_evidence(
            verification,
            evidence,
            Stance.REFUTES,
            confidence,
            settings.verdict_fake_min_confidence,
        )
        if strong:
            return VerdictDecision(
                UserVerdict.FAKE, "EVIDENCE_CONTRADICTS_CLAIM", internal
            )
        return VerdictDecision(
            UserVerdict.UNCERTAIN,
            "INSUFFICIENT_OR_WEAK_EVIDENCE",
            internal,
        )

    if internal == Verdict.CONFLICTING:
        return VerdictDecision(
            UserVerdict.UNCERTAIN, "CONFLICTING_EVIDENCE", internal
        )

    if internal == Verdict.NOT_ENOUGH_EVIDENCE:
        reason = "NO_EVIDENCE"
        if evidence.ranked_evidence:
            reason = "INSUFFICIENT_OR_WEAK_EVIDENCE"
        return VerdictDecision(UserVerdict.UNCERTAIN, reason, internal)

    return VerdictDecision(
        UserVerdict.UNCERTAIN,
        "INSUFFICIENT_OR_WEAK_EVIDENCE",
        internal,
    )


def load_cases(path: Path = VALIDATION_PATH) -> list[dict]:
    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return payload["cases"]


def build_policy_inputs(case: dict):
    verification = VerificationResult(
        claim_id=case["id"],
        verdict=Verdict(case["verification"]["verdict"]),
        confidence=case["confidence"],
        reasoning=[],
        subclaim_results=[
            SubclaimResult(
                subclaim_text=f"subclaim-{index}",
                verdict=Verdict(verdict),
            )
            for index, verdict in enumerate(
                case["verification"].get("subclaims", [])
            )
        ],
        supporting_evidence=[
            EvidenceStance(
                doc_id=item["doc_id"],
                snippet="Validation evidence passage.",
                stance=Stance(item["stance"]),
                confidence=0.90,
            )
            for item in case.get("evidence", [])
        ],
    )

    ranked = RerankedEvidenceResult(
        claim_id=case["id"],
        ranked_evidence=[
            RankedEvidence(
                doc_id=item["doc_id"],
                title=item["domain"],
                url=f"https://{item['domain']}/article",
                source_domain=item["domain"],
                published_date="2026-01-01",
                credibility_score=item["credibility"],
                relevance_score=item["relevance"],
                freshness_score=0.80,
                final_rank_score=0.80,
                passages=["Validation evidence passage."],
                is_primary_source=item.get("is_primary", False),
            )
            for item in case.get("evidence", [])
        ],
    )
    return verification, ranked


def predict(case: dict, strength_threshold: float | None = None,
            confidence_floor: float | None = None,
            policy: str = "current") -> dict:
    blacklist_match = match_blacklisted_source(case.get("source_url"))
    if blacklist_match is not None:
        return {
            "verdict": UserVerdict.FAKE.value,
            "reason": blacklist_match.reason,
            "support_score": None,
        }

    verification, ranked = build_policy_inputs(case)
    if policy == "legacy":
        decision = legacy_decide_user_verdict(
            verification,
            ranked,
            case["confidence"],
        )
    else:
        kwargs = {}
        parameters = inspect.signature(decide_user_verdict).parameters
        if "strength_threshold" in parameters and strength_threshold is not None:
            kwargs["strength_threshold"] = strength_threshold
        if "confidence_floor" in parameters and confidence_floor is not None:
            kwargs["confidence_floor"] = confidence_floor

        decision = decide_user_verdict(
            verification,
            ranked,
            case["confidence"],
            **kwargs,
        )
    return {
        "verdict": decision.verdict.value,
        "reason": decision.reason,
        "support_score": getattr(decision, "support_score", None),
    }


def metrics(predictions: list[dict]) -> dict:
    confusion = {
        expected: {predicted: 0 for predicted in LABELS}
        for expected in LABELS
    }
    for row in predictions:
        confusion[row["expected"]][row["predicted"]] += 1

    totals = {label: sum(confusion[label].values()) for label in LABELS}
    precision = {}
    recall = {}
    f1 = {}
    for label in LABELS:
        true_positive = confusion[label][label]
        predicted_positive = sum(confusion[expected][label] for expected in LABELS)
        actual_positive = totals[label]
        precision[label] = (
            true_positive / predicted_positive if predicted_positive else 0.0
        )
        recall[label] = (
            true_positive / actual_positive if actual_positive else 0.0
        )
        p = precision[label]
        r = recall[label]
        f1[label] = 2 * p * r / (p + r) if (p + r) else 0.0

    total = len(predictions)
    correct = sum(
        1 for row in predictions if row["expected"] == row["predicted"]
    )

    real_total = totals["REAL"]
    fake_total = totals["FAKE"]

    return {
        "confusion": confusion,
        "totals": totals,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "macro_f1": sum(f1.values()) / len(LABELS),
        "accuracy": correct / total if total else 0.0,
        "uncertain_rate": (
            sum(1 for row in predictions if row["predicted"] == "UNCERTAIN") / total
            if total else 0.0
        ),
        "real_false_positive_rate": (
            confusion["REAL"]["FAKE"] / real_total if real_total else 0.0
        ),
        "fake_dangerous_false_negative_rate": (
            confusion["FAKE"]["REAL"] / fake_total if fake_total else 0.0
        ),
        "fake_not_flagged_rate": (
            (confusion["FAKE"]["REAL"] + confusion["FAKE"]["UNCERTAIN"]) / fake_total
            if fake_total else 0.0
        ),
    }


def evaluate(
    cases: list[dict],
    strength_threshold: float | None = None,
    confidence_floor: float | None = None,
    policy: str = "current",
) -> tuple[list[dict], dict]:
    rows = []
    for case in cases:
        result = predict(
            case,
            strength_threshold,
            confidence_floor,
            policy=policy,
        )
        rows.append(
            {
                "id": case["id"],
                "expected": case["expected_verdict"],
                "predicted": result["verdict"],
                "reason": result["reason"],
                "support_score": result["support_score"],
                "confidence": case["confidence"],
            }
        )
    return rows, metrics(rows)


def print_rows(rows: list[dict]) -> None:
    print(
        f"{'case':32s} {'expected':10s} {'predicted':10s} "
        f"{'support':>8s} {'conf':>5s}  reason"
    )
    for row in rows:
        support = (
            f"{row['support_score']:.3f}"
            if row["support_score"] is not None
            else "-"
        )
        marker = "OK" if row["expected"] == row["predicted"] else "MISS"
        print(
            f"{row['id']:32s} {row['expected']:10s} {row['predicted']:10s} "
            f"{support:>8s} {row['confidence']:>5.2f}  {row['reason']} [{marker}]"
        )


def print_metrics(result: dict, title: str = "METRICS") -> None:
    print(f"\n{title}")
    print("confusion (rows=expected, cols=predicted):")
    print(f"{'':10s} " + " ".join(f"{label:>10s}" for label in LABELS))
    for expected in LABELS:
        counts = result["confusion"][expected]
        print(
            f"{expected:10s} "
            + " ".join(f"{counts[predicted]:>10d}" for predicted in LABELS)
        )

    print("\nper-class:")
    for label in LABELS:
        print(
            f"  {label:10s} precision={result['precision'][label]:.3f} "
            f"recall={result['recall'][label]:.3f} "
            f"f1={result['f1'][label]:.3f}"
        )
    print(
        f"\nmacro_f1={result['macro_f1']:.3f} "
        f"accuracy={result['accuracy']:.3f} "
        f"uncertain_rate={result['uncertain_rate']:.3f}"
    )
    print(
        f"real_false_positive_rate (REAL->FAKE)="
        f"{result['real_false_positive_rate']:.3f}"
    )
    print(
        f"fake_dangerous_false_negative_rate (FAKE->REAL)="
        f"{result['fake_dangerous_false_negative_rate']:.3f}"
    )
    print(
        f"fake_not_flagged_rate (FAKE->REAL/UNCERTAIN)="
        f"{result['fake_not_flagged_rate']:.3f}"
    )


def sweep(cases: list[dict]) -> None:
    print(
        f"{'strength':>8s} {'floor':>6s} {'macroF1':>8s} {'realR':>6s} "
        f"{'fakeR':>6s} {'uncert':>7s} {'realFPR':>8s} {'fakeDanger':>10s}"
    )
    for threshold in (0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85):
        for floor in (0.35, 0.40, 0.45, 0.50):
            _, result = evaluate(cases, threshold, floor)
            print(
                f"{threshold:>8.2f} {floor:>6.2f} "
                f"{result['macro_f1']:>8.3f} "
                f"{result['recall']['REAL']:>6.3f} "
                f"{result['recall']['FAKE']:>6.3f} "
                f"{result['uncertain_rate']:>7.3f} "
                f"{result['real_false_positive_rate']:>8.3f} "
                f"{result['fake_dangerous_false_negative_rate']:>10.3f}"
            )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--strength-threshold",
        type=float,
        default=None,
        help="Override the evidence-strength threshold (new policy only).",
    )
    parser.add_argument(
        "--confidence-floor",
        type=float,
        default=None,
        help="Override the confidence floor (new policy only).",
    )
    parser.add_argument(
        "--sweep",
        action="store_true",
        help="Sweep evidence-strength thresholds and confidence floors.",
    )
    parser.add_argument(
        "--legacy",
        action="store_true",
        help="Evaluate the pre-change policy for before/after comparison.",
    )
    args = parser.parse_args()

    cases = load_cases()
    if args.sweep:
        print(f"validation cases: {len(cases)}")
        sweep(cases)
        return 0

    policy = "legacy" if args.legacy else "current"
    rows, result = evaluate(
        cases,
        args.strength_threshold,
        args.confidence_floor,
        policy=policy,
    )
    print(f"validation cases: {len(cases)}")
    print(f"policy: {policy}")
    print_rows(rows)
    print_metrics(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
