

from __future__ import annotations

import structlog

from ..models.schemas import (
    RankedEvidence,
    RerankedEvidenceResult,
    VerificationResult,
    SubclaimResult,
    EvidenceStance,
)
from ..models.enums import Verdict, Stance
from ..services.llm_service import llm_service
from ..utils.redaction import redact_secrets
from ..config import settings

logger = structlog.get_logger()

_nli_model = None


def _get_nli_model():
    global _nli_model
    if _nli_model is None:
        from sentence_transformers import CrossEncoder
        logger.info("nli.loading", model=settings.nli_model)
        _nli_model = CrossEncoder(settings.nli_model)
        logger.info("nli.loaded")
    return _nli_model


VERIFICATION_PROMPT = """You are a rigorous fact-checker performing multi-step claim verification.

Given a CLAIM and EVIDENCE, you must:

1. Break the claim into atomic sub-claims (each verifiable independently).
2. For each sub-claim, assess what the evidence says.
3. Produce a chain-of-thought reasoning that CITES specific evidence snippets.
4. Determine a final verdict.

CRITICAL RULES:
- NEVER make up facts. Only use what is in the provided evidence.
- NEVER assume something is false just because it sounds unlikely.
- If evidence is insufficient, say NOT_ENOUGH_EVIDENCE — do NOT guess.
- If sources disagree, say CONFLICTING.
- Cite exact snippets from the evidence.
- Each reasoning step must reference a specific evidence document.
- Every sub-claim MUST be an assertion actually made by the CLAIM. Do NOT add
  background facts, context, or auxiliary true statements that the claim does
  not assert.
- If the claim makes one central assertion, return one sub-claim for it. If it
  makes several assertions, split those assertions.
- The FIRST sub-claim MUST be the central assertion of the claim. Later
  sub-claims are the claim's own secondary assertions, not background facts.

VERDICT VALUES: SUPPORTED, REFUTED, NOT_ENOUGH_EVIDENCE, CONFLICTING

Respond with JSON:
{
  "subclaims": [
    {
      "text": "atomic sub-claim text",
      "verdict": "SUPPORTED|REFUTED|NOT_ENOUGH_EVIDENCE|CONFLICTING",
      "reasoning": "Because evidence D1 states '...' which confirms/contradicts..."
    }
  ],
  "final_verdict": "SUPPORTED|REFUTED|NOT_ENOUGH_EVIDENCE|CONFLICTING",
  "reasoning": [
    "Step 1: The claim states X...",
    "Step 2: Evidence from source Y says '...'",
    "Step 3: Therefore the claim is supported/refuted because..."
  ]
}"""


async def verify_claim(
    claim_id: str,
    claim_text: str,
    evidence: RerankedEvidenceResult,
) -> VerificationResult:
    logger.info("claim_verifier.start", claim_id=claim_id, evidence_count=len(evidence.ranked_evidence))

    if not evidence.ranked_evidence:
        logger.info("claim_verifier.no_evidence", claim_id=claim_id)
        return VerificationResult(
            claim_id=claim_id,
            verdict=Verdict.NOT_ENOUGH_EVIDENCE,
            confidence=0.2,
            reasoning=["No evidence was retrieved for this claim."],
            subclaim_results=[],
            supporting_evidence=[],
        )

    evidence_stances = _classify_stances(claim_text, evidence.ranked_evidence)

    evidence_text = _format_evidence_for_llm(evidence.ranked_evidence, evidence_stances)

    user_prompt = f"""Verify this claim using ONLY the provided evidence.

CLAIM: "{claim_text}"

EVIDENCE:
{evidence_text}

Perform multi-step verification. Break the claim into sub-claims, analyze each against the evidence, and determine the final verdict. Always cite specific evidence snippets."""

    llm_result = await llm_service.generate_json(
        system_prompt=VERIFICATION_PROMPT,
        user_prompt=user_prompt,
        temperature=0.1,
        max_tokens=4096,
    )

    subclaim_results = []
    for sc in llm_result.get("subclaims", []):
        verdict_str = sc.get("verdict", "NOT_ENOUGH_EVIDENCE").upper()
        try:
            sc_verdict = Verdict(verdict_str)
        except ValueError:
            sc_verdict = Verdict.NOT_ENOUGH_EVIDENCE

        subclaim_results.append(SubclaimResult(
            subclaim_text=sc.get("text", ""),
            verdict=sc_verdict,
            reasoning=sc.get("reasoning", ""),
            supporting_evidence=[],
        ))

    final_verdict_str = llm_result.get("final_verdict", "NOT_ENOUGH_EVIDENCE").upper()
    try:
        final_verdict = Verdict(final_verdict_str)
    except ValueError:
        final_verdict = _aggregate_verdicts(subclaim_results, evidence_stances)

    supports = sum(1 for s in evidence_stances if s.stance == Stance.SUPPORTS)
    refutes = sum(1 for s in evidence_stances if s.stance == Stance.REFUTES)
    total_stances = len(evidence_stances) or 1
    support_ratio = supports / total_stances
    refute_ratio = refutes / total_stances

    # The LLM's multi-step chain-of-thought verification is the primary arbiter.
    # NLI stances are preserved in evidence_stances for UI display and confidence calibration.

    reasoning = llm_result.get("reasoning", [])
    if isinstance(reasoning, str):
        reasoning = [reasoning]

    result = VerificationResult(
        claim_id=claim_id,
        verdict=final_verdict,
        confidence=0.0,
        reasoning=reasoning,
        subclaim_results=subclaim_results,
        supporting_evidence=evidence_stances,
    )

    logger.info(
        "claim_verifier.done",
        claim_id=claim_id,
        verdict=final_verdict.value,
        subclaims=len(subclaim_results),
    )

    return result


def build_stance_label_map(id2label: dict | None) -> dict[int, Stance]:
    """Map the NLI model's output indices to pipeline stance labels.

    The label order is model-specific. For
    ``cross-encoder/nli-deberta-v3-base`` it is
    ``{0: contradiction, 1: entailment, 2: neutral}``. Hardcoding the
    wrong order silently turns neutral evidence into SUPPORTS and
    entailment into NEUTRAL. Prefer the model config and fall back to the
    standard NLI order only for generic ``LABEL_N`` configs.
    """
    fallback = {
        0: Stance.REFUTES,
        1: Stance.SUPPORTS,
        2: Stance.NEUTRAL,
    }

    if not id2label:
        return fallback

    labels = {str(label).strip().lower() for label in id2label.values()}
    if all(label.startswith("label_") for label in labels):
        return fallback

    mapping: dict[int, Stance] = {}
    for raw_index, raw_label in id2label.items():
        try:
            index = int(raw_index)
        except (TypeError, ValueError):
            continue

        label = str(raw_label).strip().lower()
        if "contradict" in label or "refut" in label:
            mapping[index] = Stance.REFUTES
        elif "entail" in label or "support" in label:
            mapping[index] = Stance.SUPPORTS
        else:
            mapping[index] = Stance.NEUTRAL

    if not mapping:
        return fallback
    return mapping


def _classify_stances(
    claim_text: str,
    evidence: list[RankedEvidence],
) -> list[EvidenceStance]:
    stances = []

    try:
        nli = _get_nli_model()
        model_config = getattr(getattr(nli, "model", None), "config", None)
        id2label = getattr(model_config, "id2label", None)
        label_map = build_stance_label_map(id2label)

        for doc in evidence:
            doc_text = " ".join(doc.passages[:2]) if doc.passages else doc.title
            if not doc_text:
                continue

            scores = nli.predict([(claim_text, doc_text[:512])])

            if hasattr(scores, 'tolist'):
                scores = scores.tolist()
            if isinstance(scores[0], list):
                scores = scores[0]

            best_idx = max(range(len(scores)), key=lambda i: scores[i])
            stance = label_map.get(best_idx, Stance.NEUTRAL)
            confidence = float(scores[best_idx])

            snippet = doc.passages[0][:300] if doc.passages else doc.title

            stances.append(EvidenceStance(
                doc_id=doc.doc_id,
                snippet=snippet,
                stance=stance,
                confidence=round(confidence, 3),
            ))

    except Exception as e:
        logger.warning("nli.error", error=redact_secrets(str(e)))
        for doc in evidence:
            stances.append(EvidenceStance(
                doc_id=doc.doc_id,
                snippet=doc.passages[0][:300] if doc.passages else doc.title,
                stance=Stance.NEUTRAL,
                confidence=0.33,
            ))

    return stances


def _format_evidence_for_llm(
    evidence: list[RankedEvidence],
    stances: list[EvidenceStance],
) -> str:
    parts = []
    stance_map = {s.doc_id: s for s in stances}

    for doc in evidence:
        stance_info = stance_map.get(doc.doc_id)
        stance_label = stance_info.stance.value if stance_info else "UNKNOWN"

        passages_text = "\n".join(f"  - {p[:500]}" for p in doc.passages) if doc.passages else "  (no passages extracted)"

        parts.append(f"""[{doc.doc_id}] {doc.title}
  Source: {doc.source_domain} (credibility: {doc.credibility_score:.2f})
  Published: {doc.published_date or 'Unknown'}
  NLI Stance: {stance_label}
  Primary Source: {'Yes' if doc.is_primary_source else 'No'}
  Passages:
{passages_text}""")

    return "\n\n".join(parts)


def _aggregate_verdicts(
    subclaim_results: list[SubclaimResult],
    stances: list[EvidenceStance],
) -> Verdict:
    if not subclaim_results:
        supports = sum(1 for s in stances if s.stance == Stance.SUPPORTS)
        refutes = sum(1 for s in stances if s.stance == Stance.REFUTES)

        if supports > 0 and refutes > 0:
            return Verdict.CONFLICTING
        if supports > refutes:
            return Verdict.SUPPORTED
        if refutes > supports:
            return Verdict.REFUTED
        return Verdict.NOT_ENOUGH_EVIDENCE

    verdict_counts = {}
    for sc in subclaim_results:
        verdict_counts[sc.verdict] = verdict_counts.get(sc.verdict, 0) + 1

    if Verdict.REFUTED in verdict_counts and Verdict.SUPPORTED in verdict_counts:
        return Verdict.CONFLICTING
    if Verdict.REFUTED in verdict_counts:
        return Verdict.REFUTED
    if Verdict.SUPPORTED in verdict_counts and len(verdict_counts) == 1:
        return Verdict.SUPPORTED
    if Verdict.NOT_ENOUGH_EVIDENCE in verdict_counts and len(verdict_counts) == 1:
        return Verdict.NOT_ENOUGH_EVIDENCE

    return Verdict.CONFLICTING
