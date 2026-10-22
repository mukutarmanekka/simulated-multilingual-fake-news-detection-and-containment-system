"""Step 1 — Claim Extraction.

Extracts factual, checkable claims from input text using an LLM
with spaCy NER validation. Filters out opinions, predictions,
and vague statements. Splits multi-claim sentences.
"""

from __future__ import annotations

import uuid
import structlog

from ..services.llm_service import llm_service
from ..models.schemas import ExtractedClaim, ClaimExtractionResult
from ..utils.redaction import redact_secrets

logger = structlog.get_logger()

SYSTEM_PROMPT = """You are an expert claim extraction system for a fake news detection pipeline.

Your task is to extract factual, checkable claims from the input text.

RULES:
1. Extract claims that can be verified against evidence — facts, events, statistics, scientific observations, causal claims, weather/climate events, policy actions, and statements attributed to named entities.
2. A claim IS CHECKABLE if it makes a factual assertion about the real world that can be confirmed or denied with evidence. This includes:
   - Current or recent events (elections, disasters, announcements)
   - Scientific and environmental claims (climate, weather, health)
   - Causal claims ("X caused Y", "X is because of Y")
   - Statistical claims (numbers, percentages, rankings)
   - Attributed statements ("Person X said Y")
3. A claim is NOT CHECKABLE only if it is purely a personal opinion, emotion, vague speculation with no specific assertion, or a rhetorical question.
4. When in doubt, TREAT IT AS CHECKABLE. It is far better to extract a claim that turns out to be borderline than to miss a genuinely checkable claim.
5. Split multi-claim sentences into SEPARATE individual claims.
6. Preserve exact named entities, dates, numbers, and locations from the original text.
7. Each claim must be a self-contained, unambiguous statement.
8. Do NOT alter the meaning of the original text.
9. If the input is a single short sentence making a factual assertion, extract it as a claim — do NOT discard it.
10. If the entire input genuinely contains NO checkable factual claims, return an empty claims list.

EXAMPLES OF CHECKABLE CLAIMS:
- "NASA confirmed an asteroid will hit Earth on June 5."
- "The unemployment rate dropped to 3.5% in March 2025."
- "Apple CEO Tim Cook announced a new iPhone model on January 10."
- "The temperature has been higher than usual because of the Super El Nino in 2026."
- "Global sea levels have risen by 3 inches in the last decade."
- "India's GDP grew by 7.2% in Q3 2025."
- "Elon Musk said Twitter will become a payments platform."
- "The 2026 hurricane season is expected to be the most active on record."

EXAMPLES OF NON-CHECKABLE STATEMENTS (only these should be excluded):
- "This politician is destroying the country." (pure opinion, no specific fact)
- "I feel like the economy is bad." (personal feeling)
- "What a terrible decision!" (exclamation/opinion)

Respond with valid JSON in this exact format:
{
  "claims": [
    {
      "claim_id": "C1",
      "text": "The exact factual claim text",
      "entities": ["Entity1", "Entity2"],
      "dates": ["2025-01-15"],
      "numbers": ["3.5%"],
      "is_checkable": true
    }
  ],
  "non_checkable_statements": ["Statement 1 that is purely opinion"]
}"""


async def extract_claims(input_text: str) -> ClaimExtractionResult:
    """
    Extract factual, checkable claims from input text.

    Uses LLM for intelligent extraction with structured output.
    Validates entities using spaCy NER as a secondary check.
    """
    logger.info("claim_extractor.start", text_len=len(input_text))

    # Call LLM for claim extraction
    user_prompt = f"""Extract all factual, checkable claims from the following text.
Remember: extract ANY statement that makes a factual assertion about the real world — including claims about events, science, weather, politics, statistics, and causal relationships. Only exclude pure personal opinions with no factual content.

TEXT:
\"\"\"
{input_text[:10000]}
\"\"\"

Return the structured JSON with extracted claims."""

    result = await llm_service.generate_json(
        system_prompt=SYSTEM_PROMPT,
        user_prompt=user_prompt,
        temperature=0.05,
        max_tokens=4096,
    )

    # Parse into Pydantic models
    claims = []
    for i, claim_data in enumerate(result.get("claims", [])):
        claim = ExtractedClaim(
            claim_id=str(uuid.uuid4()),  # Always use proper UUID, ignore LLM-generated IDs
            text=claim_data.get("text", ""),
            entities=claim_data.get("entities", []),
            dates=claim_data.get("dates", []),
            numbers=claim_data.get("numbers", []),
            is_checkable=claim_data.get("is_checkable", True),
        )
        if claim.text and claim.is_checkable:
            claims.append(claim)

    non_checkable = result.get("non_checkable_statements", [])

    # Optional: validate entities with spaCy
    try:
        claims = _validate_entities_spacy(claims)
    except Exception as e:
        logger.debug(
            "claim_extractor.spacy_validation_skipped",
            error=redact_secrets(str(e)),
        )

    logger.info(
        "claim_extractor.done",
        total_claims=len(claims),
        non_checkable=len(non_checkable),
    )

    return ClaimExtractionResult(
        claims=claims,
        non_checkable_statements=non_checkable,
    )


def _validate_entities_spacy(claims: list[ExtractedClaim]) -> list[ExtractedClaim]:
    """Use spaCy NER to validate and enrich entity extraction."""
    try:
        import spacy
        nlp = spacy.load("en_core_web_sm")
    except (ImportError, OSError):
        return claims  # spaCy not available, skip validation

    for claim in claims:
        doc = nlp(claim.text)

        # Extract entities from spaCy
        spacy_entities = [ent.text for ent in doc.ents if ent.label_ in (
            "PERSON", "ORG", "GPE", "LOC", "EVENT", "FAC", "NORP", "PRODUCT"
        )]

        # Merge with LLM-extracted entities (avoid duplicates)
        existing = set(e.lower() for e in claim.entities)
        for entity in spacy_entities:
            if entity.lower() not in existing:
                claim.entities.append(entity)
                existing.add(entity.lower())

        # Extract dates from spaCy
        spacy_dates = [ent.text for ent in doc.ents if ent.label_ == "DATE"]
        existing_dates = set(d.lower() for d in claim.dates)
        for date in spacy_dates:
            if date.lower() not in existing_dates:
                claim.dates.append(date)

        # Extract numbers from spaCy
        spacy_numbers = [ent.text for ent in doc.ents if ent.label_ in ("CARDINAL", "PERCENT", "MONEY", "QUANTITY")]
        existing_nums = set(n.lower() for n in claim.numbers)
        for num in spacy_numbers:
            if num.lower() not in existing_nums:
                claim.numbers.append(num)

    return claims
