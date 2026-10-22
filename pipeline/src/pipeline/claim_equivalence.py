"""Material-fact guard for semantic cache reuse.

Sentence embeddings can score contradictory claims as highly similar:
"The Louvre is in Paris" and "The Louvre is in Berlin" differ by one token
but have opposite truth conditions. A cached verdict may only be reused
when the claims are materially the same, not merely topically similar.

The guard is intentionally conservative and dependency-free:
- exact normalized text match is always safe;
- otherwise proper-noun entities, dates, numbers/quantities, and negation
  must match exactly;
- content-token overlap must clear a threshold to avoid reusing verdicts
  for claims like "reduced unemployment" vs "increased unemployment";
- any remaining lexical substitution must be an inflection of the same
  word or a known paraphrase. A changed content word can reverse a claim
  even when every entity, number, and date matches ("increased" vs
  "decreased", "safe" vs "unsafe"), and uncased scripts such as Hindi
  carry place names that capitalization-based entity checks cannot see.
  Unknown substitutions therefore trigger full verification instead of
  reusing a cached verdict.

When the check is uncertain, callers should continue full verification.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Iterable

# Articles/pronouns/very common predicates are not material facts. This is
# deliberately small: the content-overlap check handles ordinary paraphrase.
_ENTITY_LEADING_STOPWORDS = frozenset(
    {
        "the", "a", "an", "this", "that", "these", "those",
        "according", "report", "reports", "study", "studies",
        "scientists", "researchers", "doctors", "officials",
    }
)

_CONTENT_STOPWORDS = frozenset(
    {
        "a", "an", "the", "and", "or", "but", "if", "then", "than",
        "is", "are", "was", "were", "be", "been", "being", "am",
        "has", "have", "had", "do", "does", "did", "will", "would",
        "can", "could", "should", "may", "might", "must", "shall",
        "to", "of", "in", "on", "at", "by", "for", "from", "with",
        "about", "into", "over", "under", "between", "among", "as",
        "it", "its", "their", "there", "here", "which", "who", "whom",
        "whose", "what", "when", "where", "why", "how", "that", "this",
        "these", "those", "he", "she", "they", "we", "you", "i",
        "his", "her", "our", "your", "my", "me", "him", "them", "us",
        "not", "no", "never", "none", "without", "neither", "nor",
    }
)

_NEGATIONS = frozenset(
    {
        "not", "no", "never", "none", "without", "cannot",
        "isn't", "isnt", "wasn't", "wasnt", "aren't", "arent",
        "weren't", "werent", "doesn't", "doesnt", "didn't", "didnt",
        "don't", "dont", "won't", "wont", "wouldn't", "wouldnt",
        "shouldn't", "shouldnt", "couldn't", "couldnt", "hardly",
        "barely", "neither", "nor",
        # French
        "ne", "pas", "jamais", "aucun", "aucune", "sans", "rien",
        # Hindi (common negation markers)
        "नहीं", "न", "मत", "बिना", "कभी",
    }
)

# Text reaching this pattern has already had punctuation normalised to
# spaces, so a token is any run of letters/marks/numbers. The explicit
# ranges keep Devanagari matras and other combining marks intact.
_TOKEN_RE = re.compile(
    r"[^\s%€$£₹'’\-]+(?:['’\-][^\s%€$£₹'’\-]+)*",
    re.UNICODE,
)
_ENTITY_RE = re.compile(
    r"\b[A-ZÀ-ÖØ-Þ][\w'’\-]*(?:\s+[A-ZÀ-ÖØ-Þ][\w'’\-]*)*\b",
    re.UNICODE,
)
_YEAR_RE = re.compile(r"\b(?:1[5-9]\d{2}|20\d{2}|21\d{2})\b")
_ISO_DATE_RE = re.compile(r"\b\d{4}-\d{1,2}-\d{1,2}\b")
_SLASH_DATE_RE = re.compile(r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b")
_MONTH_DATE_RE = re.compile(
    r"\b(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|"
    r"Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|"
    r"Nov(?:ember)?|Dec(?:ember)?)\s+\d{1,2}(?:,?\s+\d{4})?\b",
    re.IGNORECASE,
)
_NUMBER_RE = re.compile(
    r"\b\d+(?:[.,]\d+)*\s*"
    r"(?:(?:%|percent|million|billion|trillion|thousand|"
    r"km|kilomet(?:er|re)s?|kg|kilograms?|lb|pounds?|"
    r"m|meters?|metres?|feet|ft|miles?|mph|kph|"
    r"degrees?|celsius|fahrenheit|people|users?|years?|"
    r"days?|hours?)\s*)?"
    r"(?:dollars?|euros?|rupees?|usd|eur|inr|gbp)?\b",
    re.IGNORECASE,
)

# A deliberately small set of news-text paraphrase pairs whose substitution
# does not change the fact being checked. Unknown substitutions are treated
# as material changes and re-verified.
_PARAPHRASE_PAIRS = frozenset(
    {
        frozenset({"built", "constructed"}),
        frozenset({"located", "situated"}),
        frozenset({"announced", "unveiled"}),
        frozenset({"started", "began"}),
        frozenset({"ended", "finished"}),
        frozenset({"showed", "demonstrated"}),
        frozenset({"said", "stated"}),
        frozenset({"bought", "purchased"}),
    }
)

_STEM_SUFFIXES = ("ingly", "edly", "ing", "ed", "es", "s", "d", "ly")


@dataclass(frozen=True)
class CriticalFacts:
    entities: frozenset[str]
    entity_sequence: tuple[str, ...]
    dates: frozenset[str]
    numbers: frozenset[str]
    negated: bool
    content_tokens: frozenset[str]
    content_sequence: tuple[str, ...]
    content_bigrams: frozenset[tuple[str, str]]


@dataclass(frozen=True)
class EquivalenceCheck:
    equivalent: bool
    reason: str


def _normalise_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text or "")
    text = text.lower()
    text = text.replace("’", "'").replace("‘", "'")

    kept = []
    for character in text:
        if character.isspace() or character in "%€$£₹'’-":
            kept.append(character)
            continue
        # Unicode marks (category M) carry Devanagari matras and other
        # combining vowels; dropping them turns words into fragments.
        category = unicodedata.category(character)
        if category[0] in ("L", "M", "N"):
            kept.append(character)
        else:
            kept.append(" ")
    return re.sub(r"\s+", " ", "".join(kept)).strip()


def _normalise_number(value: str) -> str:
    value = value.lower().strip()
    value = value.replace(",", "")
    value = re.sub(r"\s+", " ", value)
    return value


def _stem(token: str) -> str:
    """Reduce simple inflections so approve/approved compare equal."""
    token = token.lower()
    for suffix in _STEM_SUFFIXES:
        if token.endswith(suffix) and len(token) - len(suffix) >= 3:
            token = token[: -len(suffix)]
            break
    if len(token) > 4 and token.endswith("e"):
        token = token[:-1]
    if len(token) > 3 and token.endswith("i"):
        token = token[:-1] + "y"
    if (
        len(token) > 4
        and token[-1] == token[-2]
        and token[-1] not in "aeiou"
    ):
        token = token[:-1]
    return token


def _tokens_are_equivalent_variants(left: str, right: str) -> bool:
    if left == right:
        return True
    if frozenset({left, right}) in _PARAPHRASE_PAIRS:
        return True
    return _stem(left) == _stem(right)


def _entity_mentions(text: str) -> tuple[str, ...]:
    mentions = []
    for match in _ENTITY_RE.finditer(text or ""):
        phrase = match.group(0).strip()
        words = phrase.split()
        while words and words[0].lower() in _ENTITY_LEADING_STOPWORDS:
            words.pop(0)
        if not words:
            continue
        # A single sentence-initial capitalised common word is not a fact.
        if len(words) == 1 and words[0].lower() in _CONTENT_STOPWORDS:
            continue
        mentions.append(" ".join(words).lower())
    return tuple(mentions)


def _numbers_and_dates(text: str) -> tuple[frozenset[str], frozenset[str]]:
    dates = set()
    for pattern in (_ISO_DATE_RE, _SLASH_DATE_RE, _MONTH_DATE_RE, _YEAR_RE):
        for match in pattern.findall(text or ""):
            dates.add(_normalise_number(match))

    numbers = set()
    for match in _NUMBER_RE.findall(text or ""):
        if match.strip():
            numbers.add(_normalise_number(match))

    # Years are also quantities; keeping them in both sets is harmless.
    return frozenset(numbers), frozenset(dates)


def extract_critical_facts(text: str) -> CriticalFacts:
    """Extract the facts whose change can reverse a claim's truth."""
    normalised = _normalise_text(text)
    tokens = _TOKEN_RE.findall(normalised)
    numbers, dates = _numbers_and_dates(text or "")

    negated = any(token.lower() in _NEGATIONS for token in tokens)

    content_sequence = [
        token.lower()
        for token in tokens
        if token.lower() not in _CONTENT_STOPWORDS
        and not token.isdigit()
        and token.lower() not in _NEGATIONS
    ]
    entity_sequence = _entity_mentions(text or "")
    content_bigrams = frozenset(
        zip(content_sequence, content_sequence[1:])
    )

    return CriticalFacts(
        entities=frozenset(entity_sequence),
        entity_sequence=entity_sequence,
        dates=dates,
        numbers=numbers,
        negated=negated,
        content_tokens=frozenset(content_sequence),
        content_sequence=tuple(content_sequence),
        content_bigrams=content_bigrams,
    )


def _jaccard(left: Iterable, right: Iterable) -> float:
    left_set = set(left)
    right_set = set(right)
    union = left_set | right_set
    if not union:
        return 0.0
    return len(left_set & right_set) / len(union)


def _has_unverified_substitution(
    current: CriticalFacts,
    cached: CriticalFacts,
) -> bool:
    """True when a content word was replaced by an unrelated one.

    Pure additions/deletions (for example adding "located") are left to
    the overlap and entity checks. Replacing one content word with an
    unrelated word can reverse the claim even when all entities, dates,
    and numbers match: "increased"/"decreased", "safe"/"unsafe", or a
    place name written in a script without capitalization (for example
    Hindi). Such a change must be re-verified unless the words are known
    inflections or paraphrases of each other.
    """
    if not current.content_sequence or not cached.content_sequence:
        return False

    matcher = SequenceMatcher(
        None,
        current.content_sequence,
        cached.content_sequence,
        autojunk=False,
    )
    for tag, start_a, end_a, start_b, end_b in matcher.get_opcodes():
        if tag != "replace":
            continue
        current_tokens = current.content_sequence[start_a:end_a]
        cached_tokens = cached.content_sequence[start_b:end_b]
        if len(current_tokens) != len(cached_tokens):
            return True
        for left, right in zip(current_tokens, cached_tokens):
            if not _tokens_are_equivalent_variants(left, right):
                return True
    return False


def check_material_equivalence(
    current_text: str,
    cached_text: str,
    min_content_overlap: float = 0.60,
) -> EquivalenceCheck:
    """Decide whether a cached verdict may be reused for this claim.

    Exact normalised duplicates are safe. Otherwise every critical fact
    must match and content overlap must clear the threshold. Any mismatch
    means the caller must run independent verification.
    """
    if not current_text or not cached_text:
        return EquivalenceCheck(False, "missing_claim_text")

    if _normalise_text(current_text) == _normalise_text(cached_text):
        return EquivalenceCheck(True, "exact_text")

    current = extract_critical_facts(current_text)
    cached = extract_critical_facts(cached_text)

    if current.entities != cached.entities:
        return EquivalenceCheck(False, "entity_mismatch")
    if current.dates != cached.dates:
        return EquivalenceCheck(False, "date_mismatch")
    if current.numbers != cached.numbers:
        return EquivalenceCheck(False, "number_mismatch")
    if current.negated != cached.negated:
        return EquivalenceCheck(False, "negation_mismatch")

    # Same entities in the same order catches role reversals such as
    # "Alice gave Bob the book" vs "Bob gave Alice the book".
    if current.entity_sequence != cached.entity_sequence:
        return EquivalenceCheck(False, "entity_order_mismatch")

    overlap = _jaccard(current.content_tokens, cached.content_tokens)
    if overlap < min_content_overlap:
        return EquivalenceCheck(False, "low_content_overlap")

    # Adjacent content-token order catches reversals without proper nouns,
    # such as "the cat chased the dog" vs "the dog chased the cat".
    order_overlap = _jaccard(
        current.content_bigrams,
        cached.content_bigrams,
    )
    if order_overlap < 0.20:
        return EquivalenceCheck(False, "low_order_overlap")

    # Overlap thresholds are coarse: a long claim keeps enough shared words
    # for a single truth-reversing substitution to clear them. Check the
    # substituted words themselves before allowing cache reuse.
    if _has_unverified_substitution(current, cached):
        return EquivalenceCheck(False, "unverified_substitution")

    return EquivalenceCheck(True, "material_facts_match")


def claims_materially_equivalent(
    current_text: str,
    cached_text: str,
    min_content_overlap: float = 0.60,
) -> bool:
    return check_material_equivalence(
        current_text,
        cached_text,
        min_content_overlap,
    ).equivalent
