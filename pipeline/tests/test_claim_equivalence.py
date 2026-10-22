"""Material-fact guard for semantic cache reuse."""

import pytest

from src.pipeline.claim_equivalence import (
    check_material_equivalence,
    claims_materially_equivalent,
    extract_critical_facts,
)


def test_exact_normalized_text_is_equivalent():
    assert claims_materially_equivalent(
        "The Eiffel Tower is in Paris.",
        "  the eiffel tower is in paris  ",
    )


def test_contradictory_locations_are_not_equivalent():
    check = check_material_equivalence(
        "The Louvre Museum is located in Berlin, Germany.",
        "The Louvre Museum is located in Paris, France.",
    )
    assert check.equivalent is False
    assert check.reason == "entity_mismatch"


def test_changed_dates_are_not_equivalent():
    check = check_material_equivalence(
        "The Eiffel Tower was completed in 1889.",
        "The Eiffel Tower was completed in 1912.",
    )
    assert check.equivalent is False
    assert check.reason == "date_mismatch"


def test_changed_quantities_are_not_equivalent():
    check = check_material_equivalence(
        "The budget is 42 million euros.",
        "The budget is 300 million euros.",
    )
    assert check.equivalent is False
    assert check.reason == "number_mismatch"


def test_changed_currency_is_not_equivalent():
    check = check_material_equivalence(
        "The prize is 42 million euros.",
        "The prize is 42 million dollars.",
    )
    assert check.equivalent is False
    assert check.reason == "number_mismatch"


def test_negation_is_not_equivalent():
    check = check_material_equivalence(
        "The Louvre is in Paris.",
        "The Louvre is not in Paris.",
    )
    assert check.equivalent is False
    assert check.reason == "negation_mismatch"


def test_changed_person_or_organization_is_not_equivalent():
    for current, cached in (
        (
            "Tim Cook announced a new iPhone.",
            "Satya Nadella announced a new iPhone.",
        ),
        (
            "NASA confirmed the discovery.",
            "ESA confirmed the discovery.",
        ),
    ):
        check = check_material_equivalence(current, cached)
        assert check.equivalent is False
        assert check.reason == "entity_mismatch"


def test_changed_predicate_without_entities_is_not_equivalent():
    check = check_material_equivalence(
        "The policy reduced unemployment.",
        "The policy increased unemployment.",
    )
    assert check.equivalent is False
    assert check.reason == "low_content_overlap"


def test_same_facts_with_paraphrase_is_equivalent():
    assert claims_materially_equivalent(
        "The Eiffel Tower is in Paris.",
        "The Eiffel Tower is located in Paris.",
    )
    assert claims_materially_equivalent(
        "The Great Wall of China was built by the Romans.",
        "The Great Wall of China was constructed by the Romans.",
    )


def test_additional_location_detail_is_conservatively_rejected():
    check = check_material_equivalence(
        "The Louvre is in Paris.",
        "The Louvre is in Paris, France.",
    )
    assert check.equivalent is False
    assert check.reason == "entity_mismatch"


def test_entity_role_reversal_is_not_equivalent():
    check = check_material_equivalence(
        "Alice gave Bob the book.",
        "Bob gave Alice the book.",
    )
    assert check.equivalent is False
    assert check.reason == "entity_order_mismatch"


def test_predicate_order_reversal_without_entities_is_not_equivalent():
    check = check_material_equivalence(
        "The cat chased the dog.",
        "The dog chased the cat.",
    )
    assert check.equivalent is False
    assert check.reason == "low_order_overlap"


def test_missing_text_is_not_equivalent():
    assert not claims_materially_equivalent("", "some claim")
    assert not claims_materially_equivalent("some claim", "")


def test_critical_fact_extraction():
    facts = extract_critical_facts(
        "The Eiffel Tower was completed in 1889 for 42 million euros."
    )
    assert "eiffel tower" in facts.entities
    assert "1889" in facts.dates
    assert any("42" in number for number in facts.numbers)
    assert facts.negated is False

    negated = extract_critical_facts("The Louvre is not in Berlin.")
    assert negated.negated is True


@pytest.mark.parametrize(
    "current_predicate,cached_predicate",
    [
        ("increased", "decreased"),
        ("approved", "rejected"),
        ("was safe", "was unsafe"),
        ("won support", "lost support"),
        ("was backed by more than half", "was backed by fewer than half"),
        ("appeared before", "appeared after"),
    ],
)
def test_truth_reversing_content_word_substitution_is_not_equivalent(
    current_predicate,
    cached_predicate,
):
    context = (
        "according to the official report released after the long public "
        "consultation last year"
    )
    current = f"The measure {current_predicate} {context}."
    cached = f"The measure {cached_predicate} {context}."

    check = check_material_equivalence(current, cached)

    assert check.equivalent is False
    assert check.reason == "unverified_substitution"


def test_inflectional_substitution_is_equivalent():
    assert claims_materially_equivalent(
        "The council approved the housing plan.",
        "The council approves the housing plan.",
    )


def test_uncased_script_tokenizer_preserves_combining_marks():
    facts = extract_critical_facts("दिल्ली में बेरोजगारी बढ़ी है।")

    assert "दिल्ली" in facts.content_tokens
    assert "बेरोजगारी" in facts.content_tokens


def test_uncased_script_negation_is_detected():
    facts = extract_critical_facts(
        "सरकार ने नई शिक्षा नीति की घोषणा नहीं की है।"
    )

    assert facts.negated is True


@pytest.mark.parametrize(
    "current_place,cached_place",
    [
        ("नई दिल्ली", "मुंबई"),
        ("जयपुर", "कोलकाता"),
    ],
)
def test_uncased_script_place_substitution_is_not_equivalent(
    current_place,
    cached_place,
):
    template = "सरकार ने {place} में नई शिक्षा नीति की घोषणा की है।"
    current = template.format(place=current_place)
    cached = template.format(place=cached_place)

    check = check_material_equivalence(current, cached)

    assert check.equivalent is False
    assert check.reason == "unverified_substitution"


def test_french_direction_substitution_is_not_equivalent():
    current = (
        "Le chômage a fortement augmenté dans le pays après la nouvelle réforme."
    )
    cached = (
        "Le chômage a fortement diminué dans le pays après la nouvelle réforme."
    )

    check = check_material_equivalence(current, cached)

    assert check.equivalent is False
    assert check.reason == "unverified_substitution"
