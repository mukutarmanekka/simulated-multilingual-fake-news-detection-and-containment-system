"""The NLI model's label order must be read from its config, not assumed."""

from src.models.enums import Stance
from src.pipeline.claim_verifier import build_stance_label_map


def test_deberta_label_order_is_mapped_correctly():
    mapping = build_stance_label_map(
        {0: "contradiction", 1: "entailment", 2: "neutral"}
    )

    assert mapping[0] == Stance.REFUTES
    assert mapping[1] == Stance.SUPPORTS
    assert mapping[2] == Stance.NEUTRAL


def test_alternate_label_order_is_mapped_by_name():
    mapping = build_stance_label_map(
        {0: "ENTAILMENT", 1: "NEUTRAL", 2: "CONTRADICTION"}
    )

    assert mapping[0] == Stance.SUPPORTS
    assert mapping[1] == Stance.NEUTRAL
    assert mapping[2] == Stance.REFUTES


def test_generic_labels_use_standard_nli_fallback():
    mapping = build_stance_label_map(
        {0: "LABEL_0", 1: "LABEL_1", 2: "LABEL_2"}
    )

    assert mapping[0] == Stance.REFUTES
    assert mapping[1] == Stance.SUPPORTS
    assert mapping[2] == Stance.NEUTRAL


def test_missing_config_uses_standard_nli_fallback():
    mapping = build_stance_label_map(None)

    assert mapping[0] == Stance.REFUTES
    assert mapping[1] == Stance.SUPPORTS
    assert mapping[2] == Stance.NEUTRAL
