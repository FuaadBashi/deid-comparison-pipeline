"""
Tests for BIO repair, span extraction, and the two scoring modes.

Run with:  PYTHONPATH=src pytest tests/ -v
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from deid import labels as L
from deid.labels import repair_bio_sequence, set_label_schema
from deid.metrics import (
    extract_entities_bio,
    entity_exact_breakdown,
    token_typed_breakdown,
)


@pytest.fixture(autouse=True)
def schema():
    set_label_schema(["NAME", "DATE", "ID"])


# --------------------------------------------------------------------------
# BIO repair
# --------------------------------------------------------------------------

def test_repair_promotes_orphan_i_to_b():
    """An I- tag with no matching open entity becomes B-."""
    out, changed = repair_bio_sequence(["O", "I-NAME", "I-NAME"])
    assert out == ["O", "B-NAME", "I-NAME"]
    assert changed == 1


def test_repair_rewrites_pseudo_outside_tags():
    """B-O / I-O are not real entities and collapse to O."""
    out, changed = repair_bio_sequence(["B-O", "I-O", "O"])
    assert out == ["O", "O", "O"]
    assert changed == 2


def test_repair_drops_unknown_entity_types():
    """Types absent from the inferred schema are discarded, not invented."""
    out, changed = repair_bio_sequence(["B-UNSEEN", "O"])
    assert out == ["O", "O"]
    assert changed == 1


def test_repair_is_idempotent():
    once, _ = repair_bio_sequence(["I-NAME", "I-NAME", "B-DATE"])
    twice, changed = repair_bio_sequence(once)
    assert once == twice
    assert changed == 0


# --------------------------------------------------------------------------
# Span extraction
# --------------------------------------------------------------------------

def test_adjacent_b_tags_are_separate_spans():
    """B-DATE B-DATE is two one-token entities, not one two-token entity."""
    spans = extract_entities_bio(["B-DATE", "B-DATE"])
    assert spans == [(0, 1, "DATE"), (1, 2, "DATE")]


def test_multi_token_span_boundaries():
    spans = extract_entities_bio(["O", "B-NAME", "I-NAME", "O"])
    assert spans == [(1, 3, "NAME")]


def test_span_running_to_end_of_sequence_is_closed():
    spans = extract_entities_bio(["O", "B-ID", "I-ID"])
    assert spans == [(1, 3, "ID")]


# --------------------------------------------------------------------------
# Scoring
# --------------------------------------------------------------------------

def test_perfect_prediction_scores_one_in_both_modes():
    gold = [["O", "B-NAME", "I-NAME", "O", "B-DATE"]]
    assert entity_exact_breakdown(gold, gold)["overall"]["f1"] == pytest.approx(1.0)
    assert token_typed_breakdown(gold, gold)["overall"]["f1"] == pytest.approx(1.0)


def test_boundary_error_is_a_total_miss_under_entity_exact():
    """
    The headline behaviour of strict scoring: predicting 'John' when the gold
    span is 'John Smith' earns zero credit, and costs both a FP and a FN.
    """
    gold = [["B-NAME", "I-NAME"]]
    pred = [["B-NAME", "O"]]
    r = entity_exact_breakdown(gold, pred)
    assert r["overall"]["tp"] == 0
    assert r["overall"]["fp"] == 1
    assert r["overall"]["fn"] == 1
    assert r["overall"]["f1"] == pytest.approx(0.0)


def test_entity_exact_is_stricter_than_token_typed():
    """
    Same prediction, two metrics. Partial span credit is exactly the gap
    between them -- which is why the metric definition has to be stated
    alongside any reported F1.
    """
    gold = [["O", "B-NAME", "I-NAME", "O", "B-DATE"]]
    pred = [["O", "B-NAME", "O", "O", "B-DATE"]]

    strict = entity_exact_breakdown(gold, pred)["overall"]["f1"]
    lenient = token_typed_breakdown(gold, pred)["overall"]["f1"]

    assert strict == pytest.approx(0.5)
    assert lenient == pytest.approx(0.8)
    assert strict < lenient


def test_wrong_type_costs_both_precision_and_recall():
    gold = [["B-NAME"]]
    pred = [["B-DATE"]]
    o = token_typed_breakdown(gold, pred)["overall"]
    assert o["tp"] == 0 and o["fp"] == 1 and o["fn"] == 1


def test_empty_prediction_gives_zero_recall_not_a_crash():
    gold = [["B-NAME", "I-NAME"]]
    pred = [["O", "O"]]
    o = entity_exact_breakdown(gold, pred)["overall"]
    assert o["recall"] == pytest.approx(0.0)
    assert o["f1"] == pytest.approx(0.0)
