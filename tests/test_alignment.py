import pytest

from deid.labels import set_label_schema
from deid.metrics import entity_exact_breakdown, token_typed_breakdown


@pytest.mark.parametrize("scorer", [entity_exact_breakdown, token_typed_breakdown])
@pytest.mark.parametrize("gold,pred", [
    ([["B-NAME"], ["B-NAME"]], [["B-NAME"]]),
    ([["B-NAME"]], [["B-NAME"], ["O"]]),
    ([["B-NAME", "B-NAME"]], [["B-NAME"]]),
    ([["B-NAME"]], [["B-NAME", "O"]]),
])
def test_misaligned_predictions_are_rejected(scorer, gold, pred):
    set_label_schema(["NAME"])
    with pytest.raises(ValueError, match="count"):
        scorer(gold, pred)
