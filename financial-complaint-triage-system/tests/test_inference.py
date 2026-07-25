import math
from types import SimpleNamespace

import pytest
import torch

from financial_complaint_triage.inference import (
    FinancialComplaintTriageService,
)


LABELS = {
    index: f"Synthetic product {index}"
    for index in range(11)
}


class FakeTokenizer:
    def __init__(self, markers_by_text):
        self.markers_by_text = markers_by_text
        self.calls = []

    def __call__(self, texts, **kwargs):
        batch = list(texts)
        self.calls.append(batch)

        return {
            "input_ids": torch.tensor(
                [
                    [self.markers_by_text[text]]
                    for text in batch
                ],
                dtype=torch.long,
            )
        }


class FakeModel:
    def __init__(self, logits_by_marker):
        self.logits_by_marker = {
            marker: torch.tensor(logits, dtype=torch.float32)
            for marker, logits in logits_by_marker.items()
        }
        self.config = SimpleNamespace(id2label=LABELS)
        self.calls = []
        self.device = None
        self.eval_called = False

    def to(self, device):
        self.device = device
        return self

    def eval(self):
        self.eval_called = True
        return self

    def __call__(self, input_ids):
        markers = input_ids[:, 0].tolist()
        self.calls.append(markers)

        return SimpleNamespace(
            logits=torch.stack(
                [
                    self.logits_by_marker[marker]
                    for marker in markers
                ]
            )
        )


def make_logits(
    first_index,
    first_logit=8.0,
    second_index=1,
    second_logit=0.0,
):
    logits = [-100.0] * 11
    logits[first_index] = first_logit
    logits[second_index] = second_logit
    return logits


def make_service(
    texts_to_logits,
    confidence_threshold=0.90,
):
    markers_by_text = {
        text: marker
        for marker, text in enumerate(texts_to_logits)
    }
    logits_by_marker = {
        markers_by_text[text]: logits
        for text, logits in texts_to_logits.items()
    }
    tokenizer = FakeTokenizer(markers_by_text)
    model = FakeModel(logits_by_marker)
    service = FinancialComplaintTriageService(
        confidence_threshold=confidence_threshold,
        device="cpu",
        tokenizer=tokenizer,
        model=model,
    )

    return service, tokenizer, model


def test_clean_text_removes_html_and_normalizes_whitespace():
    text = "  Card&nbsp;charge <b>was</b>\n duplicated.  "

    assert FinancialComplaintTriageService.clean_text(text) == (
        "Card charge was duplicated."
    )


def test_empty_input_is_rejected_before_inference():
    valid_text = "My synthetic card charge appeared twice this month."
    service, tokenizer, model = make_service(
        {valid_text: make_logits(0)}
    )

    with pytest.raises(ValueError, match="cannot be empty"):
        service.predict_one(" <p> </p> ")

    with pytest.raises(ValueError, match="position 1"):
        service.predict_batch([valid_text, None])

    assert tokenizer.calls == []
    assert model.calls == []


def test_insufficient_text_requires_manual_review():
    text = "Card charge is wrong."
    service, _, _ = make_service(
        {text: make_logits(0)}
    )

    result = service.predict_one(text)

    assert result.routing_decision == "manual_review"
    assert result.decision_reason == "insufficient_text"


@pytest.mark.parametrize(
    "threshold",
    [-0.01, 1.01, math.nan, math.inf],
)
def test_invalid_confidence_threshold_is_rejected(threshold):
    with pytest.raises(ValueError, match="between 0 and 1"):
        FinancialComplaintTriageService(
            confidence_threshold=threshold,
            device="cpu",
            tokenizer=FakeTokenizer({}),
            model=FakeModel({}),
        )


def test_confidence_above_threshold_routes_automatically():
    text = "My synthetic bank transfer was charged twice unexpectedly."
    service, _, _ = make_service(
        {text: make_logits(2)},
        confidence_threshold=0.90,
    )

    result = service.predict_one(text)

    assert result.predicted_product == LABELS[2]
    assert result.routing_decision == "automatic_route"
    assert result.decision_reason == "confidence_threshold_met"
    assert set(result.to_dict()) == {
        "predicted_product",
        "confidence_score",
        "routing_decision",
        "decision_reason",
        "threshold",
        "top_candidates",
    }


def test_unrounded_confidence_below_threshold_requires_review():
    text = "My synthetic loan payment was recorded incorrectly yesterday."
    raw_confidence = 0.8999996
    first_logit = math.log(
        raw_confidence / (1.0 - raw_confidence)
    )
    service, _, _ = make_service(
        {
            text: make_logits(
                first_index=3,
                first_logit=first_logit,
                second_index=4,
                second_logit=0.0,
            )
        },
        confidence_threshold=0.90,
    )

    result = service.predict_one(text)

    assert result.confidence_score == 0.9
    assert result.routing_decision == "manual_review"
    assert result.decision_reason == "low_confidence"


def test_top_candidates_are_returned_in_descending_order():
    text = "My synthetic credit report lists an account I never opened."
    logits = [-100.0] * 11
    logits[7] = 4.0
    logits[2] = 3.0
    logits[5] = 2.0
    service, _, _ = make_service({text: logits})

    result = service.predict_one(text, top_k=3)

    assert [
        candidate.product
        for candidate in result.top_candidates
    ] == [LABELS[7], LABELS[2], LABELS[5]]
    assert [
        candidate.confidence_score
        for candidate in result.top_candidates
    ] == sorted(
        [
            candidate.confidence_score
            for candidate in result.top_candidates
        ],
        reverse=True,
    )


def test_predict_batch_preserves_order_and_uses_internal_batches():
    texts = [
        "My synthetic checking account transfer was not completed.",
        "My synthetic mortgage payment was applied to the wrong month.",
        "My synthetic credit card was charged after cancellation.",
    ]
    service, tokenizer, model = make_service(
        {
            texts[0]: make_logits(6),
            texts[1]: make_logits(1),
            texts[2]: make_logits(9),
        }
    )

    def fail_if_called(*args, **kwargs):
        raise AssertionError(
            "predict_batch must not call predict_one"
        )

    service.predict_one = fail_if_called
    results = service.predict_batch(
        texts,
        batch_size=2,
    )

    assert len(results) == len(texts)
    assert [
        result.predicted_product
        for result in results
    ] == [LABELS[6], LABELS[1], LABELS[9]]
    assert tokenizer.calls == [texts[:2], texts[2:]]
    assert model.calls == [[0, 1], [2]]
