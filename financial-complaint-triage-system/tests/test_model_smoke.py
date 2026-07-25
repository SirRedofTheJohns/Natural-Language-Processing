from pathlib import Path

import pytest

from financial_complaint_triage import (
    FinancialComplaintTriageService,
)


MODEL_PATH = (
    Path(__file__).resolve().parents[1]
    / "models"
    / "transformer"
    / "modernbert_base_512"
)


@pytest.mark.slow
def test_local_modernbert_prediction_contract():
    service = FinancialComplaintTriageService(
        model_path=MODEL_PATH,
        confidence_threshold=0.90,
        device="cpu",
    )
    result = service.predict_one(
        "A synthetic cardholder reports an unfamiliar recurring "
        "charge and asks the issuer to investigate the account."
    )

    assert len(service.id2label) == 11
    assert set(result.to_dict()) == {
        "predicted_product",
        "confidence_score",
        "routing_decision",
        "decision_reason",
        "threshold",
        "top_candidates",
    }
