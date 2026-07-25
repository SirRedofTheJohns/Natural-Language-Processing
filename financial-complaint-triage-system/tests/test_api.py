from fastapi.testclient import TestClient

from api.main import app, get_service
from financial_complaint_triage import (
    PredictionCandidate,
    TriagePrediction,
)


SYNTHETIC_TEXT = (
    "A synthetic consumer reports a duplicated card transaction."
)


class FakeService:
    def __init__(self):
        self.single_calls = []
        self.batch_calls = []
        self.error = None

    @staticmethod
    def _prediction(product):
        return TriagePrediction(
            predicted_product=product,
            confidence_score=0.95,
            routing_decision="automatic_route",
            decision_reason="confidence_threshold_met",
            threshold=0.90,
            top_candidates=[
                PredictionCandidate(
                    product=product,
                    confidence_score=0.95,
                ),
                PredictionCandidate(
                    product="Synthetic alternative",
                    confidence_score=0.03,
                ),
            ],
        )

    def predict_one(self, text):
        if self.error is not None:
            raise self.error

        self.single_calls.append(text)
        return self._prediction("Synthetic card product")

    def predict_batch(self, texts, batch_size):
        if self.error is not None:
            raise self.error

        self.batch_calls.append(
            (list(texts), batch_size)
        )
        return [
            self._prediction(
                f"Synthetic product {index}"
            )
            for index, _ in enumerate(texts)
        ]


def use_fake_service(fake_service):
    app.dependency_overrides[get_service] = (
        lambda: fake_service
    )
    return TestClient(app)


def clear_overrides():
    app.dependency_overrides.clear()


def test_health_does_not_require_the_inference_service():
    def fail_if_called():
        raise AssertionError(
            "Health must not request the model service."
        )

    app.dependency_overrides[get_service] = fail_if_called

    try:
        response = TestClient(app).get("/health")
    finally:
        clear_overrides()

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}
    assert {
        route.path
        for route in app.routes
    } == {
        "/health",
        "/predict",
        "/predict/batch",
    }


def test_valid_single_prediction():
    fake_service = FakeService()
    client = use_fake_service(fake_service)

    try:
        response = client.post(
            "/predict",
            json={"text": SYNTHETIC_TEXT},
        )
    finally:
        clear_overrides()

    assert response.status_code == 200
    assert response.json()["predicted_product"] == (
        "Synthetic card product"
    )
    assert fake_service.single_calls == [SYNTHETIC_TEXT]


def test_valid_batch_prediction():
    texts = [
        SYNTHETIC_TEXT,
        "A synthetic borrower reports a misplaced loan payment.",
    ]
    fake_service = FakeService()
    client = use_fake_service(fake_service)

    try:
        response = client.post(
            "/predict/batch",
            json={"texts": texts},
        )
    finally:
        clear_overrides()

    assert response.status_code == 200
    assert len(response.json()) == 2
    assert fake_service.batch_calls == [(texts, 16)]


def test_empty_input_is_rejected():
    fake_service = FakeService()
    client = use_fake_service(fake_service)

    try:
        response = client.post(
            "/predict",
            json={"text": ""},
        )
    finally:
        clear_overrides()

    assert response.status_code == 422
    assert fake_service.single_calls == []


def test_text_length_limit_is_enforced():
    fake_service = FakeService()
    client = use_fake_service(fake_service)

    try:
        response = client.post(
            "/predict",
            json={"text": "x" * 10_001},
        )
    finally:
        clear_overrides()

    assert response.status_code == 422
    assert fake_service.single_calls == []


def test_batch_size_limit_is_enforced():
    fake_service = FakeService()
    client = use_fake_service(fake_service)

    try:
        response = client.post(
            "/predict/batch",
            json={"texts": [SYNTHETIC_TEXT] * 17},
        )
    finally:
        clear_overrides()

    assert response.status_code == 422
    assert fake_service.batch_calls == []


def test_service_errors_are_safe():
    fake_service = FakeService()
    fake_service.error = RuntimeError(
        "private narrative at /private/local/model"
    )
    client = use_fake_service(fake_service)

    try:
        response = client.post(
            "/predict",
            json={"text": SYNTHETIC_TEXT},
        )
    finally:
        clear_overrides()

    assert response.status_code == 503
    assert response.json() == {
        "detail": "The inference service is unavailable."
    }
    assert "private narrative" not in response.text
    assert "/private/local/model" not in response.text
