from functools import lru_cache
from pathlib import Path
from typing import Annotated, Literal

from fastapi import Depends, FastAPI, HTTPException
from pydantic import BaseModel, Field, StringConstraints

from financial_complaint_triage import (
    FinancialComplaintTriageService,
)


MAX_TEXT_LENGTH = 10_000
MAX_BATCH_SIZE = 16
MODEL_PATH = (
    Path(__file__).resolve().parents[1]
    / "models"
    / "transformer"
    / "modernbert_base_512"
)

NarrativeText = Annotated[
    str,
    StringConstraints(
        min_length=1,
        max_length=MAX_TEXT_LENGTH,
    ),
]


class HealthResponse(BaseModel):
    status: Literal["ok"]


class PredictRequest(BaseModel):
    text: NarrativeText


class BatchPredictRequest(BaseModel):
    texts: list[NarrativeText] = Field(
        min_length=1,
        max_length=MAX_BATCH_SIZE,
    )


class CandidateResponse(BaseModel):
    product: str
    confidence_score: float


class PredictionResponse(BaseModel):
    predicted_product: str
    confidence_score: float
    routing_decision: Literal[
        "automatic_route",
        "manual_review",
    ]
    decision_reason: str
    threshold: float
    top_candidates: list[CandidateResponse]


app = FastAPI(
    title="Financial Complaint Triage API",
    version="0.1.0",
    openapi_url=None,
    docs_url=None,
    redoc_url=None,
)


@lru_cache(maxsize=1)
def get_service() -> FinancialComplaintTriageService:
    """Create the shared inference service only when first requested."""

    return FinancialComplaintTriageService(
        model_path=MODEL_PATH,
        confidence_threshold=0.90,
        max_length=512,
    )


def _safe_prediction(
    predict,
) -> PredictionResponse | list[PredictionResponse]:
    """Run one service operation without exposing its input or internals."""

    try:
        result = predict()
    except ValueError:
        raise HTTPException(
            status_code=422,
            detail="The complaint input could not be processed.",
        ) from None
    except Exception:
        raise HTTPException(
            status_code=503,
            detail="The inference service is unavailable.",
        ) from None

    if isinstance(result, list):
        return [
            PredictionResponse.model_validate(
                prediction.to_dict()
            )
            for prediction in result
        ]

    return PredictionResponse.model_validate(
        result.to_dict()
    )


@app.get(
    "/health",
    response_model=HealthResponse,
)
def health() -> HealthResponse:
    """Report process health without loading the model."""

    return HealthResponse(status="ok")


@app.post(
    "/predict",
    response_model=PredictionResponse,
)
def predict(
    request: PredictRequest,
    service: Annotated[
        FinancialComplaintTriageService,
        Depends(get_service),
    ],
) -> PredictionResponse:
    """Classify one complaint narrative."""

    return _safe_prediction(
        lambda: service.predict_one(request.text)
    )


@app.post(
    "/predict/batch",
    response_model=list[PredictionResponse],
)
def predict_batch(
    request: BatchPredictRequest,
    service: Annotated[
        FinancialComplaintTriageService,
        Depends(get_service),
    ],
) -> list[PredictionResponse]:
    """Classify up to 16 complaint narratives in one request."""

    return _safe_prediction(
        lambda: service.predict_batch(
            request.texts,
            batch_size=MAX_BATCH_SIZE,
        )
    )
