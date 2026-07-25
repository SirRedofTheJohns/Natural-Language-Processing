"""Validation and construction helpers for Kaggle Titanic submissions."""

from __future__ import annotations

from collections.abc import Sequence
from os import PathLike
from pathlib import Path

import numpy as np
import pandas as pd

SUBMISSION_COLUMNS: tuple[str, str] = ("PassengerId", "Survived")


def validate_passenger_ids(
    passenger_ids: pd.Series | Sequence[int],
) -> pd.Series:
    """Validate positive, unique, whole-number passenger identifiers."""

    ids = pd.Series(passenger_ids, copy=True, name="PassengerId").reset_index(drop=True)
    if ids.empty:
        raise ValueError("PassengerId values must not be empty.")
    if ids.isna().any():
        raise ValueError("PassengerId values must not be missing.")
    try:
        numeric = pd.to_numeric(ids, errors="raise")
    except (TypeError, ValueError) as exc:
        raise ValueError("PassengerId values must be numeric whole numbers.") from exc
    values = numeric.to_numpy(dtype=float)
    if not np.isfinite(values).all() or not np.equal(values, np.floor(values)).all():
        raise ValueError("PassengerId values must be finite whole numbers.")
    validated = numeric.astype("int64")
    if (validated <= 0).any():
        raise ValueError("PassengerId values must be positive.")
    if validated.duplicated().any():
        raise ValueError("PassengerId values must be unique.")
    return validated.rename("PassengerId")


def validate_predictions(
    predictions: pd.Series | Sequence[int],
    expected_length: int | None = None,
) -> pd.Series:
    """Validate a one-dimensional sequence containing only integer 0 and 1."""

    values = pd.Series(predictions, copy=True, name="Survived").reset_index(drop=True)
    if expected_length is not None and len(values) != expected_length:
        raise ValueError(
            f"Prediction length {len(values)} does not match "
            f"PassengerId length {expected_length}."
        )
    if values.isna().any():
        raise ValueError("Predictions must not be missing.")
    try:
        numeric = pd.to_numeric(values, errors="raise")
    except (TypeError, ValueError) as exc:
        raise ValueError("Predictions must contain only integer 0 and 1.") from exc
    array = numeric.to_numpy(dtype=float)
    if (
        not np.isfinite(array).all()
        or not np.equal(array, np.floor(array)).all()
        or not numeric.isin([0, 1]).all()
    ):
        raise ValueError("Predictions must contain only integer 0 and 1.")
    return numeric.astype("int64").rename("Survived")


def build_submission(
    passenger_ids: pd.Series | Sequence[int],
    predictions: pd.Series | Sequence[int],
) -> pd.DataFrame:
    """Construct an ordered ``PassengerId, Survived`` submission DataFrame."""

    ids = validate_passenger_ids(passenger_ids)
    labels = validate_predictions(predictions, expected_length=len(ids))
    return pd.DataFrame({"PassengerId": ids, "Survived": labels}).loc[
        :, list(SUBMISSION_COLUMNS)
    ]


def save_submission(
    submission: pd.DataFrame,
    path: str | PathLike[str],
    expected_passenger_ids: pd.Series | Sequence[int] | None = None,
) -> Path:
    """Validate and save a submission when the caller explicitly supplies a path."""

    if list(submission.columns) != list(SUBMISSION_COLUMNS):
        raise ValueError(
            "Submission columns must be exactly PassengerId and Survived in that order."
        )
    validated = build_submission(submission["PassengerId"], submission["Survived"])
    if expected_passenger_ids is not None:
        expected = validate_passenger_ids(expected_passenger_ids)
        if len(validated) != len(expected):
            raise ValueError(
                f"Submission row count {len(validated)} does not match "
                f"expected row count {len(expected)}."
            )
        if not validated["PassengerId"].equals(expected):
            raise ValueError("Submission PassengerId order does not match test data.")
    destination = Path(path)
    if destination.suffix.lower() != ".csv":
        raise ValueError("Submission path must use a .csv extension.")
    if not destination.parent.exists():
        raise FileNotFoundError(
            f"Submission directory does not exist: {destination.parent}"
        )
    validated.to_csv(destination, index=False)
    return destination


def validate_submission_file(
    path: str | PathLike[str],
    expected_passenger_ids: pd.Series | Sequence[int],
) -> pd.DataFrame:
    """Load and validate a saved submission against official test identifiers."""

    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"Submission file does not exist: {source}")
    submission = pd.read_csv(source)
    if list(submission.columns) != list(SUBMISSION_COLUMNS):
        raise ValueError(
            "Submission columns must be exactly PassengerId and Survived in that order."
        )
    expected = validate_passenger_ids(expected_passenger_ids)
    validated = build_submission(submission["PassengerId"], submission["Survived"])
    if len(validated) != len(expected):
        raise ValueError(
            f"Submission row count {len(validated)} does not match "
            f"expected row count {len(expected)}."
        )
    if not validated["PassengerId"].equals(expected):
        raise ValueError("Submission PassengerId order does not match test data.")
    return validated
