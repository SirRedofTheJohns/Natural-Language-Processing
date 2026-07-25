"""Synthetic tests for Kaggle submission validation."""

import pandas as pd
import pytest

from titanic_survival_prediction.submission import (
    SUBMISSION_COLUMNS,
    build_submission,
    save_submission,
    validate_submission_file,
)


def test_valid_submission_has_schema_and_preserves_passenger_order() -> None:
    submission = build_submission([905, 901, 903], [1, 0, 1])
    assert tuple(submission.columns) == SUBMISSION_COLUMNS
    assert submission["PassengerId"].tolist() == [905, 901, 903]
    assert submission["Survived"].tolist() == [1, 0, 1]
    assert submission.dtypes.tolist() == ["int64", "int64"]


@pytest.mark.parametrize("predictions", [[0, 2], [0, -1], [0, 0.5], [0, None]])
def test_invalid_prediction_values_raise(predictions: list[object]) -> None:
    with pytest.raises(ValueError, match="Predictions"):
        build_submission([101, 102], predictions)


def test_prediction_length_mismatch_raises() -> None:
    with pytest.raises(ValueError, match="length"):
        build_submission([101, 102, 103], [0, 1])


def test_missing_passenger_id_raises() -> None:
    with pytest.raises(ValueError, match="must not be missing"):
        build_submission(pd.Series([101, None]), [0, 1])


@pytest.mark.parametrize(
    "passenger_ids, message",
    [
        ([101, 101], "unique"),
        ([0, 102], "positive"),
        ([101.5, 102], "whole numbers"),
        (["not-an-id", 102], "numeric whole numbers"),
    ],
)
def test_duplicate_or_invalid_passenger_ids_raise(
    passenger_ids: list[object], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        build_submission(passenger_ids, [0, 1])


def test_explicit_save_and_file_validation(tmp_path) -> None:
    submission = build_submission([201, 202, 203], [0, 1, 0])
    destination = save_submission(
        submission,
        tmp_path / "candidate.csv",
        expected_passenger_ids=[201, 202, 203],
    )
    validated = validate_submission_file(destination, [201, 202, 203])
    pd.testing.assert_frame_equal(validated, submission)


def test_save_rejects_wrong_expected_order(tmp_path) -> None:
    submission = build_submission([201, 202], [0, 1])
    with pytest.raises(ValueError, match="order"):
        save_submission(
            submission,
            tmp_path / "candidate.csv",
            expected_passenger_ids=[202, 201],
        )
