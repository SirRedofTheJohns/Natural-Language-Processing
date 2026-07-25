"""Synthetic schema tests for official-data handling."""

import pandas as pd
import pytest

from titanic_survival_prediction.data import validate_competition_frames


def synthetic_frames() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train = pd.DataFrame(
        {
            "PassengerId": [1, 2, 3, 4],
            "Survived": [0, 1, 0, 1],
            "Pclass": [3, 1, 2, 3],
            "Name": ["A, Mr. One", "B, Mrs. Two", "C, Miss. Three", "D, Mr. Four"],
            "Sex": ["male", "female", "female", "male"],
            "Age": [30.0, 40.0, None, 20.0],
            "SibSp": [0, 1, 0, 0],
            "Parch": [0, 1, 0, 0],
            "Ticket": ["A 1", "B 2", "C 3", "D 4"],
            "Fare": [8.0, 80.0, 20.0, 9.0],
            "Cabin": [None, "C1", None, None],
            "Embarked": ["S", "C", "Q", "S"],
        }
    )
    test = train.drop(columns="Survived").copy()
    test["PassengerId"] = [11, 12, 13, 14]
    sample = pd.DataFrame(
        {"PassengerId": test["PassengerId"], "Survived": [0, 1, 0, 1]}
    )
    return train, test, sample


def test_valid_competition_frames() -> None:
    validate_competition_frames(*synthetic_frames())


def test_test_target_is_rejected() -> None:
    train, test, sample = synthetic_frames()
    test.insert(1, "Survived", [0, 0, 0, 0])
    with pytest.raises(ValueError, match="test.csv columns"):
        validate_competition_frames(train, test, sample)


def test_duplicate_passenger_id_is_rejected() -> None:
    train, test, sample = synthetic_frames()
    test.loc[1, "PassengerId"] = test.loc[0, "PassengerId"]
    with pytest.raises(ValueError, match="unique"):
        validate_competition_frames(train, test, sample)
