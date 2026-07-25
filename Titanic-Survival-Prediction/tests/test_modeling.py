"""Synthetic leakage and pipeline tests."""

import numpy as np
import pandas as pd
from sklearn.base import clone

from titanic_survival_prediction.features import TitanicFeatureBuilder
from titanic_survival_prediction.modeling import build_estimator


def synthetic_rows() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "PassengerId": np.arange(1, 13),
            "Pclass": [1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3],
            "Name": [
                f"Family{i // 2}, {'Mrs' if i % 2 else 'Mr'}. Person{i}"
                for i in range(12)
            ],
            "Sex": ["male", "female"] * 6,
            "Age": [20, 30, None, 25, 40, 12, 55, 29, 18, None, 33, 42],
            "SibSp": [1] * 12,
            "Parch": [0, 0, 1, 1, 0, 0, 2, 2, 0, 0, 1, 1],
            "Ticket": ["A", "A", "B", "B", "C", "C", "D", "D", "E", "E", "F", "F"],
            "Fare": [10, 10, 20, 20, 30, 30, 40, 40, 15, 15, 25, 25],
            "Cabin": [
                None,
                "C1",
                None,
                "D1",
                None,
                None,
                "B1",
                "B2",
                None,
                None,
                "E1",
                "E2",
            ],
            "Embarked": ["S", "C", "Q"] * 4,
        }
    )


def test_group_counts_use_fit_rows_and_unseen_groups_fall_back_to_one() -> None:
    rows = synthetic_rows()
    builder = TitanicFeatureBuilder(feature_set="group_counts").fit(rows.iloc[:4])
    transformed = builder.transform(rows.iloc[[0, 4]])
    assert transformed["TicketGroupSize"].tolist() == [2.0, 1.0]
    assert transformed["FamilyGroupSize"].tolist() == [2.0, 1.0]


def test_group_count_fit_is_target_independent_and_deterministic() -> None:
    rows = synthetic_rows()
    first = TitanicFeatureBuilder(feature_set="group_counts").fit(
        rows, np.zeros(len(rows))
    )
    second = TitanicFeatureBuilder(feature_set="group_counts").fit(
        rows, np.ones(len(rows))
    )
    pd.testing.assert_frame_equal(first.transform(rows), second.transform(rows))


def test_logistic_pipeline_handles_unseen_categories_and_is_deterministic() -> None:
    rows = synthetic_rows()
    target = pd.Series([0, 1] * 6)
    first = build_estimator("logistic", "ticket_cabin", seed=123)
    second = build_estimator("logistic", "ticket_cabin", seed=123)
    first.fit(rows.iloc[:10], target.iloc[:10])
    second.fit(rows.iloc[:10], target.iloc[:10])
    unseen = rows.iloc[10:].copy()
    unseen["Embarked"] = "NEW"
    first_probability = first.predict_proba(unseen)
    second_probability = second.predict_proba(unseen)
    assert first_probability.shape == (2, 2)
    np.testing.assert_allclose(first_probability, second_probability)
    assert set(first.predict(unseen)).issubset({0, 1})


def test_catboost_pipeline_is_sklearn_clone_safe() -> None:
    estimator = build_estimator("catboost", "ticket_cabin", seed=123)
    cloned = clone(estimator)
    assert cloned.get_params()["model__random_seed"] == 123
