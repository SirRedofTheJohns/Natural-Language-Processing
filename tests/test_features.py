"""Synthetic tests for deterministic passenger feature engineering."""

import pandas as pd

from titanic_survival_prediction.features import (
    BASIC_ENGINEERED_COLUMNS,
    add_basic_features,
    cabin_is_known,
    calculate_family_size,
    count_cabins,
    create_is_alone,
    extract_cabin_deck,
    extract_surname,
    extract_title,
    normalize_ticket_prefix,
)


def synthetic_passengers() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Name": [
                "North, Mr. Alex",
                "River, Mlle. Bea",
                "Vale, Major. Chen",
                None,
            ],
            "Ticket": ["A/5. 21171", "113803", "STON/O2. 3101282", None],
            "Cabin": ["C85", None, "F G73 F G75", ""],
            "SibSp": [0, 1, 0, 0],
            "Parch": [0, 1, 2, 0],
            "Sex": ["male", "female", "male", None],
            "Age": [34, 28, 42, None],
            "PassengerId": [1, 2, 3, 4],
            "Pclass": [1, 2, 3, 3],
            "Fare": [20.0, 30.0, 10.0, 8.0],
            "Embarked": ["S", "C", "Q", "S"],
        }
    )


def test_title_normalization_covers_common_alias_rare_and_missing() -> None:
    names = pd.Series(
        ["North, Mr. Alex", "River, Mlle. Bea", "Vale, Major. Chen", None]
    )
    assert extract_title(names).tolist() == ["Mr", "Miss", "Officer", "Unknown"]
    assert extract_title(pd.Series(["Fox, Baron. Dee"])).iloc[0] == "Rare"
    assert extract_title(pd.Series(["Vale, the Countess. Dee"])).iloc[0] == "Nobility"


def test_surname_extraction_and_missing_name() -> None:
    names = pd.Series(["  North Smith, Mr. Alex", None, "malformed"])
    assert extract_surname(names).tolist() == [
        "North Smith",
        "Unknown",
        "Unknown",
    ]


def test_ticket_prefix_normalization() -> None:
    tickets = pd.Series(["A/5. 21171", "113803", "STON/O2. 3101282", None, "LINE"])
    assert normalize_ticket_prefix(tickets).tolist() == [
        "A5",
        "NONE",
        "STONO2",
        "UNKNOWN",
        "LINE",
    ]


def test_cabin_features_cover_missing_and_multiple_cabins() -> None:
    cabins = pd.Series([None, "", "F G73 F G75", "C85"])
    assert extract_cabin_deck(cabins).tolist() == ["U", "U", "F", "C"]
    assert cabin_is_known(cabins).tolist() == [False, False, True, True]
    assert count_cabins(cabins).tolist() == [0, 0, 2, 1]


def test_family_size_and_is_alone() -> None:
    data = pd.DataFrame({"SibSp": [0, 1, None], "Parch": [0, 2, 0]})
    family_size = calculate_family_size(data)
    assert family_size.tolist() == [1, 4, 1]
    assert create_is_alone(family_size).tolist() == [True, False, True]


def test_basic_features_are_stable_and_input_is_not_mutated() -> None:
    data = synthetic_passengers()
    original = data.copy(deep=True)
    result = add_basic_features(data)

    assert tuple(result.columns[-len(BASIC_ENGINEERED_COLUMNS) :]) == (
        BASIC_ENGINEERED_COLUMNS
    )
    assert result.loc[:, list(BASIC_ENGINEERED_COLUMNS)].columns.tolist() == list(
        BASIC_ENGINEERED_COLUMNS
    )
    assert result["IsMother"].tolist() == [False, True, False, False]
    assert result["IsFather"].tolist() == [False, False, True, False]
    assert result["NumberOfCabins"].tolist() == [1, 0, 2, 0]
    assert result["AgeMissing"].tolist() == [False, False, False, True]
    assert result["FareMissing"].tolist() == [False, False, False, False]
    pd.testing.assert_frame_equal(data, original)
