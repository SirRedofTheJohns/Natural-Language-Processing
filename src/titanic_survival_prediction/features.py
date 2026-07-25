"""Deterministic and fold-fitted feature engineering for Titanic records."""

from __future__ import annotations

import re
from collections.abc import Iterable

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

UNKNOWN = "Unknown"

BASIC_ENGINEERED_COLUMNS: tuple[str, ...] = (
    "Title",
    "Surname",
    "TicketPrefix",
    "CabinDeck",
    "CabinKnown",
    "NumberOfCabins",
    "FamilySize",
    "IsAlone",
    "IsChild",
    "IsMother",
    "IsFather",
    "AgeMissing",
    "FareMissing",
    "LogFare",
    "FamilySizeCategory",
    "SexClass",
)

FEATURE_SETS: dict[str, tuple[str, ...]] = {
    "raw": (
        "Pclass",
        "Sex",
        "Age",
        "SibSp",
        "Parch",
        "Fare",
        "Embarked",
    ),
    "basic": (
        "Pclass",
        "Sex",
        "Age",
        "SibSp",
        "Parch",
        "Fare",
        "Embarked",
        "Title",
        "FamilySize",
        "IsAlone",
        "IsChild",
        "IsMother",
        "IsFather",
        "AgeMissing",
        "FareMissing",
        "LogFare",
        "FamilySizeCategory",
        "SexClass",
    ),
    "ticket_cabin": (
        "Pclass",
        "Sex",
        "Age",
        "SibSp",
        "Parch",
        "Fare",
        "Embarked",
        "Title",
        "FamilySize",
        "IsAlone",
        "IsChild",
        "IsMother",
        "IsFather",
        "AgeMissing",
        "FareMissing",
        "LogFare",
        "FamilySizeCategory",
        "SexClass",
        "TicketPrefix",
        "CabinDeck",
        "CabinKnown",
        "NumberOfCabins",
    ),
    "group_counts": (
        "Pclass",
        "Sex",
        "Age",
        "SibSp",
        "Parch",
        "Fare",
        "Embarked",
        "Title",
        "FamilySize",
        "IsAlone",
        "IsChild",
        "IsMother",
        "IsFather",
        "AgeMissing",
        "FareMissing",
        "LogFare",
        "FamilySizeCategory",
        "SexClass",
        "TicketPrefix",
        "CabinDeck",
        "CabinKnown",
        "NumberOfCabins",
        "TicketGroupSize",
        "FamilyGroupSize",
        "FarePerTicketMember",
        "FarePerFamilyMember",
        "TicketGroupCategory",
        "FamilyGroupCategory",
    ),
}

CATEGORICAL_FEATURES: frozenset[str] = frozenset(
    {
        "Pclass",
        "Sex",
        "Embarked",
        "Title",
        "FamilySizeCategory",
        "SexClass",
        "TicketPrefix",
        "CabinDeck",
        "TicketGroupCategory",
        "FamilyGroupCategory",
    }
)

_TITLE_ALIASES = {
    "mr": "Mr",
    "mrs": "Mrs",
    "mme": "Mrs",
    "miss": "Miss",
    "mlle": "Miss",
    "ms": "Miss",
    "master": "Master",
    "dr": "Professional",
    "rev": "Clergy",
    "capt": "Officer",
    "col": "Officer",
    "major": "Officer",
    "lady": "Nobility",
    "countess": "Nobility",
    "the countess": "Nobility",
    "sir": "Nobility",
    "jonkheer": "Nobility",
    "don": "Nobility",
    "dona": "Nobility",
}

_BASE_REQUIRED_COLUMNS = {
    "PassengerId",
    "Pclass",
    "Name",
    "Sex",
    "Age",
    "SibSp",
    "Parch",
    "Ticket",
    "Fare",
    "Cabin",
    "Embarked",
}


def _string_series(values: pd.Series, argument: str) -> pd.Series:
    if not isinstance(values, pd.Series):
        raise TypeError(f"{argument} must be a pandas Series.")
    return values.astype("string")


def _required_columns(data: pd.DataFrame, columns: Iterable[str]) -> None:
    if not isinstance(data, pd.DataFrame):
        raise TypeError("data must be a pandas DataFrame.")
    missing = sorted(set(columns).difference(data.columns))
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(missing)}")


def extract_title(names: pd.Series) -> pd.Series:
    """Extract a normalized title from a Titanic-style passenger name."""

    text = _string_series(names, "names")
    raw = text.str.extract(r",\s*([^.]*)\.", expand=False).str.strip().str.lower()
    normalized = raw.map(_TITLE_ALIASES)
    normalized = normalized.mask(raw.notna() & normalized.isna(), "Rare")
    return normalized.fillna(UNKNOWN).astype("string").rename("Title")


def extract_surname(names: pd.Series) -> pd.Series:
    """Extract and whitespace-normalize the text before the first comma."""

    text = _string_series(names, "names")
    surname = text.str.extract(r"^\s*([^,]+),", expand=False)
    surname = surname.str.replace(r"\s+", " ", regex=True).str.strip()
    return surname.fillna(UNKNOWN).astype("string").rename("Surname")


def normalize_ticket_prefix(tickets: pd.Series) -> pd.Series:
    """Normalize a ticket prefix, distinguishing numeric and missing tickets."""

    text = _string_series(tickets, "tickets").str.strip().str.upper()

    def normalize(value: object) -> str:
        if pd.isna(value) or not str(value).strip():
            return "UNKNOWN"
        prefix = re.sub(r"\s*\d+\s*$", "", str(value).strip())
        prefix = re.sub(r"[^A-Z0-9]+", "", prefix)
        return prefix or "NONE"

    return text.map(normalize, na_action=None).astype("string").rename("TicketPrefix")


def normalize_ticket_id(tickets: pd.Series) -> pd.Series:
    """Normalize the complete ticket string for equality-based group features."""

    text = _string_series(tickets, "tickets").str.strip().str.upper()
    text = text.str.replace(r"\s+", " ", regex=True)
    return text.fillna("UNKNOWN").replace("", "UNKNOWN").rename("TicketKey")


def extract_cabin_deck(cabins: pd.Series) -> pd.Series:
    """Return the first alphabetic cabin-deck letter or ``"U"`` if unknown."""

    text = _string_series(cabins, "cabins").str.strip().str.upper()
    deck = text.str.extract(r"([A-Z])", expand=False)
    return deck.fillna("U").astype("string").rename("CabinDeck")


def cabin_is_known(cabins: pd.Series) -> pd.Series:
    """Return whether non-empty cabin information is present."""

    text = _string_series(cabins, "cabins")
    known = text.notna() & text.str.strip().ne("")
    return known.astype("bool").rename("CabinKnown")


def count_cabins(cabins: pd.Series) -> pd.Series:
    """Count cabin identifiers, returning zero if cabin information is missing."""

    text = _string_series(cabins, "cabins").str.strip()
    counts = text.map(
        lambda value: (
            0
            if pd.isna(value) or not str(value)
            else max(1, len(re.findall(r"[A-Z]?\d+", str(value).upper())))
        )
    )
    return counts.astype("Int64").rename("NumberOfCabins")


def calculate_family_size(data: pd.DataFrame) -> pd.Series:
    """Calculate ``SibSp + Parch + 1`` with missing counts treated as zero."""

    _required_columns(data, {"SibSp", "Parch"})
    counts = data.loc[:, ["SibSp", "Parch"]].apply(pd.to_numeric, errors="raise")
    counts = counts.fillna(0)
    if (counts < 0).any().any() or ((counts % 1) != 0).any().any():
        raise ValueError("SibSp and Parch must contain non-negative whole numbers.")
    return (counts.sum(axis=1) + 1).astype("Int64").rename("FamilySize")


def create_is_alone(family_size: pd.Series) -> pd.Series:
    """Return whether each row has a family size of exactly one."""

    if not isinstance(family_size, pd.Series):
        raise TypeError("family_size must be a pandas Series.")
    numeric = pd.to_numeric(family_size, errors="raise")
    if numeric.isna().any() or (numeric < 1).any():
        raise ValueError("family_size must contain positive values.")
    return numeric.eq(1).astype("bool").rename("IsAlone")


def create_family_role_indicators(data: pd.DataFrame) -> pd.DataFrame:
    """Create conservative child, mother, and father row-level indicators."""

    _required_columns(data, {"Age", "Sex", "Parch"})
    age = pd.to_numeric(data["Age"], errors="coerce")
    parch = pd.to_numeric(data["Parch"], errors="coerce").fillna(0)
    sex = data["Sex"].astype("string").str.strip().str.lower()
    adult_with_child = age.ge(18) & parch.gt(0)
    return pd.DataFrame(
        {
            "IsChild": age.lt(16).fillna(False),
            "IsMother": (sex.eq("female") & adult_with_child).fillna(False),
            "IsFather": (sex.eq("male") & adult_with_child).fillna(False),
        },
        index=data.index,
        dtype="bool",
    )


def make_family_key(data: pd.DataFrame) -> pd.Series:
    """Create a target-independent surname and family-size group identifier."""

    _required_columns(data, {"Name", "SibSp", "Parch"})
    surname = extract_surname(data["Name"]).str.upper()
    family_size = calculate_family_size(data).astype("string")
    return (surname + "|" + family_size).rename("FamilyKey")


def _group_size_category(values: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(values, errors="raise")
    categories = np.select(
        [numeric.eq(1), numeric.eq(2), numeric.between(3, 4)],
        ["Solo", "Pair", "Small"],
        default="Large",
    )
    return pd.Series(categories, index=values.index, dtype="string")


def add_basic_features(data: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with deterministic row-level engineered columns."""

    _required_columns(data, _BASE_REQUIRED_COLUMNS)
    result = data.copy()
    result["Title"] = extract_title(data["Name"])
    result["Surname"] = extract_surname(data["Name"])
    result["TicketPrefix"] = normalize_ticket_prefix(data["Ticket"])
    result["CabinDeck"] = extract_cabin_deck(data["Cabin"])
    result["CabinKnown"] = cabin_is_known(data["Cabin"])
    result["NumberOfCabins"] = count_cabins(data["Cabin"])
    result["FamilySize"] = calculate_family_size(data)
    result["IsAlone"] = create_is_alone(result["FamilySize"])
    roles = create_family_role_indicators(data)
    result[list(roles.columns)] = roles
    result["AgeMissing"] = data["Age"].isna()
    result["FareMissing"] = data["Fare"].isna()
    fare = pd.to_numeric(data["Fare"], errors="coerce")
    if fare.dropna().lt(0).any():
        raise ValueError("Fare must not contain negative values.")
    result["LogFare"] = np.log1p(fare)
    result["FamilySizeCategory"] = _group_size_category(result["FamilySize"])
    sex = data["Sex"].astype("string").str.strip().str.lower().fillna(UNKNOWN)
    pclass = data["Pclass"].astype("string").fillna(UNKNOWN)
    result["SexClass"] = (sex + "_class_" + pclass).astype("string")
    return result


def feature_columns(feature_set: str) -> tuple[str, ...]:
    """Return the stable output columns for a named feature set."""

    try:
        return FEATURE_SETS[feature_set]
    except KeyError as exc:
        choices = ", ".join(FEATURE_SETS)
        raise ValueError(
            f"Unknown feature_set {feature_set!r}; choose from {choices}."
        ) from exc


def categorical_columns(feature_set: str) -> list[str]:
    """Return categorical output names for a named feature set."""

    return [
        column
        for column in feature_columns(feature_set)
        if column in CATEGORICAL_FEATURES
    ]


def numeric_columns(feature_set: str) -> list[str]:
    """Return numeric output names for a named feature set."""

    return [
        column
        for column in feature_columns(feature_set)
        if column not in CATEGORICAL_FEATURES
    ]


class TitanicFeatureBuilder(TransformerMixin, BaseEstimator):
    """Build a selected feature set with fold-fitted group-count mappings."""

    def __init__(self, feature_set: str = "basic") -> None:
        self.feature_set = feature_set

    def fit(self, X: pd.DataFrame, y: object = None) -> TitanicFeatureBuilder:
        """Fit target-independent count mappings on supplied training rows only."""

        del y
        _required_columns(X, _BASE_REQUIRED_COLUMNS)
        feature_columns(self.feature_set)
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        if self.feature_set == "group_counts":
            self.ticket_counts_ = (
                normalize_ticket_id(X["Ticket"]).value_counts().to_dict()
            )
            self.family_counts_ = make_family_key(X).value_counts().to_dict()
        else:
            self.ticket_counts_ = {}
            self.family_counts_ = {}
        self.is_fitted_ = True
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform rows without learning from them."""

        check_is_fitted(self, "is_fitted_")
        _required_columns(X, _BASE_REQUIRED_COLUMNS)
        result = add_basic_features(X)
        if self.feature_set == "group_counts":
            ticket_key = normalize_ticket_id(X["Ticket"])
            family_key = make_family_key(X)
            result["TicketGroupSize"] = (
                ticket_key.map(self.ticket_counts_).fillna(1).astype("int64")
            )
            result["FamilyGroupSize"] = (
                family_key.map(self.family_counts_).fillna(1).astype("int64")
            )
            fare = pd.to_numeric(result["Fare"], errors="coerce")
            result["FarePerTicketMember"] = fare / result["TicketGroupSize"]
            result["FarePerFamilyMember"] = fare / result["FamilyGroupSize"]
            result["TicketGroupCategory"] = _group_size_category(
                result["TicketGroupSize"]
            )
            result["FamilyGroupCategory"] = _group_size_category(
                result["FamilyGroupSize"]
            )

        columns = list(feature_columns(self.feature_set))
        output = result.loc[:, columns].copy()
        for column in categorical_columns(self.feature_set):
            output[column] = (
                output[column].astype("string").fillna(UNKNOWN).replace("", UNKNOWN)
            )
        for column in numeric_columns(self.feature_set):
            output[column] = pd.to_numeric(output[column], errors="coerce").astype(
                float
            )
        return output

    def get_feature_names_out(self, input_features: object = None) -> np.ndarray:
        """Return stable output names for sklearn compatibility."""

        del input_features
        return np.asarray(feature_columns(self.feature_set), dtype=object)


def build_validation_groups(data: pd.DataFrame, kind: str) -> pd.Series:
    """Build family or ticket groups, assigning unique keys to singletons."""

    _required_columns(data, _BASE_REQUIRED_COLUMNS)
    if kind == "family":
        base = make_family_key(data)
    elif kind == "ticket":
        base = normalize_ticket_id(data["Ticket"])
    else:
        raise ValueError("kind must be 'family' or 'ticket'.")
    counts = base.value_counts()
    singleton = base.map(counts).eq(1)
    groups = base.astype("string").copy()
    groups.loc[singleton] = (
        "SINGLE_"
        + kind.upper()
        + "_"
        + data.loc[singleton, "PassengerId"].astype("string")
    )
    return groups.rename(f"{kind.title()}Group")
