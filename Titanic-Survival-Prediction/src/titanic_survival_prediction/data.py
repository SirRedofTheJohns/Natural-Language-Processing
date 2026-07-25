"""Official Titanic data loading, validation, and aggregate audit helpers."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from titanic_survival_prediction.features import (
    UNKNOWN,
    add_basic_features,
    make_family_key,
    normalize_ticket_id,
)

FEATURE_COLUMNS: tuple[str, ...] = (
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
)
TRAIN_COLUMNS: tuple[str, ...] = (
    "PassengerId",
    "Survived",
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
)
SUBMISSION_COLUMNS: tuple[str, ...] = ("PassengerId", "Survived")
OFFICIAL_FILENAMES: tuple[str, ...] = (
    "gender_submission.csv",
    "test.csv",
    "train.csv",
)


@dataclass(frozen=True)
class CompetitionData:
    """Validated official train, test, and sample-submission frames."""

    train: pd.DataFrame
    test: pd.DataFrame
    sample_submission: pd.DataFrame


def _validate_ids(frame: pd.DataFrame, label: str) -> None:
    if frame["PassengerId"].isna().any():
        raise ValueError(f"{label} PassengerId contains missing values.")
    if not frame["PassengerId"].is_unique:
        raise ValueError(f"{label} PassengerId must be unique.")
    if not frame["PassengerId"].gt(0).all():
        raise ValueError(f"{label} PassengerId must be positive.")


def validate_competition_frames(
    train: pd.DataFrame,
    test: pd.DataFrame,
    sample_submission: pd.DataFrame,
) -> None:
    """Validate official schemas without inspecting hidden labels."""

    if tuple(train.columns) != TRAIN_COLUMNS:
        raise ValueError(
            f"train.csv columns differ from the expected schema: {list(train.columns)}"
        )
    if tuple(test.columns) != FEATURE_COLUMNS:
        raise ValueError(
            f"test.csv columns differ from the expected schema: {list(test.columns)}"
        )
    if tuple(sample_submission.columns) != SUBMISSION_COLUMNS:
        raise ValueError("gender_submission.csv must contain PassengerId and Survived.")
    if "Survived" in test.columns:
        raise ValueError("test.csv must not contain Survived.")
    if set(train["Survived"].dropna().unique()) != {0, 1}:
        raise ValueError("train.csv Survived must contain both binary classes.")
    _validate_ids(train, "train.csv")
    _validate_ids(test, "test.csv")
    _validate_ids(sample_submission, "gender_submission.csv")
    if (
        not sample_submission["PassengerId"]
        .reset_index(drop=True)
        .equals(test["PassengerId"].reset_index(drop=True))
    ):
        raise ValueError("Sample-submission PassengerId order must match test.csv.")
    if not sample_submission["Survived"].isin([0, 1]).all():
        raise ValueError("Sample submission must contain binary predictions.")
    if (~train["Pclass"].isin([1, 2, 3])).any() or (
        ~test["Pclass"].isin([1, 2, 3])
    ).any():
        raise ValueError("Pclass must be 1, 2, or 3.")
    for frame, label in ((train, "train.csv"), (test, "test.csv")):
        if frame["Age"].dropna().lt(0).any():
            raise ValueError(f"{label} contains a negative Age.")
        if frame["Fare"].dropna().lt(0).any():
            raise ValueError(f"{label} contains a negative Fare.")
        if frame[["SibSp", "Parch"]].lt(0).any().any():
            raise ValueError(f"{label} contains a negative relative count.")


def load_competition_data(data_dir: str | Path) -> CompetitionData:
    """Load and validate the three official competition CSV files."""

    directory = Path(data_dir)
    expected = {directory / name for name in OFFICIAL_FILENAMES}
    missing = sorted(path.name for path in expected if not path.is_file())
    if missing:
        raise FileNotFoundError(f"Missing official files: {', '.join(missing)}")
    csv_files = {path.name for path in directory.glob("*.csv")}
    unexpected = sorted(csv_files.difference(OFFICIAL_FILENAMES))
    if unexpected:
        raise ValueError(
            f"Unexpected CSV files in official data directory: {unexpected}"
        )
    train = pd.read_csv(directory / "train.csv")
    test = pd.read_csv(directory / "test.csv")
    sample = pd.read_csv(directory / "gender_submission.csv")
    validate_competition_frames(train, test, sample)
    return CompetitionData(train=train, test=test, sample_submission=sample)


def sha256_file(path: str | Path) -> str:
    """Return a streaming SHA-256 hash for a file."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(65_536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_data_manifest(data_dir: str | Path, data: CompetitionData) -> dict[str, Any]:
    """Build a tracked aggregate manifest without passenger rows."""

    directory = Path(data_dir)
    frames = {
        "train.csv": data.train,
        "test.csv": data.test,
        "gender_submission.csv": data.sample_submission,
    }
    return {
        "acquired_utc": datetime.now(UTC).isoformat(),
        "official_source": "https://www.kaggle.com/competitions/titanic/data",
        "competition": "titanic",
        "files": {
            name: {
                "sha256": sha256_file(directory / name),
                "bytes": (directory / name).stat().st_size,
                "rows": int(len(frame)),
                "columns": int(len(frame.columns)),
                "column_names": frame.columns.tolist(),
                "dtypes": frame.dtypes.astype(str).to_dict(),
                "missing_counts": {
                    key: int(value) for key, value in frame.isna().sum().items()
                },
            }
            for name, frame in frames.items()
        },
    }


def _group_structure(
    train_key: pd.Series,
    test_key: pd.Series,
    target: pd.Series,
) -> dict[str, int]:
    train_counts = train_key.value_counts()
    test_counts = test_key.value_counts()
    outcomes = (
        pd.DataFrame({"group": train_key, "target": target})
        .groupby("group")
        .agg(size=("target", "size"), outcomes=("target", "nunique"))
    )
    return {
        "train_repeated_groups": int(train_counts.gt(1).sum()),
        "test_repeated_groups": int(test_counts.gt(1).sum()),
        "train_rows_in_multi_member_groups": int(
            train_key.map(train_counts).gt(1).sum()
        ),
        "test_rows_in_multi_member_groups": int(test_key.map(test_counts).gt(1).sum()),
        "overlap_groups": int(len(set(train_key) & set(test_key))),
        "conflicting_train_groups": int(
            (outcomes["size"].gt(1) & outcomes["outcomes"].gt(1)).sum()
        ),
    }


def build_data_audit(data: CompetitionData) -> dict[str, Any]:
    """Return focused aggregate data-quality and relationship diagnostics."""

    train = data.train
    test = data.test
    train_features = add_basic_features(train)
    test_features = add_basic_features(test)
    train_ticket = normalize_ticket_id(train["Ticket"])
    test_ticket = normalize_ticket_id(test["Ticket"])
    train_family = make_family_key(train)
    test_family = make_family_key(test)

    def records(frame: pd.DataFrame) -> list[dict[str, Any]]:
        return frame.reset_index().to_dict(orient="records")

    survival_by_sex_class = (
        train.groupby(["Sex", "Pclass"])["Survived"]
        .agg(passenger_count="size", survival_rate="mean")
        .round(6)
    )
    survival_by_title = (
        train_features.groupby("Title")["Survived"]
        .agg(passenger_count="size", survival_rate="mean")
        .round(6)
    )
    survival_by_family = (
        train_features.groupby("FamilySize")["Survived"]
        .agg(passenger_count="size", survival_rate="mean")
        .round(6)
    )

    return {
        "schema": {
            "train_shape": list(train.shape),
            "test_shape": list(test.shape),
            "sample_submission_shape": list(data.sample_submission.shape),
            "train_exact_duplicates": int(train.duplicated().sum()),
            "test_exact_duplicates": int(test.duplicated().sum()),
            "train_duplicates_excluding_id": int(
                train.drop(columns=["PassengerId", "Survived"]).duplicated().sum()
            ),
            "test_duplicates_excluding_id": int(
                test.drop(columns=["PassengerId"]).duplicated().sum()
            ),
        },
        "target": {
            "counts": {
                str(key): int(value)
                for key, value in train["Survived"].value_counts().sort_index().items()
            },
            "proportions": {
                str(key): float(value)
                for key, value in train["Survived"]
                .value_counts(normalize=True)
                .sort_index()
                .items()
            },
            "gender_rule_training_accuracy": float(
                train["Survived"].eq(train["Sex"].eq("female").astype(int)).mean()
            ),
        },
        "missing_counts": {
            "train": {key: int(value) for key, value in train.isna().sum().items()},
            "test": {key: int(value) for key, value in test.isna().sum().items()},
        },
        "numeric_summary": {
            "train": train[["Age", "Fare", "SibSp", "Parch"]]
            .describe(percentiles=[0.01, 0.25, 0.5, 0.75, 0.95, 0.99])
            .round(6)
            .to_dict(),
            "test": test[["Age", "Fare", "SibSp", "Parch"]]
            .describe(percentiles=[0.01, 0.25, 0.5, 0.75, 0.95, 0.99])
            .round(6)
            .to_dict(),
        },
        "parser_coverage": {
            "train_unknown_titles": int(train_features["Title"].eq(UNKNOWN).sum()),
            "test_unknown_titles": int(test_features["Title"].eq(UNKNOWN).sum()),
            "train_rare_titles": int(train_features["Title"].eq("Rare").sum()),
            "test_rare_titles": int(test_features["Title"].eq("Rare").sum()),
            "train_unknown_surnames": int(train_features["Surname"].eq(UNKNOWN).sum()),
            "test_unknown_surnames": int(test_features["Surname"].eq(UNKNOWN).sum()),
            "train_numeric_only_tickets": int(
                train_features["TicketPrefix"].eq("NONE").sum()
            ),
            "test_numeric_only_tickets": int(
                test_features["TicketPrefix"].eq("NONE").sum()
            ),
            "train_multiple_cabins": int(train_features["NumberOfCabins"].gt(1).sum()),
            "test_multiple_cabins": int(test_features["NumberOfCabins"].gt(1).sum()),
        },
        "group_structure": {
            "ticket": _group_structure(train_ticket, test_ticket, train["Survived"]),
            "family": _group_structure(train_family, test_family, train["Survived"]),
        },
        "aggregate_survival": {
            "sex_class": records(survival_by_sex_class),
            "title": records(survival_by_title),
            "family_size": records(survival_by_family),
        },
        "covariate_shift": {
            "missing_rate_difference_test_minus_train": {
                key: float(value)
                for key, value in (
                    test.isna().mean() - train.drop(columns="Survived").isna().mean()
                ).items()
            },
            "age_median_train": float(train["Age"].median()),
            "age_median_test": float(test["Age"].median()),
            "fare_median_train": float(train["Fare"].median()),
            "fare_median_test": float(test["Fare"].median()),
            "family_size_mean_train": float(train_features["FamilySize"].mean()),
            "family_size_mean_test": float(test_features["FamilySize"].mean()),
            "ticket_overlap_groups": int(len(set(train_ticket) & set(test_ticket))),
            "family_overlap_groups": int(len(set(train_family) & set(test_family))),
        },
        "decisions": [
            "Merge 'the Countess' into the normalized Nobility title group.",
            "Add NumberOfCabins because multi-cabin strings occur in both files.",
            (
                "Retain AgeMissing, FareMissing, LogFare, and coarse "
                "family-size categories."
            ),
            (
                "Fit ticket/family count mappings on each training fold; "
                "unseen groups map to one."
            ),
            (
                "Do not use target-derived group priors: many repeated training "
                "groups have conflicting outcomes and the added leakage "
                "complexity is not required."
            ),
            (
                "Do not use transductive train/test counts because explicit "
                "permission was not confirmed."
            ),
        ],
    }
