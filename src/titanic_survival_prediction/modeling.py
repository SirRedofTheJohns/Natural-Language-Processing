"""Leakage-safe model construction, validation, tuning, and comparison."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.utils.validation import check_is_fitted

from titanic_survival_prediction.features import (
    TitanicFeatureBuilder,
    categorical_columns,
    numeric_columns,
)

MODEL_COMPLEXITY: dict[str, int] = {
    "gender_rule": 0,
    "dummy": 0,
    "logistic": 1,
    "hist_gradient_boosting": 2,
    "extra_trees": 3,
    "catboost": 4,
}

TUNING_SPACES: dict[str, dict[str, list[Any]]] = {
    "extra_trees": {
        "model__n_estimators": [250, 400, 600],
        "model__max_depth": [None, 5, 7, 10],
        "model__min_samples_leaf": [1, 2, 3, 5],
        "model__max_features": ["sqrt", 0.6, 0.9],
        "model__bootstrap": [False, True],
        "model__class_weight": [None, "balanced"],
    },
    "hist_gradient_boosting": {
        "model__learning_rate": [0.03, 0.05, 0.08, 0.12],
        "model__max_iter": [150, 250, 350],
        "model__max_leaf_nodes": [7, 15, 31],
        "model__min_samples_leaf": [10, 20, 30],
        "model__l2_regularization": [0.0, 0.5, 1.0, 2.0],
    },
    "catboost": {
        "model__depth": [4, 5, 6, 7],
        "model__learning_rate": [0.02, 0.04, 0.06, 0.1],
        "model__iterations": [250, 400, 600],
        "model__l2_leaf_reg": [1.0, 3.0, 5.0, 8.0],
        "model__random_strength": [0.25, 0.75, 1.5, 3.0],
    },
}


@dataclass
class EvaluationResult:
    """Fold metrics and aligned out-of-fold prediction records."""

    name: str
    summary: dict[str, Any]
    folds: pd.DataFrame
    predictions: pd.DataFrame


class CatBoostFrameClassifier(ClassifierMixin, BaseEstimator):
    """Clone-safe CatBoost adapter that supplies categorical columns at fit."""

    def __init__(
        self,
        categorical_features: tuple[str, ...] = (),
        iterations: int = 400,
        depth: int = 5,
        learning_rate: float = 0.04,
        l2_leaf_reg: float = 3.0,
        random_strength: float = 1.0,
        random_seed: int = 0,
    ) -> None:
        self.categorical_features = categorical_features
        self.iterations = iterations
        self.depth = depth
        self.learning_rate = learning_rate
        self.l2_leaf_reg = l2_leaf_reg
        self.random_strength = random_strength
        self.random_seed = random_seed

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series | np.ndarray,
    ) -> CatBoostFrameClassifier:
        """Fit CatBoost without modifying sklearn constructor parameters."""

        self.model_ = CatBoostClassifier(
            iterations=self.iterations,
            depth=self.depth,
            learning_rate=self.learning_rate,
            l2_leaf_reg=self.l2_leaf_reg,
            random_strength=self.random_strength,
            loss_function="Logloss",
            random_seed=self.random_seed,
            thread_count=1,
            verbose=False,
            allow_writing_files=False,
        )
        self.model_.fit(
            X,
            y,
            cat_features=list(self.categorical_features),
        )
        self.classes_ = np.asarray(self.model_.classes_)
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Return class probabilities."""

        check_is_fitted(self, "model_")
        return np.asarray(self.model_.predict_proba(X))

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Return flattened integer class predictions."""

        check_is_fitted(self, "model_")
        return np.asarray(self.model_.predict(X)).reshape(-1).astype("int64")


def _sklearn_preprocessor(feature_set: str, scale_numeric: bool) -> ColumnTransformer:
    numeric_steps: list[tuple[str, BaseEstimator]] = [
        ("imputer", SimpleImputer(strategy="median")),
    ]
    if scale_numeric:
        numeric_steps.append(("scaler", StandardScaler()))
    numeric_pipeline = Pipeline(numeric_steps)
    categorical_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "encoder",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False,
                ),
            ),
        ]
    )
    return ColumnTransformer(
        [
            ("numeric", numeric_pipeline, numeric_columns(feature_set)),
            (
                "categorical",
                categorical_pipeline,
                categorical_columns(feature_set),
            ),
        ],
        verbose_feature_names_out=False,
    )


def build_estimator(
    model_name: str,
    feature_set: str,
    seed: int,
    parameters: dict[str, Any] | None = None,
) -> BaseEstimator:
    """Build a model-specific fold-safe pipeline."""

    features = TitanicFeatureBuilder(feature_set=feature_set)
    if model_name == "catboost":
        model: BaseEstimator = CatBoostFrameClassifier(
            categorical_features=tuple(categorical_columns(feature_set)),
            iterations=400,
            depth=5,
            learning_rate=0.04,
            l2_leaf_reg=3.0,
            random_strength=1.0,
            random_seed=seed,
        )
        estimator: BaseEstimator = Pipeline([("features", features), ("model", model)])
    else:
        scale = model_name == "logistic"
        preprocessor = _sklearn_preprocessor(feature_set, scale_numeric=scale)
        if model_name == "dummy":
            model = DummyClassifier(strategy="most_frequent")
        elif model_name == "logistic":
            model = LogisticRegression(
                C=1.0,
                max_iter=2_000,
                random_state=seed,
            )
        elif model_name == "extra_trees":
            model = ExtraTreesClassifier(
                n_estimators=400,
                min_samples_leaf=3,
                max_features="sqrt",
                n_jobs=1,
                random_state=seed,
            )
        elif model_name == "hist_gradient_boosting":
            model = HistGradientBoostingClassifier(
                learning_rate=0.06,
                max_iter=250,
                max_leaf_nodes=15,
                min_samples_leaf=20,
                l2_regularization=1.0,
                random_state=seed,
            )
        else:
            raise ValueError(f"Unknown model_name: {model_name}")
        estimator = Pipeline(
            [
                ("features", features),
                ("preprocess", preprocessor),
                ("model", model),
            ]
        )
    if parameters:
        estimator.set_params(**parameters)
    return estimator


def _classification_metrics(
    target: pd.Series | np.ndarray,
    predictions: np.ndarray,
) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(target, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(target, predictions)),
        "precision": float(precision_score(target, predictions, zero_division=0)),
        "recall": float(recall_score(target, predictions, zero_division=0)),
        "f1": float(f1_score(target, predictions, zero_division=0)),
    }


def _summarize_folds(folds: pd.DataFrame) -> dict[str, Any]:
    summary: dict[str, Any] = {"fold_count": int(len(folds))}
    for metric in ("accuracy", "balanced_accuracy", "precision", "recall", "f1"):
        summary[f"{metric}_mean"] = float(folds[metric].mean())
        summary[f"{metric}_std"] = float(folds[metric].std(ddof=0))
    summary["accuracy_min"] = float(folds["accuracy"].min())
    summary["accuracy_max"] = float(folds["accuracy"].max())
    summary["fit_seconds_mean"] = float(folds["fit_seconds"].mean())
    return summary


def evaluate_estimator(
    name: str,
    estimator: BaseEstimator,
    X: pd.DataFrame,
    y: pd.Series,
    splits: list[tuple[np.ndarray, np.ndarray]],
    n_splits: int = 5,
) -> EvaluationResult:
    """Evaluate an estimator on precomputed splits and retain OOF probabilities."""

    fold_rows: list[dict[str, Any]] = []
    prediction_frames: list[pd.DataFrame] = []
    for split_number, (train_index, validation_index) in enumerate(splits):
        fitted = clone(estimator)
        started = time.perf_counter()
        fitted.fit(X.iloc[train_index], y.iloc[train_index])
        fit_seconds = time.perf_counter() - started
        probabilities = np.asarray(fitted.predict_proba(X.iloc[validation_index]))[:, 1]
        predictions = (probabilities >= 0.5).astype("int64")
        metrics = _classification_metrics(y.iloc[validation_index], predictions)
        fold_rows.append(
            {
                "model": name,
                "split": split_number,
                "repeat": split_number // n_splits,
                "fold": split_number % n_splits,
                "train_rows": int(len(train_index)),
                "validation_rows": int(len(validation_index)),
                "fit_seconds": fit_seconds,
                **metrics,
            }
        )
        prediction_frames.append(
            pd.DataFrame(
                {
                    "model": name,
                    "sample_index": validation_index,
                    "split": split_number,
                    "repeat": split_number // n_splits,
                    "fold": split_number % n_splits,
                    "target": y.iloc[validation_index].to_numpy(dtype="int64"),
                    "probability": probabilities,
                }
            )
        )
    folds = pd.DataFrame(fold_rows)
    return EvaluationResult(
        name=name,
        summary={"model": name, **_summarize_folds(folds)},
        folds=folds,
        predictions=pd.concat(prediction_frames, ignore_index=True),
    )


def evaluate_gender_rule(
    X: pd.DataFrame,
    y: pd.Series,
    splits: list[tuple[np.ndarray, np.ndarray]],
    n_splits: int = 5,
) -> EvaluationResult:
    """Evaluate the transparent women-survive rule on shared folds."""

    fold_rows: list[dict[str, Any]] = []
    prediction_frames: list[pd.DataFrame] = []
    for split_number, (_, validation_index) in enumerate(splits):
        predictions = (
            X.iloc[validation_index]["Sex"].astype("string").str.lower().eq("female")
        ).astype("int64")
        probabilities = predictions.to_numpy(dtype=float)
        metrics = _classification_metrics(y.iloc[validation_index], predictions)
        fold_rows.append(
            {
                "model": "gender_rule",
                "split": split_number,
                "repeat": split_number // n_splits,
                "fold": split_number % n_splits,
                "train_rows": int(len(y) - len(validation_index)),
                "validation_rows": int(len(validation_index)),
                "fit_seconds": 0.0,
                **metrics,
            }
        )
        prediction_frames.append(
            pd.DataFrame(
                {
                    "model": "gender_rule",
                    "sample_index": validation_index,
                    "split": split_number,
                    "repeat": split_number // n_splits,
                    "fold": split_number % n_splits,
                    "target": y.iloc[validation_index].to_numpy(dtype="int64"),
                    "probability": probabilities,
                }
            )
        )
    folds = pd.DataFrame(fold_rows)
    return EvaluationResult(
        name="gender_rule",
        summary={"model": "gender_rule", **_summarize_folds(folds)},
        folds=folds,
        predictions=pd.concat(prediction_frames, ignore_index=True),
    )


def tune_estimator(
    model_name: str,
    feature_set: str,
    X: pd.DataFrame,
    y: pd.Series,
    seed: int,
    n_iter: int = 12,
) -> tuple[dict[str, Any], pd.DataFrame, dict[str, Any]]:
    """Run a bounded transparent randomized search on one serious candidate."""

    if model_name not in TUNING_SPACES:
        raise ValueError(f"No tuning space is defined for {model_name}.")
    estimator = build_estimator(model_name, feature_set, seed)
    inner_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    search = RandomizedSearchCV(
        estimator=estimator,
        param_distributions=TUNING_SPACES[model_name],
        n_iter=n_iter,
        scoring="accuracy",
        cv=inner_cv,
        random_state=seed,
        n_jobs=1,
        refit=True,
        return_train_score=False,
        error_score="raise",
    )
    started = time.perf_counter()
    search.fit(X, y)
    elapsed = time.perf_counter() - started
    results = pd.DataFrame(search.cv_results_)
    selected = results[
        [
            "rank_test_score",
            "mean_test_score",
            "std_test_score",
            "mean_fit_time",
            "params",
        ]
    ].sort_values("rank_test_score")
    metadata = {
        "model": model_name,
        "feature_set": feature_set,
        "n_iter": n_iter,
        "folds_per_configuration": 5,
        "total_fits": n_iter * 5,
        "elapsed_seconds": elapsed,
        "search_best_mean_accuracy": float(search.best_score_),
        "best_params": search.best_params_,
    }
    return search.best_params_, selected.reset_index(drop=True), metadata


def threshold_analysis(
    records: pd.DataFrame,
    thresholds: np.ndarray,
) -> pd.DataFrame:
    """Evaluate a fixed threshold grid across aligned OOF repeats."""

    rows: list[dict[str, Any]] = []
    averaged = (
        records.groupby("sample_index")
        .agg(target=("target", "first"), probability=("probability", "mean"))
        .sort_index()
    )
    for threshold in thresholds:
        repeat_scores: list[float] = []
        for _, repeat_frame in records.groupby("repeat"):
            labels = (repeat_frame["probability"] >= threshold).astype("int64")
            repeat_scores.append(float(accuracy_score(repeat_frame["target"], labels)))
        averaged_labels = (averaged["probability"] >= threshold).astype("int64")
        rows.append(
            {
                "threshold": float(threshold),
                "repeat_accuracy_mean": float(np.mean(repeat_scores)),
                "repeat_accuracy_std": float(np.std(repeat_scores, ddof=0)),
                "repeat_accuracy_min": float(np.min(repeat_scores)),
                "repeat_accuracy_max": float(np.max(repeat_scores)),
                "averaged_oof_accuracy": float(
                    accuracy_score(averaged["target"], averaged_labels)
                ),
                "repeat_improvements_over_050": 0,
            }
        )
    result = pd.DataFrame(rows)
    base_scores: dict[int, float] = {}
    base_records = records.assign(label=(records["probability"] >= 0.5).astype("int64"))
    for repeat, repeat_frame in base_records.groupby("repeat"):
        base_scores[int(repeat)] = float(
            accuracy_score(repeat_frame["target"], repeat_frame["label"])
        )
    for index, row in result.iterrows():
        improvements = 0
        for repeat, repeat_frame in records.groupby("repeat"):
            labels = (repeat_frame["probability"] >= row["threshold"]).astype("int64")
            score = float(accuracy_score(repeat_frame["target"], labels))
            improvements += score > base_scores[int(repeat)]
        result.loc[index, "repeat_improvements_over_050"] = improvements
    return result


def choose_threshold(analysis: pd.DataFrame, minimum_gain: float = 0.003) -> float:
    """Choose a non-default threshold only for a stable material improvement."""

    baseline = analysis.loc[
        np.isclose(analysis["threshold"], 0.5), "repeat_accuracy_mean"
    ].iloc[0]
    best = analysis.sort_values(
        ["repeat_accuracy_mean", "repeat_accuracy_std"],
        ascending=[False, True],
    ).iloc[0]
    if (
        float(best["repeat_accuracy_mean"] - baseline) >= minimum_gain
        and int(best["repeat_improvements_over_050"]) >= 3
    ):
        return float(best["threshold"])
    return 0.5


def combine_prediction_records(
    first: EvaluationResult,
    second: EvaluationResult,
    first_weight: float,
    name: str,
) -> pd.DataFrame:
    """Create aligned soft-voting OOF prediction records."""

    keys = ["sample_index", "split", "repeat", "fold", "target"]
    merged = first.predictions.merge(
        second.predictions,
        on=keys,
        suffixes=("_first", "_second"),
        validate="one_to_one",
    )
    merged["probability"] = (
        first_weight * merged["probability_first"]
        + (1.0 - first_weight) * merged["probability_second"]
    )
    merged["model"] = name
    return merged[[*keys, "model", "probability"]]


def evaluate_prediction_records(
    name: str,
    records: pd.DataFrame,
    threshold: float = 0.5,
) -> EvaluationResult:
    """Summarize already-aligned OOF probabilities by their original folds."""

    fold_rows: list[dict[str, Any]] = []
    for split, frame in records.groupby("split", sort=True):
        labels = (frame["probability"] >= threshold).astype("int64")
        fold_rows.append(
            {
                "model": name,
                "split": int(split),
                "repeat": int(frame["repeat"].iloc[0]),
                "fold": int(frame["fold"].iloc[0]),
                "train_rows": 0,
                "validation_rows": int(len(frame)),
                "fit_seconds": 0.0,
                **_classification_metrics(frame["target"], labels),
            }
        )
    folds = pd.DataFrame(fold_rows)
    return EvaluationResult(
        name=name,
        summary={"model": name, **_summarize_folds(folds)},
        folds=folds,
        predictions=records.copy(),
    )


def coarse_ensemble_search(
    first: EvaluationResult,
    second: EvaluationResult,
) -> pd.DataFrame:
    """Evaluate a coarse two-model soft-voting weight grid at threshold 0.5."""

    rows: list[dict[str, Any]] = []
    for weight in np.linspace(0.0, 1.0, 5):
        records = combine_prediction_records(
            first,
            second,
            first_weight=float(weight),
            name="soft_voting",
        )
        evaluated = evaluate_prediction_records("soft_voting", records)
        rows.append(
            {
                "first_weight": float(weight),
                "second_weight": float(1.0 - weight),
                **{
                    key: value
                    for key, value in evaluated.summary.items()
                    if key != "model"
                },
            }
        )
    return pd.DataFrame(rows)


def averaged_oof(records: pd.DataFrame, threshold: float) -> pd.DataFrame:
    """Average repeated OOF probabilities to one prediction per training row."""

    averaged = (
        records.groupby("sample_index")
        .agg(target=("target", "first"), probability=("probability", "mean"))
        .sort_index()
    )
    averaged["prediction"] = (averaged["probability"] >= threshold).astype("int64")
    return averaged


def fold_permutation_importance(
    estimator: BaseEstimator,
    X: pd.DataFrame,
    y: pd.Series,
    splits: list[tuple[np.ndarray, np.ndarray]],
    seed: int,
) -> pd.DataFrame:
    """Compute fold-aggregated permutation importance on raw input concepts."""

    rows: list[dict[str, Any]] = []
    for fold, (train_index, validation_index) in enumerate(splits[:5]):
        fitted = clone(estimator)
        fitted.fit(X.iloc[train_index], y.iloc[train_index])
        result = permutation_importance(
            fitted,
            X.iloc[validation_index],
            y.iloc[validation_index],
            scoring="accuracy",
            n_repeats=5,
            random_state=seed + fold,
            n_jobs=1,
        )
        for feature, mean, std in zip(
            X.columns,
            result.importances_mean,
            result.importances_std,
            strict=True,
        ):
            rows.append(
                {
                    "fold": fold,
                    "feature": feature,
                    "importance_mean": float(mean),
                    "importance_std": float(std),
                }
            )
    frame = pd.DataFrame(rows)
    return (
        frame.groupby("feature")
        .agg(
            importance_mean=("importance_mean", "mean"),
            importance_std=("importance_mean", "std"),
        )
        .sort_values("importance_mean", ascending=False)
        .reset_index()
    )
