"""Run the authoritative leakage-safe Titanic experiment end to end."""

from __future__ import annotations

import argparse
import json
import platform
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedGroupKFold

from titanic_survival_prediction.data import (
    build_data_audit,
    build_data_manifest,
    load_competition_data,
    sha256_file,
)
from titanic_survival_prediction.features import (
    FEATURE_SETS,
    add_basic_features,
    build_validation_groups,
)
from titanic_survival_prediction.modeling import (
    MODEL_COMPLEXITY,
    EvaluationResult,
    averaged_oof,
    build_estimator,
    choose_threshold,
    coarse_ensemble_search,
    combine_prediction_records,
    evaluate_estimator,
    evaluate_gender_rule,
    evaluate_prediction_records,
    fold_permutation_importance,
    threshold_analysis,
    tune_estimator,
)
from titanic_survival_prediction.submission import (
    build_submission,
    save_submission,
    validate_submission_file,
)

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

PRIMARY_MODEL_NAMES = (
    "dummy",
    "logistic",
    "extra_trees",
    "hist_gradient_boosting",
    "catboost",
)
ADVANCED_MODEL_NAMES = (
    "extra_trees",
    "hist_gradient_boosting",
    "catboost",
)
FEATURE_SET_ORDER = ("raw", "basic", "ticket_cabin", "group_counts")


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return value.as_posix()
    if pd.isna(value):
        return None
    raise TypeError(f"Cannot serialize {type(value).__name__}.")


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=_json_default) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _write_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def _environment_report() -> dict[str, Any]:
    packages = (
        "catboost",
        "jupyter",
        "kaggle",
        "matplotlib",
        "nbclient",
        "nbformat",
        "numpy",
        "pandas",
        "pytest",
        "PyYAML",
        "ruff",
        "scikit-learn",
    )
    return {
        "generated_utc": datetime.now(UTC).isoformat(),
        "python": platform.python_version(),
        "implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "packages": {package: version(package) for package in packages},
    }


def _shared_splits(
    X: pd.DataFrame,
    y: pd.Series,
    seed: int,
    repeats: int = 5,
) -> list[tuple[np.ndarray, np.ndarray]]:
    splitter = RepeatedStratifiedKFold(
        n_splits=5,
        n_repeats=repeats,
        random_state=seed,
    )
    return list(splitter.split(X, y))


def _group_splits(
    X: pd.DataFrame,
    y: pd.Series,
    kind: str,
    seed: int,
) -> list[tuple[np.ndarray, np.ndarray]]:
    groups = build_validation_groups(X, kind)
    splitter = StratifiedGroupKFold(
        n_splits=5,
        shuffle=True,
        random_state=seed,
    )
    splits = list(splitter.split(X, y, groups))
    for train_index, validation_index in splits:
        train_groups = set(groups.iloc[train_index])
        validation_groups = set(groups.iloc[validation_index])
        if train_groups & validation_groups:
            raise RuntimeError(f"{kind} groups crossed a validation boundary.")
    return splits


def _choose_with_practical_tie(
    rows: pd.DataFrame,
    name_column: str,
    score_column: str = "accuracy_mean",
    tolerance: float = 0.003,
) -> str:
    best_score = float(rows[score_column].max())
    tied = rows.loc[rows[score_column].ge(best_score - tolerance)].copy()
    if name_column == "model_base":
        tied["complexity"] = tied[name_column].map(MODEL_COMPLEXITY)
    else:
        order = {name: index for index, name in enumerate(FEATURE_SET_ORDER)}
        tied["complexity"] = tied[name_column].map(order)
    return str(
        tied.sort_values(["complexity", score_column], ascending=[True, False]).iloc[0][
            name_column
        ]
    )


def _repeat_accuracies(records: pd.DataFrame, threshold: float) -> pd.Series:
    return records.groupby("repeat").apply(
        lambda frame: accuracy_score(
            frame["target"],
            frame["probability"].ge(threshold).astype("int64"),
        ),
        include_groups=False,
    )


def _subgroup_metrics(
    X: pd.DataFrame,
    averaged: pd.DataFrame,
) -> pd.DataFrame:
    frame = X.copy()
    frame["target"] = averaged["target"].to_numpy()
    frame["prediction"] = averaged["prediction"].to_numpy()
    engineered = add_basic_features(frame)
    age_group = (
        pd.cut(
            pd.to_numeric(engineered["Age"], errors="coerce"),
            bins=[-np.inf, 15, 29, 49, np.inf],
            labels=["Child", "Young adult", "Adult", "Older adult"],
        )
        .astype("string")
        .fillna("Age missing")
    )
    grouping = {
        "Sex": engineered["Sex"].astype("string"),
        "Pclass": engineered["Pclass"].astype("string"),
        "AgeGroup": age_group,
        "FamilySizeCategory": engineered["FamilySizeCategory"],
    }
    rows: list[dict[str, Any]] = []
    for dimension, values in grouping.items():
        diagnostic = pd.DataFrame(
            {
                "group": values,
                "target": engineered["target"],
                "prediction": engineered["prediction"],
            }
        )
        for group, subset in diagnostic.groupby("group", dropna=False):
            if len(subset) < 10:
                continue
            rows.append(
                {
                    "dimension": dimension,
                    "group": str(group),
                    "rows": int(len(subset)),
                    "accuracy": float(
                        accuracy_score(subset["target"], subset["prediction"])
                    ),
                    "balanced_accuracy": float(
                        balanced_accuracy_score(subset["target"], subset["prediction"])
                    ),
                    "precision": float(
                        precision_score(
                            subset["target"],
                            subset["prediction"],
                            zero_division=0,
                        )
                    ),
                    "recall": float(
                        recall_score(
                            subset["target"],
                            subset["prediction"],
                            zero_division=0,
                        )
                    ),
                    "f1": float(
                        f1_score(
                            subset["target"],
                            subset["prediction"],
                            zero_division=0,
                        )
                    ),
                    "false_positives": int(
                        (subset["prediction"].eq(1) & subset["target"].eq(0)).sum()
                    ),
                    "false_negatives": int(
                        (subset["prediction"].eq(0) & subset["target"].eq(1)).sum()
                    ),
                }
            )
    return pd.DataFrame(rows)


def _plot_audit_figures(
    train: pd.DataFrame,
    test: pd.DataFrame,
    figure_dir: Path,
) -> None:
    figure_dir.mkdir(parents=True, exist_ok=True)
    engineered = add_basic_features(train)
    test_engineered = add_basic_features(test)

    fig, axis = plt.subplots(figsize=(6.4, 4.2))
    counts = train["Survived"].value_counts().sort_index()
    axis.bar(
        ["Did not survive (0)", "Survived (1)"], counts, color=["#4C78A8", "#F58518"]
    )
    axis.set(title="Training target balance", ylabel="Passengers")
    for index, value in enumerate(counts):
        axis.text(index, value + 8, str(value), ha="center")
    fig.tight_layout()
    fig.savefig(figure_dir / "target_balance.png", dpi=160)
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(8, 4.5))
    missing = pd.DataFrame(
        {
            "Train": train.isna().mean(),
            "Test": test.isna().mean(),
        }
    ).fillna(0)
    missing.plot.bar(ax=axis, color=["#4C78A8", "#F58518"])
    axis.set(
        title="Missingness is concentrated in cabin and age", ylabel="Missing fraction"
    )
    axis.tick_params(axis="x", rotation=35)
    fig.tight_layout()
    fig.savefig(figure_dir / "missingness.png", dpi=160)
    plt.close(fig)

    rates = (
        train.groupby(["Sex", "Pclass"])["Survived"]
        .agg(rate="mean", count="size")
        .reset_index()
    )
    fig, axis = plt.subplots(figsize=(7.2, 4.5))
    for sex, subset in rates.groupby("Sex"):
        axis.plot(
            subset["Pclass"],
            subset["rate"],
            marker="o",
            linewidth=2,
            label=sex.title(),
        )
    axis.set(
        title="Observed survival rates differ by sex and ticket class",
        xlabel="Passenger class",
        ylabel="Training survival rate",
        xticks=[1, 2, 3],
        ylim=(0, 1.05),
    )
    axis.legend(title="Sex")
    fig.tight_layout()
    fig.savefig(figure_dir / "survival_by_sex_class.png", dpi=160)
    plt.close(fig)

    family = (
        engineered.groupby("FamilySize")["Survived"]
        .agg(rate="mean", count="size")
        .reset_index()
    )
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    axes[0].hist(
        [
            train.loc[train["Survived"].eq(0), "Age"].dropna(),
            train.loc[train["Survived"].eq(1), "Age"].dropna(),
        ],
        bins=16,
        label=["0", "1"],
        color=["#4C78A8", "#F58518"],
        alpha=0.75,
    )
    axes[0].set(title="Age distribution by target", xlabel="Age", ylabel="Passengers")
    axes[0].legend(title="Survived")
    axes[1].plot(family["FamilySize"], family["rate"], marker="o", color="#54A24B")
    axes[1].set(
        title="Observed survival rate by family size",
        xlabel="Family size aboard",
        ylabel="Training survival rate",
        ylim=(0, 1.05),
    )
    fig.tight_layout()
    fig.savefig(figure_dir / "age_and_family_patterns.png", dpi=160)
    plt.close(fig)

    comparison = pd.DataFrame(
        {
            "Train": [
                train["Age"].isna().mean(),
                train["Cabin"].isna().mean(),
                engineered["FamilySize"].mean(),
                train["Fare"].median(),
            ],
            "Test": [
                test["Age"].isna().mean(),
                test["Cabin"].isna().mean(),
                test_engineered["FamilySize"].mean(),
                test["Fare"].median(),
            ],
        },
        index=["Age missing", "Cabin missing", "Mean family size", "Median fare"],
    )
    normalized = comparison.div(comparison.max(axis=1), axis=0)
    fig, axis = plt.subplots(figsize=(7.4, 4.4))
    normalized.plot.bar(ax=axis, color=["#4C78A8", "#F58518"])
    axis.set(
        title="Selected train/test covariates are broadly similar",
        ylabel="Within-measure relative value",
    )
    axis.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    fig.savefig(figure_dir / "train_test_covariate_comparison.png", dpi=160)
    plt.close(fig)


def _plot_model_figures(
    model_comparison: pd.DataFrame,
    feature_ablation: pd.DataFrame,
    group_robustness: pd.DataFrame,
    confusion: np.ndarray,
    importance: pd.DataFrame,
    figure_dir: Path,
) -> None:
    fig, axis = plt.subplots(figsize=(8.5, 4.8))
    ordered = model_comparison.sort_values("accuracy_mean")
    axis.barh(
        ordered["model"],
        ordered["accuracy_mean"],
        xerr=ordered["accuracy_std"],
        color="#4C78A8",
        alpha=0.85,
    )
    axis.set(
        title="Repeated 5×5 cross-validation comparison",
        xlabel="Accuracy (mean ± fold standard deviation)",
        xlim=(0.55, min(0.9, ordered["accuracy_mean"].max() + 0.06)),
    )
    fig.tight_layout()
    fig.savefig(figure_dir / "model_cv_comparison.png", dpi=160)
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(7.5, 4.5))
    axis.errorbar(
        feature_ablation["feature_set"],
        feature_ablation["accuracy_mean"],
        yerr=feature_ablation["accuracy_std"],
        marker="o",
        linewidth=2,
        color="#54A24B",
    )
    axis.set(
        title="Feature ablation under identical repeated CV",
        xlabel="Feature set",
        ylabel="Accuracy",
    )
    axis.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    fig.savefig(figure_dir / "feature_ablation.png", dpi=160)
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(7.3, 4.3))
    robustness = group_robustness.set_index("validation")["accuracy_mean"]
    robustness.plot.bar(ax=axis, color=["#4C78A8", "#F58518", "#E45756"])
    axis.set(
        title="Group-aware validation is a stricter robustness stress test",
        xlabel="Validation design",
        ylabel="Accuracy",
    )
    axis.tick_params(axis="x", rotation=15)
    fig.tight_layout()
    fig.savefig(figure_dir / "group_robustness.png", dpi=160)
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(5.5, 4.8))
    ConfusionMatrixDisplay(confusion, display_labels=["0", "1"]).plot(
        ax=axis,
        cmap="Blues",
        colorbar=False,
    )
    axis.set_title("Averaged out-of-fold confusion matrix")
    fig.tight_layout()
    fig.savefig(figure_dir / "confusion_matrix.png", dpi=160)
    plt.close(fig)

    top = importance.head(10).sort_values("importance_mean")
    fig, axis = plt.subplots(figsize=(7.5, 5))
    axis.barh(
        top["feature"],
        top["importance_mean"],
        xerr=top["importance_std"].fillna(0),
        color="#B279A2",
    )
    axis.set(
        title="Fold-aggregated permutation importance",
        xlabel="Mean validation-accuracy decrease",
        ylabel="Raw input concept",
    )
    fig.tight_layout()
    fig.savefig(figure_dir / "feature_importance.png", dpi=160)
    plt.close(fig)


def _fit_predict(
    model_name: str,
    feature_set: str,
    parameters: dict[str, Any],
    seed: int,
    train_X: pd.DataFrame,
    y: pd.Series,
    test_X: pd.DataFrame,
) -> np.ndarray:
    estimator = build_estimator(model_name, feature_set, seed, parameters)
    estimator.fit(train_X, y)
    return np.asarray(estimator.predict_proba(test_X))[:, 1]


def run_experiment(
    data_dir: Path,
    report_dir: Path,
    submissions_dir: Path,
    seed: int,
    tune: bool,
    generate_submissions: bool,
) -> dict[str, Any]:
    """Execute the approved audit, model selection, and artifact generation."""

    metric_dir = report_dir / "metrics"
    figure_dir = report_dir / "figures"
    metric_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)

    data = load_competition_data(data_dir)
    train = data.train
    test = data.test
    X = train.drop(columns="Survived")
    y = train["Survived"].astype("int64")

    _write_json(report_dir / "data_manifest.json", build_data_manifest(data_dir, data))
    audit = build_data_audit(data)
    _write_json(metric_dir / "data_audit.json", audit)
    _write_json(report_dir / "environment.json", _environment_report())
    _plot_audit_figures(train, test, figure_dir)

    primary_splits = _shared_splits(X, y, seed, repeats=5)
    family_splits = _group_splits(X, y, "family", seed)
    ticket_splits = _group_splits(X, y, "ticket", seed)

    initial_results: dict[str, EvaluationResult] = {
        "gender_rule": evaluate_gender_rule(X, y, primary_splits)
    }
    for model_name in PRIMARY_MODEL_NAMES:
        estimator = build_estimator(model_name, "ticket_cabin", seed)
        initial_results[model_name] = evaluate_estimator(
            model_name,
            estimator,
            X,
            y,
            primary_splits,
        )

    initial_summary = pd.DataFrame(
        [result.summary for result in initial_results.values()]
    )
    initial_folds = pd.concat(
        [result.folds for result in initial_results.values()],
        ignore_index=True,
    )
    _write_csv(metric_dir / "baseline_comparison.csv", initial_summary.iloc[:3])
    _write_csv(metric_dir / "initial_model_comparison.csv", initial_summary)
    _write_csv(metric_dir / "initial_fold_results.csv", initial_folds)

    advanced_summary = initial_summary[
        initial_summary["model"].isin(ADVANCED_MODEL_NAMES)
    ].sort_values("accuracy_mean", ascending=False)
    tuning_candidates = advanced_summary.head(2)["model"].tolist()
    tuning_metadata: list[dict[str, Any]] = []
    tuning_trials: list[pd.DataFrame] = []
    tuned_parameters: dict[str, dict[str, Any]] = {}
    tuned_results: dict[str, EvaluationResult] = {}
    for model_name in tuning_candidates:
        if tune:
            best_params, trials, metadata = tune_estimator(
                model_name,
                "ticket_cabin",
                X,
                y,
                seed,
                n_iter=12,
            )
            trials.insert(0, "model", model_name)
            tuning_trials.append(trials)
            tuning_metadata.append(metadata)
            tuned_parameters[model_name] = best_params
        else:
            tuned_parameters[model_name] = {}
            tuning_metadata.append(
                {
                    "model": model_name,
                    "feature_set": "ticket_cabin",
                    "n_iter": 0,
                    "total_fits": 0,
                    "best_params": {},
                }
            )
        estimator = build_estimator(
            model_name,
            "ticket_cabin",
            seed,
            tuned_parameters[model_name],
        )
        tuned_results[model_name] = evaluate_estimator(
            f"{model_name}_selected_config",
            estimator,
            X,
            y,
            primary_splits,
        )
    if tuning_trials:
        tuning_frame = pd.concat(tuning_trials, ignore_index=True)
        tuning_frame["params"] = tuning_frame["params"].map(
            lambda value: json.dumps(value, sort_keys=True)
        )
        _write_csv(metric_dir / "tuning_trials.csv", tuning_frame)
    _write_json(metric_dir / "tuning_summary.json", tuning_metadata)

    tuned_summary = pd.DataFrame(
        [
            {
                **result.summary,
                "model_base": model_name,
                "parameters": json.dumps(
                    tuned_parameters[model_name],
                    sort_keys=True,
                ),
            }
            for model_name, result in tuned_results.items()
        ]
    )
    strongest_model = _choose_with_practical_tie(tuned_summary, "model_base")
    strongest_parameters = tuned_parameters[strongest_model]

    ablation_rows: list[dict[str, Any]] = []
    ablation_results: dict[str, EvaluationResult] = {}
    for feature_set in FEATURE_SET_ORDER:
        estimator = build_estimator(
            strongest_model,
            feature_set,
            seed,
            strongest_parameters,
        )
        primary = evaluate_estimator(
            f"{strongest_model}_{feature_set}",
            estimator,
            X,
            y,
            primary_splits,
        )
        family = evaluate_estimator(
            f"{strongest_model}_{feature_set}_family",
            estimator,
            X,
            y,
            family_splits,
        )
        ticket = evaluate_estimator(
            f"{strongest_model}_{feature_set}_ticket",
            estimator,
            X,
            y,
            ticket_splits,
        )
        ablation_results[feature_set] = primary
        ablation_rows.append(
            {
                "feature_set": feature_set,
                "feature_count": len(FEATURE_SETS[feature_set]),
                "accuracy_mean": primary.summary["accuracy_mean"],
                "accuracy_std": primary.summary["accuracy_std"],
                "accuracy_min": primary.summary["accuracy_min"],
                "accuracy_max": primary.summary["accuracy_max"],
                "family_group_accuracy": family.summary["accuracy_mean"],
                "ticket_group_accuracy": ticket.summary["accuracy_mean"],
                "notes": (
                    "Inductive counts fitted on fold training rows."
                    if feature_set == "group_counts"
                    else "Deterministic row features only."
                ),
            }
        )
    feature_ablation = pd.DataFrame(ablation_rows)
    feature_ablation["delta_from_prior"] = (
        feature_ablation["accuracy_mean"].diff().fillna(0.0)
    )
    _write_csv(metric_dir / "feature_ablation.csv", feature_ablation)
    selected_feature_set = _choose_with_practical_tie(
        feature_ablation,
        "feature_set",
    )

    final_candidate_results: dict[str, EvaluationResult] = {}
    for model_name in tuning_candidates:
        estimator = build_estimator(
            model_name,
            selected_feature_set,
            seed,
            tuned_parameters[model_name],
        )
        final_candidate_results[model_name] = evaluate_estimator(
            model_name,
            estimator,
            X,
            y,
            primary_splits,
        )
    candidate_summary = pd.DataFrame(
        [
            {
                **result.summary,
                "model_base": model_name,
                "feature_set": selected_feature_set,
                "parameters": json.dumps(
                    tuned_parameters[model_name],
                    sort_keys=True,
                ),
            }
            for model_name, result in final_candidate_results.items()
        ]
    )
    selected_single_model = _choose_with_practical_tie(
        candidate_summary,
        "model_base",
    )
    runner_up_model = next(
        model for model in tuning_candidates if model != selected_single_model
    )
    selected_single_result = final_candidate_results[selected_single_model]
    runner_up_result = final_candidate_results[runner_up_model]

    threshold_grid = np.round(np.arange(0.35, 0.651, 0.025), 3)
    single_threshold_frame = threshold_analysis(
        selected_single_result.predictions,
        threshold_grid,
    )
    selected_single_threshold = choose_threshold(single_threshold_frame)
    single_threshold_frame["selected"] = np.isclose(
        single_threshold_frame["threshold"],
        selected_single_threshold,
    )
    single_threshold_frame.insert(0, "candidate", selected_single_model)

    ensemble_grid = coarse_ensemble_search(
        selected_single_result,
        runner_up_result,
    )
    best_ensemble_row = ensemble_grid.sort_values(
        ["accuracy_mean", "accuracy_std"],
        ascending=[False, True],
    ).iloc[0]
    ensemble_weight = float(best_ensemble_row["first_weight"])
    ensemble_records = combine_prediction_records(
        selected_single_result,
        runner_up_result,
        first_weight=ensemble_weight,
        name="soft_voting",
    )
    single_repeat = _repeat_accuracies(
        selected_single_result.predictions,
        selected_single_threshold,
    )
    ensemble_repeat_at_half = _repeat_accuracies(ensemble_records, 0.5)
    ensemble_gain = float(ensemble_repeat_at_half.mean() - single_repeat.mean())
    ensemble_improved_repeats = int((ensemble_repeat_at_half > single_repeat).sum())
    ensemble_qualifies = (
        0.0 < ensemble_weight < 1.0
        and ensemble_gain >= 0.003
        and ensemble_improved_repeats >= 3
    )
    ensemble_threshold_frame = threshold_analysis(ensemble_records, threshold_grid)
    ensemble_threshold = (
        choose_threshold(ensemble_threshold_frame) if ensemble_qualifies else 0.5
    )
    ensemble_threshold_frame["selected"] = np.isclose(
        ensemble_threshold_frame["threshold"],
        ensemble_threshold,
    )
    ensemble_threshold_frame.insert(0, "candidate", "soft_voting")
    threshold_frame = pd.concat(
        [single_threshold_frame, ensemble_threshold_frame],
        ignore_index=True,
    )
    _write_csv(metric_dir / "threshold_analysis.csv", threshold_frame)
    ensemble_summary = {
        "component_models": [selected_single_model, runner_up_model],
        "weights": [ensemble_weight, 1.0 - ensemble_weight],
        "threshold": ensemble_threshold,
        "qualifies": ensemble_qualifies,
        "gain_over_best_single_repeat_mean": ensemble_gain,
        "repeat_improvements": ensemble_improved_repeats,
        "weight_grid": ensemble_grid.to_dict(orient="records"),
    }
    _write_json(metric_dir / "ensemble_analysis.json", ensemble_summary)

    if ensemble_qualifies:
        locked_name = "soft_voting"
        locked_records = ensemble_records
        locked_threshold = ensemble_threshold
    else:
        locked_name = selected_single_model
        locked_records = selected_single_result.predictions
        locked_threshold = selected_single_threshold
    locked_primary = evaluate_prediction_records(
        locked_name,
        locked_records,
        threshold=locked_threshold,
    )

    group_rows: list[dict[str, Any]] = [
        {
            "validation": "RepeatedStratifiedKFold 5x5",
            **{
                key: value
                for key, value in locked_primary.summary.items()
                if key != "model"
            },
        }
    ]
    group_evaluations: dict[str, EvaluationResult] = {}
    for kind, splits in (
        ("Family groups", family_splits),
        ("Ticket groups", ticket_splits),
    ):
        first_estimator = build_estimator(
            selected_single_model,
            selected_feature_set,
            seed,
            tuned_parameters[selected_single_model],
        )
        first_group = evaluate_estimator(
            f"{selected_single_model}_{kind}",
            first_estimator,
            X,
            y,
            splits,
        )
        if ensemble_qualifies:
            second_estimator = build_estimator(
                runner_up_model,
                selected_feature_set,
                seed,
                tuned_parameters[runner_up_model],
            )
            second_group = evaluate_estimator(
                f"{runner_up_model}_{kind}",
                second_estimator,
                X,
                y,
                splits,
            )
            records = combine_prediction_records(
                first_group,
                second_group,
                ensemble_weight,
                f"soft_voting_{kind}",
            )
            group_result = evaluate_prediction_records(
                f"soft_voting_{kind}",
                records,
                threshold=locked_threshold,
            )
        else:
            group_result = evaluate_prediction_records(
                f"{selected_single_model}_{kind}",
                first_group.predictions,
                threshold=locked_threshold,
            )
        group_evaluations[kind] = group_result
        group_rows.append(
            {
                "validation": kind,
                **{
                    key: value
                    for key, value in group_result.summary.items()
                    if key != "model"
                },
            }
        )
    group_robustness = pd.DataFrame(group_rows)
    _write_csv(metric_dir / "group_robustness.csv", group_robustness)

    averaged = averaged_oof(locked_records, locked_threshold)
    confusion = confusion_matrix(averaged["target"], averaged["prediction"])
    subgroup = _subgroup_metrics(X, averaged)
    _write_csv(metric_dir / "subgroup_metrics.csv", subgroup)

    selected_single_estimator = build_estimator(
        selected_single_model,
        selected_feature_set,
        seed,
        tuned_parameters[selected_single_model],
    )
    importance = fold_permutation_importance(
        selected_single_estimator,
        X,
        y,
        primary_splits,
        seed,
    )
    _write_csv(metric_dir / "feature_importance.csv", importance)

    second_seed_splits = _shared_splits(X, y, seed + 101, repeats=2)
    second_first = evaluate_estimator(
        f"{selected_single_model}_second_seed",
        selected_single_estimator,
        X,
        y,
        second_seed_splits,
    )
    if ensemble_qualifies:
        second_estimator = build_estimator(
            runner_up_model,
            selected_feature_set,
            seed + 101,
            tuned_parameters[runner_up_model],
        )
        second_second = evaluate_estimator(
            f"{runner_up_model}_second_seed",
            second_estimator,
            X,
            y,
            second_seed_splits,
        )
        second_records = combine_prediction_records(
            second_first,
            second_second,
            ensemble_weight,
            "soft_voting_second_seed",
        )
        stability = evaluate_prediction_records(
            "soft_voting_second_seed",
            second_records,
            locked_threshold,
        )
    else:
        stability = evaluate_prediction_records(
            f"{selected_single_model}_second_seed",
            second_first.predictions,
            locked_threshold,
        )

    model_comparison = pd.concat(
        [
            initial_summary,
            tuned_summary.drop(columns=["model_base", "parameters"]),
            candidate_summary.drop(columns=["model_base", "feature_set", "parameters"]),
        ],
        ignore_index=True,
    ).drop_duplicates(subset=["model"], keep="last")
    _write_csv(metric_dir / "model_comparison.csv", model_comparison)
    _write_csv(
        metric_dir / "final_fold_results.csv",
        locked_primary.folds,
    )

    final_selection = {
        "locked_utc": datetime.now(UTC).isoformat(),
        "name": locked_name,
        "feature_set": selected_feature_set,
        "single_model": selected_single_model,
        "single_model_parameters": tuned_parameters[selected_single_model],
        "single_model_threshold": selected_single_threshold,
        "ensemble": ensemble_summary if ensemble_qualifies else None,
        "threshold": locked_threshold,
        "seed": seed,
        "primary_validation": locked_primary.summary,
        "family_group_validation": group_evaluations["Family groups"].summary,
        "ticket_group_validation": group_evaluations["Ticket groups"].summary,
        "second_seed_validation": stability.summary,
        "confusion_matrix": confusion.tolist(),
        "selection_rule": (
            "Highest repeated-CV mean, treating differences under 0.003 as "
            "practically tied and preferring simpler stable configurations."
        ),
        "limitations": [
            (
                "Group-aware results are robustness diagnostics, not "
                "leaderboard estimates."
            ),
            (
                "No hidden test labels, external data, transductive counts, "
                "or leaderboard feedback were used."
            ),
            "Permutation importance is predictive association, not causality.",
        ],
    }
    _write_json(metric_dir / "final_model_selection.json", final_selection)

    _plot_model_figures(
        model_comparison,
        feature_ablation,
        group_robustness,
        confusion,
        importance,
        figure_dir,
    )

    generated_submissions: list[dict[str, Any]] = []
    if generate_submissions:
        submissions_dir.mkdir(parents=True, exist_ok=True)
        baseline = build_submission(
            test["PassengerId"],
            test["Sex"].astype("string").str.lower().eq("female").astype("int64"),
        )
        baseline_path = save_submission(
            baseline,
            submissions_dir / "baseline_gender.csv",
            expected_passenger_ids=test["PassengerId"],
        )
        generated_submissions.append(
            {
                "filename": baseline_path.name,
                "model": "gender_rule",
                "feature_set": "Sex only",
                "threshold": 0.5,
                "ensemble_weights": "",
                "path": baseline_path,
                "sha256": sha256_file(baseline_path),
                "submit": False,
            }
        )

        single_probability = _fit_predict(
            selected_single_model,
            selected_feature_set,
            tuned_parameters[selected_single_model],
            seed,
            X,
            y,
            test,
        )
        single_submission = build_submission(
            test["PassengerId"],
            (single_probability >= selected_single_threshold).astype("int64"),
        )
        single_path = save_submission(
            single_submission,
            submissions_dir / f"{selected_single_model}_selected.csv",
            expected_passenger_ids=test["PassengerId"],
        )
        generated_submissions.append(
            {
                "filename": single_path.name,
                "model": selected_single_model,
                "feature_set": selected_feature_set,
                "threshold": selected_single_threshold,
                "ensemble_weights": "",
                "path": single_path,
                "sha256": sha256_file(single_path),
                "submit": True,
            }
        )

        if ensemble_qualifies:
            runner_probability = _fit_predict(
                runner_up_model,
                selected_feature_set,
                tuned_parameters[runner_up_model],
                seed,
                X,
                y,
                test,
            )
            ensemble_probability = (
                ensemble_weight * single_probability
                + (1.0 - ensemble_weight) * runner_probability
            )
            ensemble_submission = build_submission(
                test["PassengerId"],
                (ensemble_probability >= locked_threshold).astype("int64"),
            )
            ensemble_path = save_submission(
                ensemble_submission,
                submissions_dir / "soft_voting_selected.csv",
                expected_passenger_ids=test["PassengerId"],
            )
            generated_submissions.append(
                {
                    "filename": ensemble_path.name,
                    "model": "soft_voting",
                    "feature_set": selected_feature_set,
                    "threshold": locked_threshold,
                    "ensemble_weights": json.dumps(
                        [ensemble_weight, 1.0 - ensemble_weight]
                    ),
                    "path": ensemble_path,
                    "sha256": sha256_file(ensemble_path),
                    "submit": True,
                }
            )
        for item in generated_submissions:
            validate_submission_file(item["path"], test["PassengerId"])

    submission_log_columns = [
        "timestamp_utc",
        "filename",
        "model",
        "feature_set",
        "threshold",
        "ensemble_weights",
        "repeated_cv_accuracy",
        "repeated_cv_std",
        "family_group_score",
        "ticket_group_score",
        "kaggle_public_score",
        "submission_ref",
        "notes",
    ]
    submission_rows = []
    for item in generated_submissions:
        submission_rows.append(
            {
                "timestamp_utc": datetime.now(UTC).isoformat(),
                "filename": item["filename"],
                "model": item["model"],
                "feature_set": item["feature_set"],
                "threshold": item["threshold"],
                "ensemble_weights": item["ensemble_weights"],
                "repeated_cv_accuracy": (
                    audit["target"]["gender_rule_training_accuracy"]
                    if item["model"] == "gender_rule"
                    else locked_primary.summary["accuracy_mean"]
                ),
                "repeated_cv_std": (
                    0.0
                    if item["model"] == "gender_rule"
                    else locked_primary.summary["accuracy_std"]
                ),
                "family_group_score": (
                    ""
                    if item["model"] == "gender_rule"
                    else group_evaluations["Family groups"].summary["accuracy_mean"]
                ),
                "ticket_group_score": (
                    ""
                    if item["model"] == "gender_rule"
                    else group_evaluations["Ticket groups"].summary["accuracy_mean"]
                ),
                "kaggle_public_score": "",
                "submission_ref": "",
                "notes": (
                    "Generated for schema validation; not intended for submission."
                    if not item["submit"]
                    else f"Validated SHA-256 {item['sha256']}"
                ),
            }
        )
    submission_log = pd.DataFrame(submission_rows, columns=submission_log_columns)
    _write_csv(report_dir / "submission_log.csv", submission_log)

    final_summary = {
        "generated_utc": datetime.now(UTC).isoformat(),
        "data": {
            "train_rows": int(len(train)),
            "test_rows": int(len(test)),
            "target_survival_rate": float(y.mean()),
        },
        "initial_model_comparison": initial_summary.to_dict(orient="records"),
        "tuning": tuning_metadata,
        "feature_ablation": feature_ablation.to_dict(orient="records"),
        "selected": final_selection,
        "subgroup_rows": int(len(subgroup)),
        "generated_submissions": [
            {
                key: (value.as_posix() if isinstance(value, Path) else value)
                for key, value in item.items()
                if key != "sha256"
            }
            for item in generated_submissions
        ],
        "integrity": {
            "external_data_used": False,
            "test_labels_accessed": False,
            "transductive_counts_used": False,
            "leaderboard_used_for_selection": False,
            "target_group_priors_used": False,
        },
    }
    _write_json(metric_dir / "final_project_summary.json", final_summary)
    return final_summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--output-dir", type=Path, default=Path("reports"))
    parser.add_argument(
        "--submissions-dir",
        type=Path,
        default=Path("submissions"),
    )
    parser.add_argument("--random-seed", type=int, default=20260725)
    parser.add_argument(
        "--no-tuning",
        action="store_true",
        help="Use the initial configurations without controlled tuning.",
    )
    parser.add_argument(
        "--generate-submissions",
        action="store_true",
        help="Fit locked configurations and create ignored submission CSVs.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    run_experiment(
        data_dir=args.data_dir,
        report_dir=args.output_dir,
        submissions_dir=args.submissions_dir,
        seed=args.random_seed,
        tune=not args.no_tuning,
        generate_submissions=args.generate_submissions,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
