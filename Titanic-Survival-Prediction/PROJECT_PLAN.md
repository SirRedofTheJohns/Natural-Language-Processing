# Titanic Survival Prediction — executed project record

## Project definition

**Problem:** predict survival in Kaggle's historical Titanic benchmark.
**Audience:** technical recruiters and data practitioners. **Input:** official
passenger-manifest covariates. **Transformation:** audit, leakage-safe feature
engineering, repeated validation, controlled optimization, and interpretation.
**Output:** binary predictions in Kaggle's required schema. **Demonstrated
value:** reproducible tabular modeling with explicit integrity controls, not a
modern safety or life-critical system.

## Official-source research

Research was performed on 2026-07-25 using only primary documentation:

- [Titanic overview](https://www.kaggle.com/competitions/titanic/overview)
- [Evaluation and submission format](https://www.kaggle.com/competitions/titanic/overview/evaluation)
- [Official data page](https://www.kaggle.com/competitions/titanic/data)
- [Competition rules](https://www.kaggle.com/competitions/titanic/rules)
- [Kaggle CLI authentication](https://github.com/Kaggle/kaggle-cli/blob/main/docs/README.md)
- [Kaggle CLI tutorial](https://github.com/Kaggle/kaggle-cli/blob/main/docs/tutorials.md)
- [Kaggle competition commands](https://github.com/Kaggle/kaggle-cli/blob/main/docs/competitions.md)
- [RepeatedStratifiedKFold](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.RepeatedStratifiedKFold.html)
- [StratifiedGroupKFold](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedGroupKFold.html)
- [HistGradientBoostingClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.HistGradientBoostingClassifier.html)
- [CatBoostClassifier](https://catboost.ai/docs/en/concepts/python-reference_catboostclassifier)

The official CLI authenticated successfully, listed the three competition
files, and downloaded `train.csv`, `test.csv`, and `gender_submission.csv`.
The dynamically rendered competition pages did not expose stable
Titanic-specific text for every rules detail. The conservative policy was
therefore: no external data, no transductive train/test transforms, no overall
leaderboard inspection, and no public-score-driven model changes.

## Phase completion record

| Phase | Completed work and files | Validation and decisions |
|---|---|---|
| 1 — environment and acquisition | Created `.venv`; corrected `pyproject.toml`; authenticated with the official CLI; downloaded and extracted the three official files; wrote `reports/environment.json` and `reports/data_manifest.json`. | Python 3.12.5; editable development install; authenticated file listing; schema and SHA-256 checks. The downloaded archive was removed and raw data remains ignored. |
| 2 — audit and feature foundations | Completed aggregate audit/EDA; expanded `features.py`; added `data.py`, parser tests, audit metrics, figures, and the executed audit notebook. | Real-data parser coverage, missingness, duplicate, range, train/test-covariate, and family/ticket checks passed. Added the observed `the countess` title alias. Target priors and transduction were omitted. |
| 3 — baselines and validation | Added model-specific pipelines, shared repeated splits, group-aware diagnostics, metric artifacts, and synthetic model tests. | Same 25 folds for every primary candidate; family and ticket groups kept separate as stress tests. |
| 4 — advanced models and tuning | Compared CatBoost, ExtraTrees, and HistGradientBoosting; ran 120 bounded tuning fits; saved all trials and summaries. | CatBoost and HistGradientBoosting proceeded. XGBoost, LightGBM, Optuna, and neural networks were unnecessary and not added. |
| 5 — lock strategy | Ran four ablations, threshold grid, coarse soft vote, subgroup analysis, permutation importance, and second-seed validation. | Chose the basic-set HistGradientBoosting model at 0.575. Rejected the ensemble and fold-fitted counts under predeclared criteria. |
| 6 — final fit and submissions | Refit the locked pipeline; generated and validated the ignored baseline and selected-model CSVs; uploaded the one qualified model; wrote the tracked submission log. | Exact 418-row schema, ID order, binary labels, and SHA-256 were verified. Submission 54982397 completed with public score 0.77033; it caused no retuning. |
| 7 — portfolio publication | Completed README, project record, executed notebooks, source/tests, figures, workflow, license, and final audits. | Ruff, pytest, compilation/AST, notebook, TOML/YAML, secret/path, ignore, staged-file, and Git checks are required before the single final commit. |

Deviations from the initial plan were evidence-based: no cross-fitted target
prior was implemented because conflicting group labels and added leakage
complexity did not justify it; no transductive experiment was run because
permission was not confirmed; one qualified final model replaced a larger
submission slate; and the ensemble was rejected rather than retained
automatically.

## Data record

| Item | Value |
|---|---:|
| Training shape | 891 × 12 |
| Test shape | 418 × 11 |
| Training survival rate | 0.3838 |
| Exact duplicate rows | 0 train / 0 test |
| Missing `Age` | 177 train / 86 test |
| Missing `Cabin` | 687 train / 327 test |
| Missing `Embarked` | 2 train / 0 test |
| Missing `Fare` | 0 train / 1 test |
| Sex-only rule training accuracy | 0.7868 |

Passenger IDs are positive and unique, labels are binary, no impossible
negative counts were found, and the sample submission has the expected 418
rows and exact `PassengerId,Survived` columns. Exact file hashes and schemas
are stored in `reports/data_manifest.json`.

## Integrity controls

- Hidden or reconstructed test labels, copied submissions, historical-outcome
  corrections, and hard-coded passenger predictions are prohibited.
- The test set is unlabeled covariate data used only for locked final inference.
- Imputation, encoding, scaling, tuning, and dataset-level counts are fitted on
  training folds only.
- Group counts use a custom transformer with unseen groups mapped to one.
- No target-derived family/ticket priors were used.
- No train/test covariate combination or external data was used.
- Threshold and ensemble decisions use training out-of-fold probabilities only.
- Kaggle feedback is recorded only after model selection is frozen and never
  changes the model.

## Executed experiment protocol

Primary comparison used one stored
`RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=20260725)`
design for every candidate. Metrics include accuracy, balanced accuracy,
precision, recall, F1, fold range, and fold standard deviation.

Family and normalized-ticket identifiers were tested separately with
five-fold `StratifiedGroupKFold`. These are stricter robustness diagnostics,
not leaderboard estimates. A second primary run used another fixed seed with
five folds and two repeats.

The candidate ladder was:

1. sex-only rule and most-frequent dummy;
2. regularized logistic regression;
3. ExtraTrees and HistGradientBoosting;
4. CatBoost as the only external booster.

The two strongest advanced candidates entered
`RandomizedSearchCV`: 12 configurations × 5 folds each, or 120 fits total.
Four cumulative feature sets were ablated under identical folds. A
0.35–0.65 threshold grid in 0.025 increments and a five-point two-model
soft-vote weight grid were then evaluated.

## Results and locked decision

| Candidate | Repeated-CV accuracy |
|---|---:|
| Dummy | 0.6162 ± 0.0023 |
| Sex-only rule | 0.7867 ± 0.0267 |
| ExtraTrees | 0.8222 ± 0.0269 |
| Logistic regression | 0.8258 ± 0.0248 |
| CatBoost, tuned configuration | 0.8263 ± 0.0230 |
| HistGradientBoosting, tuned configuration | 0.8280 ± 0.0241 |

Feature ablations with the tuned HistGradientBoosting configuration:

| Feature set | Primary | Family groups | Ticket groups |
|---|---:|---:|---:|
| Raw (7) | 0.8245 | 0.8092 | 0.8081 |
| Basic (18) | 0.8256 | 0.8193 | 0.8125 |
| Ticket/cabin (22) | 0.8280 | 0.8261 | 0.8204 |
| Fold-fitted counts (28) | 0.8278 | 0.8003 | 0.7946 |

Differences below 0.003 were treated as practical ties in favor of a simpler
stable configuration. Threshold 0.575 improved repeat-level mean accuracy to
**0.8323 ± 0.0231** and improved four of five repeats relative to 0.5. The
locked model therefore uses HistGradientBoosting with the 18-feature basic
set and threshold 0.575.

Locked robustness results are **0.8204 ± 0.0286** for family groups,
**0.8282 ± 0.0250** for ticket groups, and **0.8249 ± 0.0302** under the
second seed. The averaged out-of-fold confusion matrix is
`[[514, 35], [121, 221]]`.

The best coarse blend used 25% HistGradientBoosting and 75% CatBoost, but its
repeat-level mean was 0.0018 below the best single model and it improved only
two of five repeats. It failed the predeclared rule (gain ≥0.003 and at least
three repeat improvements), so no ensemble was fitted or submitted.

The locked `hist_gradient_boosting_selected.csv` was then uploaded once.
Kaggle submission **54982397** completed with public score **0.77033**. This
hidden-label public measurement is distinct from local CV, was obtained only
after the configuration was frozen, and was not used for any model change.

## Artifacts and reproducibility

- `reports/metrics/`: fold results, tuning trials, ablations, threshold and
  ensemble analyses, robustness, subgroups, importance, and the final lock.
- `reports/figures/`: aggregate EDA and model-evidence figures.
- `reports/data_manifest.json`: official file schemas, sizes, and SHA-256 hashes.
- `reports/environment.json`: exact Python and package versions.
- `reports/submission_log.csv`: generated candidates, hashes, and submission
  outcome fields without prediction rows.
- `notebooks/`: exactly two executed, GitHub-renderable narratives.

The authoritative command is:

```powershell
$env:MPLCONFIGDIR = (Join-Path (Get-Location) ".matplotlib")
$env:LOKY_MAX_CPU_COUNT = "4"
.\.venv\Scripts\python.exe -m titanic_survival_prediction.experiment `
  --data-dir data\raw --output-dir reports --submissions-dir submissions `
  --random-seed 20260725 --generate-submissions
```

## Limitations

Only 891 labels are available, folds contain related passengers, and score
variance is material. Historical social categories and missingness restrict
generalization. Accuracy obscures error asymmetry: the locked out-of-fold
predictions contain more false negatives than false positives. Subgroup
metrics and permutation importance describe predictive behavior, not causal
effects. Results must not be transferred to contemporary maritime policy or
individual survival decisions.
