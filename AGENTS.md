# Project contribution contract

## 1. User approval

Stop and obtain explicit user approval before:

- installing or upgrading dependencies;
- downloading competition data;
- authenticating with Kaggle;
- adding an external dataset;
- beginning expensive tuning;
- adding a major dependency;
- changing modeling logic outside an approved phase;
- generating or uploading a Kaggle submission;
- reading or recording leaderboard results;
- creating or changing a license;
- staging, committing, or pushing;
- deleting or moving files.

Within an approved phase, modify only its approved files and objectives. Preserve unrelated work.

## 2. Project communication

Every project-facing explanation begins with the problem, audience, input, transformation, output, and demonstrated value. The README must be understandable in about 30 seconds. Clearly distinguish measured results, hypotheses, non-authoritative community observations, and business implications. Never present this historical benchmark as a modern survival-decision system.

## 3. Integrity and leakage prevention

Never use hidden or reconstructed test labels, copied submissions, known historical outcomes as test corrections, hard-coded `PassengerId` predictions, public-leaderboard probing for model selection, reverse-engineered leaderboard scores, validation-row target statistics, or target encoding without strict out-of-fold construction.

All supervised preprocessing belongs inside each training fold, including imputation, encoding, feature selection, target-derived features, model tuning, probability-threshold selection, and ensemble-weight selection. Any target-derived family or ticket statistic must be constructed strictly out of fold.

The Kaggle test set is only unlabeled covariate data and may be used for final prediction after model-selection decisions are locked. Combining train and test covariates is prohibited unless current rules are verified to permit it; the transformation is entirely unsupervised and target-independent; it is documented; and it is evaluated separately from the ordinary non-transductive pipeline. Transduction is never the default.

## 4. Modeling discipline

Use this progressive ladder:

1. Kaggle sample or a transparent rule baseline.
2. `DummyClassifier`.
3. Logistic regression.
4. One tree-ensemble baseline.
5. No more than three serious champion candidates.

Research candidates from CatBoost, ExtraTrees, HistGradientBoosting, XGBoost, and LightGBM, but do not include every model. Keep the final candidate set compact and justify it by dataset size, data types, installation complexity, cross-validation evidence, and ensemble complementarity. Do not default to a neural network and do not preselect a champion.

## 5. Validation

Do not use a single train/validation split for model comparison. The primary comparison is repeated stratified cross-validation with fixed seeds. The robustness diagnostic is group-aware validation using defensible family or ticket groups; it remains a stress test unless experiments justify otherwise.

Report mean accuracy, accuracy standard deviation, balanced accuracy, precision, recall, F1, fold-level results, and a final confusion matrix. Kaggle accuracy is the competition metric.

## 6. Optimization

Use a controlled budget with one justified method: `RandomizedSearchCV`, a small Optuna study, or model-native search. Do not run thousands of trials or optimize against one split, one seed, or the public leaderboard.

## 7. Thresholds and ensembles

Select any non-default classification threshold from out-of-fold training predictions only. Evaluate a small weighted soft-voting ensemble only after identifying strong individual models. Do not implement stacking unless it shows a consistent leakage-safe advantage.

## 8. Feature engineering

Potential features include normalized title, surname, family size, is-alone, family-role indicators, ticket prefix, ticket group size, fare per group member, cabin deck, cabin-known indicator, number of cabins, age/fare missingness, age/fare bands, group identifiers, and justified sex/class/age/title/family-size interactions.

Treat every feature as a hypothesis. Use feature-group ablations. Row-independent transformations may run directly. Dataset-level counts must be fitted within folds, and target-derived features require strict out-of-fold construction.

## 9. Repository scope

Keep the project notebook-first, compact, and proportional. Do not add APIs, user interfaces, deployment systems, databases, microservices, agent frameworks, tracking servers, or large configuration frameworks unless the objective changes with explicit approval.

## 10. Notebook style

Maintain exactly two final notebooks unless explicitly approved otherwise. Use logical phases rather than many tiny cells or giant unrelated cells. Final notebooks must render on GitHub, save aggregate visualizations, avoid installation output and personal paths, avoid excessive raw tables, explain important results, and never expose hidden or leaked labels. Prefer about eight executable cells or fewer per notebook unless technically necessary.

## 11. Testing

Test meaningful behavior: title and surname extraction, ticket-prefix normalization, cabin-deck extraction, family size, is-alone, stable feature columns, submission schema, `PassengerId` order, and binary predictions. Do not add tests only to increase test count.

## 12. End-of-phase report

At each phase end, report files changed, research performed, commands executed, validation performed, unresolved issues, and decisions needing approval.
