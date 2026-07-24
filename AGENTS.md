# Financial Complaint Triage System — Agent Instructions

## Project goal

Build a production-quality portfolio project that classifies CFPB consumer
complaint narratives into 11 financial product categories and decides whether
a prediction can be routed automatically or requires human review.

The repository must demonstrate:

- rigorous NLP and machine-learning experimentation;
- comparison between a classical baseline and a transformer;
- leakage-safe evaluation;
- reusable inference code;
- an API and interactive demo;
- automated testing;
- clear documentation and reproducibility.

## Current validated results

The experimentation phase is complete.

Selected transformer:
- Model: answerdotai/ModernBERT-base
- Maximum length: 512 tokens
- Precision: bfloat16 on supported CUDA hardware
- Final test accuracy: 0.8222
- Final test macro F1: 0.8040
- Operational confidence threshold: 0.90
- Automatic routing coverage on test: 61.0320%
- Accuracy on automatically routed test cases: 0.9587

Classical baseline:
- TF-IDF + Logistic Regression
- Final test accuracy: 0.8015
- Final test macro F1: 0.7837
- Baseline operational threshold: 0.80

Do not alter, tune, or reinterpret these final test results.

## Hardware and environment

Development environment:
- Native Windows
- Python 3.12
- NVIDIA GeForce RTX 4080 SUPER
- Approximately 16 GB VRAM
- CUDA-enabled PyTorch
- bfloat16 supported

Use CUDA when available and provide a CPU fallback for inference.

Do not hard-code the developer's absolute Windows user path. Use pathlib and
project-relative paths.

## Critical machine-learning constraints

- The final test set has already been opened and evaluated.
- Never use test results for further tuning, model selection, threshold
  selection, feature selection, or hyperparameter changes.
- Do not retrain ModernBERT or repeat expensive experiments unless explicitly
  requested.
- Do not rescan the full 8.53 GB raw CSV when clean Parquet artifacts already
  contain the required data.
- Do not use Issue or Sub-issue as model input because they create target
  leakage.
- Preserve the existing saved models, reports, predictions, and metrics.
- Refer to model outputs as confidence scores unless probability calibration
  is explicitly implemented and evaluated.

## Notebook rules

Notebooks are concise experiment reports, not application modules or test
suites.

- Consolidate related diagnostics into one meaningful cell per phase.
- Aim for no more than 5–7 executable cells per notebook unless technically
  unavoidable.
- Each cell must perform a complete logical phase.
- Print only information required to understand results or make a decision.
- Remove repeated debugging cells, redundant imports, exploratory fragments,
  and duplicate outputs.
- Preserve important experimental results and explanations.
- Do not rerun expensive training merely to regenerate notebook outputs.
- Use markdown to explain decisions instead of creating extra code cells.
- Move reusable functions to src/.
- Move repeatable validation to tests/ or scripts/.

## Code architecture

Prefer this separation:

- src/financial_complaint_triage/: reusable application logic
- api/: API entry points
- app/: interactive demo
- tests/: automated tests
- scripts/: repository verification and utility scripts
- notebooks/: concise experimentation reports
- reports/: metrics, predictions, and generated reports
- models/: trained model artifacts
- data/: raw, interim, and processed data

Reuse FinancialComplaintTriageService from
src/financial_complaint_triage/inference.py. Do not duplicate inference logic
inside the API, app, notebooks, or tests.

## Engineering rules

- Inspect existing code before modifying it.
- Present a concise plan before broad repository changes.
- Do not silently delete files or experiment results.
- Ask before adding new production dependencies.
- Prefer small, focused modules with type annotations and clear error handling.
- Use consistent project-relative configuration.
- Avoid premature abstractions and unnecessary frameworks.
- Do not create placeholder functionality presented as complete.
- Preserve backward compatibility when reasonable.
- Keep user-facing output concise and understandable.

## Verification strategy

Testing must be rigorous but efficient.

Use automated tests rather than adding notebook cells.

At minimum verify:

- package imports;
- text cleaning;
- empty and insufficient text handling;
- confidence-threshold routing;
- top-k predictions;
- invalid model paths and invalid configuration;
- model label mappings;
- CPU fallback;
- saved model loading;
- one transformer inference smoke test;
- baseline inference smoke test;
- API request and response schema;
- application import/startup smoke test;
- processed dataset schema;
- absence of ID or normalized-text overlap between stored splits;
- notebook validity and compact cell counts.

Avoid loading the transformer repeatedly across separate tests. Use
session-scoped fixtures or an equivalent efficient strategy.

Create one repository verification command or script that executes the
appropriate checks and prints a compact pass/fail summary.

## Definition of done

Work is complete only when:

1. Existing notebooks are compact, readable, and reproducible without
   unnecessary cells.
2. Reusable logic lives in src/.
3. The inference service works with the saved ModernBERT model.
4. The API and interactive application use the same inference service.
5. Automated tests cover critical behavior.
6. The verification command completes successfully.
7. README.md accurately explains setup, architecture, results, limitations,
   commands, and demo usage.
8. requirements or project dependency configuration matches the actual code.
9. .gitignore protects datasets, caches, temporary checkpoints, virtual
   environments, and generated artifacts as appropriate.
10. No final test data was used for additional tuning.
11. A final diff review finds no obvious regressions, duplicated logic,
    misleading claims, or broken paths.

## Final response format

At the end of a task, report only:

- files created or modified;
- tests and commands executed;
- pass/fail results;
- remaining limitations or blockers;
- any action requiring the user's approval.