# Financial Complaint Triage System — Codex Instructions

## Mission

Convert the completed NLP experiments into a small, professional,
production-oriented portfolio prototype.

The project classifies CFPB consumer complaint narratives into 11 financial
product categories and decides whether a complaint can be routed automatically
or should be sent to human review.

The final repository must demonstrate:

- efficient data preparation;
- leakage-safe machine-learning evaluation;
- comparison between classical NLP and a transformer;
- reusable model inference;
- confidence-based human review;
- a usable API and interactive demo;
- automated tests;
- honest documentation of trade-offs and limitations.

This is a portfolio prototype, not a production-certified financial system.

## Avoid overengineering

This is a focused portfolio project.

Do not create:

- microservices;
- model registries;
- plugin systems;
- abstract backend hierarchies;
- dependency-injection frameworks;
- multiple configuration packages;
- a custom CLI;
- Docker configuration;
- separate model-card and data-card documents;
- duplicate baseline and transformer production backends;
- unnecessary scripts or placeholder directories.

Do not create new files outside the target structure unless a concrete
technical need exists. Explain that need before creating them.

Prefer the smallest implementation that is clear, testable, and useful.

## Target repository structure

```text
.github/workflows/tests.yml
api/main.py
app/streamlit_app.py
notebooks/01_dataset_audit.ipynb
notebooks/02_model_experiments.ipynb
reports/metrics/final_test_results.json
src/financial_complaint_triage/__init__.py
src/financial_complaint_triage/inference.py
tests/test_inference.py
tests/test_api.py
tests/test_model_smoke.py
.gitignore
AGENTS.md
LICENSE
README.md
pyproject.toml
requirements.txt
```

Adapt existing files to this structure. Do not recreate files that already
serve the required purpose.

Do not create empty directories or placeholder files merely to match the target
tree.

## Validated experimental state

The experiment is complete.

Dataset:

- clean rows: 112,394;
- product classes: 11;
- train rows: 78,675;
- validation rows: 16,859;
- test rows: 16,860;
- selected date range: 2023-09-01 through 2026-07-03.

Classical baseline:

- TF-IDF + Logistic Regression;
- test accuracy: 0.8015;
- test macro F1: 0.7837;
- test weighted F1: 0.8001;
- fixed routing threshold: 0.80;
- test automation coverage: 42.6216%;
- accuracy on automatically routed cases: 0.9641;
- macro F1 on automatically routed cases: 0.9390.

Selected product model:

- `answerdotai/ModernBERT-base`;
- maximum length: 512 tokens;
- test accuracy: 0.8222;
- test macro F1: 0.8040;
- test weighted F1: 0.8209;
- fixed routing threshold: 0.90;
- test automation coverage: 61.0320%;
- accuracy on automatically routed cases: 0.9587;
- macro F1 on automatically routed cases: 0.9413.

Hardware used:

- Windows;
- Python 3.12;
- NVIDIA RTX 4080 SUPER;
- approximately 16 GB VRAM;
- CUDA and bfloat16 support.

Throughput and model-size measurements are specific to the local hardware and
software environment. Do not present them as universal guarantees.

## Machine-learning integrity

The final test set has already been opened.

Never:

- retrain models without explicit permission;
- repeat final test evaluation;
- tune thresholds using test results;
- modify final metrics;
- use test results for model, feature, or hyperparameter selection;
- use `issue` or `sub_issue` as model inputs;
- rescan the full raw dataset unnecessarily;
- describe confidence scores as calibrated probabilities;
- fabricate experiment or notebook outputs;
- claim validation inside a real financial institution.

Use saved metrics and artifacts instead of rerunning expensive experiments.

The attempted temporal split was rejected because one product class had only
18 examples in the candidate 2026 test period.

The final benchmark uses a deterministic stratified random split.

Document clearly that this benchmark:

- enables comparison under a similar data distribution;
- does not prove robustness to future temporal drift;
- does not prove robustness to future taxonomy changes.

## Product behavior

The product prototype uses ModernBERT as its inference model.

Logistic Regression remains documented as:

- the classical baseline;
- a lightweight deployment comparison;
- evidence of the quality, latency, and size trade-off.

Do not build a second production backend unless explicitly requested later.

The existing `FinancialComplaintTriageService` is the single source of
preprocessing and inference logic.

The API and Streamlit application must reuse this service and must not duplicate:

- text cleaning;
- tokenization;
- model inference;
- confidence calculation;
- routing decisions;
- top-candidate generation.

The service must support:

- `predict_one`;
- true batched `predict_batch`;
- configurable confidence threshold;
- CUDA with bfloat16 when supported;
- CPU fallback;
- project-relative model paths;
- clear errors when model artifacts are missing.

`predict_batch` must tokenize and infer over batches.

It must not call `predict_one` repeatedly.

Each prediction result must contain:

- predicted product;
- confidence score;
- routing decision;
- decision reason;
- threshold;
- top candidates.

Use the terms:

- `automatic_route`;
- `manual_review`.

Do not describe the score as a calibrated probability.

## API scope

Implement only:

- `GET /health`;
- `POST /predict`;
- `POST /predict/batch`.

Requirements:

- FastAPI and Pydantic schemas;
- one shared service instance;
- dependency override support for tests;
- maximum text length;
- maximum batch size;
- safe and understandable errors;
- no raw complaint text in logs;
- no personal local paths in responses;
- no model loading per request.

Do not add:

- authentication;
- databases;
- queues;
- background workers;
- monitoring platforms;
- cloud deployment infrastructure;
- user management;
- persistent request history.

## Streamlit scope

The demo must support:

- entering one complaint narrative;
- selecting a synthetic example;
- showing the predicted product;
- showing the confidence score;
- showing automatic route or manual review;
- showing the decision reason;
- showing the top candidates;
- showing the active threshold;
- showing a short limitations notice.

Do not include real complaint narratives as examples.

Do not add:

- dashboards;
- authentication;
- stored prediction history;
- user accounts;
- analytics;
- batch-file uploads;
- administrative panels.

The demo should be focused and understandable in a few minutes.

## Notebook rules

The public repository should contain two notebooks:

1. `01_dataset_audit.ipynb`
2. `02_model_experiments.ipynb`

The second notebook may contain the baseline and transformer experiments
together because they form one model-selection story.

Target 4–8 executable cells per notebook.

Do not exceed 10 executable cells without a clear technical reason.

Never create a huge unreadable cell merely to reduce the cell count.

Each code cell must represent one complete logical phase.

### Dataset notebook phases

1. configuration and efficient loading;
2. consolidated audit;
3. period and sample selection;
4. cleaning, deduplication, and export;
5. final decisions and dataset summary.

### Model notebook phases

1. load prepared splits;
2. classical baseline comparison;
3. confidence-routing analysis;
4. transformer configuration and recorded training results;
5. final model comparison and held-out test results;
6. conclusions and limitations.

Remove:

- repeated diagnostics;
- debugging cells;
- duplicate imports;
- failed temporary cells with no explanatory value;
- personal absolute paths;
- raw complaint previews;
- repeated outputs;
- checks that belong in automated tests.

Preserve:

- decision-relevant tables;
- validated final metrics;
- methodology explanations;
- model-selection reasoning;
- confidence-routing results;
- limitations;
- evidence of the baseline-versus-transformer trade-off.

Never rerun expensive training merely to regenerate notebook outputs.

Never fabricate notebook outputs.

Use notebook-aware editing and verify that modified notebooks remain valid.

When a saved metric can replace an expensive recomputation, load and present the
saved metric clearly.

## Data and privacy

Do not commit:

- raw data;
- processed datasets;
- model weights;
- checkpoints;
- Hugging Face caches;
- prediction-level Parquet files;
- secrets;
- personal absolute paths;
- environment-specific temporary files.

Do not delete local ignored data or model artifacts.

Ignoring an artifact in Git is not permission to remove it.

Complaint narratives may contain sensitive financial or personal information.

Do not:

- log complete narratives;
- persist API requests by default;
- echo full narratives in errors;
- use real complaints in tests;
- use real complaints as demo examples;
- include real complaints in screenshots;
- expose local filesystem paths.

Use synthetic complaint text in public examples and tests.

## Testing scope

Keep testing focused and meaningful.

Create only:

- `tests/test_inference.py`
- `tests/test_api.py`
- `tests/test_model_smoke.py`

Do not create separate unit, integration, fixtures, helpers, or test-utility
packages unless a demonstrated need exists.

### `tests/test_inference.py`

Test:

- text cleaning;
- empty input;
- insufficient-text handling;
- invalid confidence threshold;
- routing above the threshold;
- routing below the threshold;
- top-k candidate ordering;
- batch result count;
- true batch behavior using a lightweight fake model or injected test double.

Do not load ModernBERT in these fast tests.

### `tests/test_api.py`

Test:

- health endpoint;
- valid single prediction;
- valid batch prediction;
- empty input;
- text-length limit;
- batch-size limit;
- safe error responses.

API tests must use a test double and must not load ModernBERT.

### `tests/test_model_smoke.py`

Create one locally marked slow smoke test that:

- loads the saved ModernBERT artifact;
- runs one synthetic complaint;
- confirms the output contract;
- confirms the model has the expected 11 labels.

Do not create dozens of minor tests.

GitHub Actions must run only fast tests.

GitHub Actions must not:

- download ModernBERT;
- download the full dataset;
- require CUDA;
- run the slow model smoke test.

Never claim a test passed unless it was actually executed.

## Dependencies

Inspect the existing environment before changing dependencies.

A minimal `pyproject.toml` may be added for:

- editable installation;
- package metadata;
- pytest configuration;
- test markers.

Keep `requirements.txt` usable.

Expected dependencies include only what the project actually uses, such as:

- pandas;
- numpy;
- scikit-learn;
- torch;
- transformers;
- datasets;
- accelerate;
- pyarrow;
- joblib;
- fastapi;
- uvicorn;
- streamlit;
- pydantic;
- pytest;
- httpx;
- nbformat.

Do not add a dependency solely for a minor convenience that standard Python can
handle.

Do not remove existing required dependencies without confirming that the
project still works.

## README goals

The README must be written for recruiters and technical reviewers.

Recommended order:

1. problem and intended user;
2. system behavior;
3. key result;
4. demo and quick start;
5. model comparison;
6. architecture;
7. API usage;
8. data and experimental methodology;
9. testing;
10. limitations and responsible use;
11. project structure.

The README should begin with the real operational use case, not with a long
machine-learning theory section.

Make the principal claim accurately:

> On the held-out benchmark, ModernBERT automatically routed approximately 61%
> of complaints with approximately 95.9% accuracy on the accepted subset, while
> sending the remaining cases to human review.

Do not claim that a real organization already reduced its workload by 61%.

Explain that the result represents potential operational value under the
benchmark conditions.

Clearly communicate the trade-off:

- ModernBERT provides better quality and higher automation coverage;
- Logistic Regression is much smaller and faster.

Document limitations honestly, especially:

- uncalibrated confidence scores;
- possible temporal drift;
- historical taxonomy changes;
- truncation at 512 tokens;
- the difficult `Debt or credit management` class;
- no evaluation inside a real financial institution;
- no fairness or subgroup analysis;
- the system is for routing support, not financial decision-making.

## GitHub and artifact rules

The public repository must not contain:

- the 8.53 GB source dataset;
- compressed copies of the source dataset;
- processed Parquet files;
- ModernBERT weights;
- training checkpoints;
- Hugging Face caches;
- prediction-level records;
- secrets;
- personal paths.

Do not create download scripts for artifacts that are not actually hosted.

Document clearly where local artifacts are expected and how a user can
reproduce or provide them.

Do not upload anything to GitHub, Hugging Face, or another external service.

Do not commit or push.

## Working method

Before broad edits:

1. inspect the repository;
2. identify the exact files to change;
3. present a concise plan;
4. wait for approval.

After an approved implementation phase:

1. run the relevant tests;
2. inspect the complete diff;
3. fix confirmed problems;
4. report the commands actually executed;
5. report pass or fail results honestly;
6. stop before the next phase.

Do not perform the entire project refactor in one uncontrolled change set.

Do not:

- commit;
- push;
- publish;
- upload;
- train;
- repeat final evaluation;
- delete artifacts.

## Approval boundaries

Pause before:

- destructive operations;
- deleting or moving local artifacts;
- retraining a model;
- modifying final test outputs;
- downloading large files;
- adding dependencies outside the expected list;
- changing the license;
- creating files outside the target structure;
- rewriting Git history;
- committing or pushing.

Routine edits inside an approved phase do not require repeated confirmation.

## Definition of done

The project is complete when:

- the inference service works without duplicated logic;
- true batch inference is implemented;
- FastAPI endpoints work;
- the Streamlit demo works;
- fast tests pass;
- the local model smoke test passes when the artifact is available;
- notebooks are concise and valid;
- README accurately explains the real use case and results;
- GitHub Actions runs the fast tests;
- no sensitive or large artifacts are staged;
- no personal absolute paths remain;
- no final metrics were changed;
- the final diff contains no unnecessary architecture.

## Completion report

At the end of each approved phase, report only:

- files created or modified;
- commands actually executed;
- tests and their results;
- unresolved blockers or limitations;
- decisions requiring approval.

Keep the report compact and factual.
