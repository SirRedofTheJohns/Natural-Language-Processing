# Financial Complaint Triage System

Consumer complaint teams need to direct free-text narratives to the right
financial product queue without automatically routing uncertain cases. This
portfolio prototype classifies a narrative into one of 11 CFPB product
categories, then uses a fixed confidence threshold to choose between
`automatic_route` and `manual_review`.

The intended user is an operations reviewer who needs routing support, not an
automated financial decision. The notebooks are the primary portfolio
walkthrough; the API and Streamlit application remain optional technical
prototypes built around the same inference service.

## Project walkthrough

1. [Dataset audit and preparation](notebooks/01_dataset_audit.ipynb) covers data
   quality, efficient sampling, privacy-aware cleaning, leakage prevention, and
   the deterministic stratified split methodology.
2. [Model experiments and selection](notebooks/02_model_experiments.ipynb)
   compares the classical baseline with ModernBERT, explains validation-only
   threshold selection, and presents the held-out confidence-routing results.
3. The final sections of the model notebook interpret the potential operational
   value and document the benchmark's responsible-use limitations.

Together, these notebooks provide the main technical and decision-making story
without requiring the local dataset or model artifact.

## Key result

On the held-out benchmark, ModernBERT automatically routed approximately 61%
of complaints with approximately 95.9% accuracy on the accepted subset, while
sending the remaining cases to human review.

This is potential operational value under the benchmark conditions. It is not
evidence that a real institution has reduced its workload by 61%.

The score is the model's confidence, not a calibrated probability. A complaint
is automatically routed only when its unrounded confidence meets the fixed
0.90 threshold and the narrative contains enough text. All other cases are
marked for manual review.

## Model comparison

The selected product model is `answerdotai/ModernBERT-base` with a maximum
length of 512 tokens. Logistic Regression remains the classical baseline and a
useful lightweight deployment comparison.

| Held-out test result | TF-IDF + Logistic Regression | ModernBERT |
| --- | ---: | ---: |
| Accuracy | 0.8015 | 0.8222 |
| Macro F1 | 0.7837 | 0.8040 |
| Weighted F1 | 0.8001 | 0.8209 |
| Fixed routing threshold | 0.80 | 0.90 |
| Automation coverage | 42.6216% | 61.0320% |
| Accuracy on automatically routed cases | 0.9641 | 0.9587 |
| Macro F1 on automatically routed cases | 0.9390 | 0.9413 |

ModernBERT provides better overall quality and substantially higher automation
coverage. Logistic Regression is much smaller and faster. Any recorded
throughput or model-size measurements are specific to the Windows, Python,
CUDA, and RTX 4080 SUPER environment used for the experiments; they are not
universal guarantees.

## Optional technical prototype

The runtime implementation demonstrates how the selected model could be exposed
after the notebook-based experimental work. It is a portfolio prototype, not a
production-certified financial system.

### Architecture

```text
FastAPI or Streamlit
        |
        v
FinancialComplaintTriageService
  - shared text cleaning
  - batched tokenization and inference
  - confidence and top candidates
  - threshold-based routing
        |
        v
Local ModernBERT artifact
```

The service supports single predictions and genuine batched inference. Batch
requests are tokenized and evaluated in internal batches rather than by
repeatedly calling the single-prediction method. Public results contain only:

- predicted product;
- confidence score;
- `automatic_route` or `manual_review`;
- decision reason;
- active threshold;
- top candidates.

Complaint text, cleaned text, tokens, and local paths are not returned.

### Local setup

Python 3.12 is required.

```bash
python -m venv .venv
python -m pip install -r requirements.txt
python -m pip install -e . --no-deps
```

The trained model is intentionally excluded from Git. Provide the local
Hugging Face model and tokenizer artifact at:

```text
models/transformer/modernbert_base_512/
```

At minimum, that directory must contain the saved model configuration,
tokenizer files, and model weights. The repository does not include a download
script because the artifact is not publicly hosted.

### API

Start the API from the repository root:

```bash
uvicorn api.main:app --reload
```

The API exposes only:

- `GET /health` — process health without loading ModernBERT;
- `POST /predict` — one narrative, limited to 10,000 characters;
- `POST /predict/batch` — 1–16 narratives, each limited to 10,000 characters.

Single prediction with synthetic text:

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"text":"A synthetic cardholder reports that the same online purchase appears twice on the statement."}'
```

Batch prediction:

```bash
curl -X POST http://127.0.0.1:8000/predict/batch \
  -H "Content-Type: application/json" \
  -d '{"texts":["A synthetic borrower reports that an on-time mortgage payment is marked late.","A synthetic consumer found an unfamiliar account on a credit report and submitted a dispute."]}'
```

The service is created lazily and shared across requests. Validation and
service failures return safe messages without complaint narratives or local
filesystem paths.

### Streamlit demo

Run the focused single-page demo:

```bash
streamlit run app/streamlit_app.py
```

The page provides synthetic examples, a narrative text area, the predicted
product, confidence, routing decision, decision reason, active threshold, and
top candidates. The model is loaded only when classification is requested and
is cached as a Streamlit resource.

## Data and experimental methodology

The prepared benchmark contains 112,394 clean rows across 11 product classes,
covering 2023-09-01 through 2026-07-03:

- train: 78,675 rows;
- validation: 16,859 rows;
- test: 16,860 rows.

Narratives were cleaned and deduplicated before a deterministic stratified
random split. Only complaint narrative text was used as model input;
`issue` and `sub_issue` were not features. Models and confidence policies were
selected using validation macro F1 and validation routing analysis. The final
test set was opened once after model and threshold selection, and its saved
metrics must not be used for further tuning.

A temporal split was considered but rejected because one product class had
only 18 examples in the candidate 2026 test period. The stratified benchmark
supports comparison under a similar data distribution; it does not establish
robustness to future temporal drift or taxonomy changes.

## Testing

Run the fast suite without loading ModernBERT:

```bash
python -m pytest -m "not slow"
```

Fast tests use synthetic narratives and lightweight test doubles for inference
and API behavior. The API dependency is overridden so tests never instantiate
the real model.

One local smoke test is marked `slow`. It loads the saved artifact on CPU,
classifies one synthetic narrative, verifies the six-field public contract,
and confirms the expected 11 labels:

```bash
python -m pytest -m slow
```

GitHub Actions runs only the fast suite and does not download model artifacts,
datasets, or require CUDA.

## Limitations and responsible use

- Confidence scores are not calibrated probabilities.
- The 512-token maximum truncates longer narratives.
- The difficult `Debt or credit management` class remains a source of error.
- A stratified random split does not prove robustness to temporal drift.
- Historical and future taxonomy changes may reduce reliability.
- The system has not been evaluated inside a real financial institution.
- No fairness or subgroup analysis has been completed.
- Automatically routed cases can still be wrong; human review remains part of
  the design.
- This prototype supports complaint routing only. It must not be used for
  lending, eligibility, pricing, or other financial decisions.
- Complaint narratives may contain sensitive personal and financial data.
  Requests are not logged or persisted by this prototype.

## Repository structure

```text
api/main.py                              FastAPI application
app/streamlit_app.py                     Interactive demo
notebooks/01_dataset_audit.ipynb         Dataset audit and preparation
notebooks/02_model_experiments.ipynb     Model selection and recorded results
src/financial_complaint_triage/          Shared inference package
tests/test_inference.py                  Fast inference tests
tests/test_api.py                        Fast API tests
tests/test_model_smoke.py                One local slow model test
reports/metrics/final_test_results.json  Immutable aggregate test metrics
.github/workflows/tests.yml              Fast-only CI
```
