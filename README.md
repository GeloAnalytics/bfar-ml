# ML Flask Service (Propensity Score Matching)

A single Flask service (`app.py`, using shared logic in `psm_core.py`) grounded in a
frozen **bfar.csv baseline** (`models/best_model.pkl`, `models/scaler.pkl`,
`models/all_features.json`, `models/core_features.json`, `models/remaining_features.json`
-- produced by `build_model.py`). The service never retrains or overwrites that
baseline:

- **Records/CSV covers all 57 baseline features** -> scored directly against the
  frozen baseline model. No fitting happens.
- **Covers the 30 "core" baseline features but not all 57** -> the core 30 stay
  fixed, and up to `n_flex` (default 27) of the request's own top-ranked numeric
  columns fill the remaining slots. A throwaway model is fit **for that one request
  only** and discarded immediately after -- nothing is ever written back to disk or
  reused by a later request.
- **Missing a core feature** -> `400` error.

If this repo is dropped into another project as a subfolder (commonly named `ml/`), prefix the commands below with that folder (e.g. `pip install -r ml/requirements.txt`, `python ml/app.py`). All paths resolve relative to each file's own location, so it works the same whether this is the repo root or a nested subfolder -- `bfar.csv` just needs to stay next to `build_model.py`.

## Integration guide (backend & frontend)

**Architecture.** `app.py` is a plain, unauthenticated Flask HTTP API -- there's no
API key, session, or user concept anywhere in this repo. Treat this as an internal
ML layer that your **backend** calls server-to-server, not something a frontend talks
to directly in production. It binds `0.0.0.0` by default (LAN/container reachable)
purely so a backend running on a different machine/container can reach it; it is not
meant to be exposed to the public internet as-is.

```
frontend  -->  your backend  -->  app.py (HOST:PORT from .env, default 0.0.0.0:8000)
```

**Configuration (`.env`).** `app.py` loads a `.env` file on startup (via
`python-dotenv`) -- copy/edit the committed one or set real environment variables in
your deployment instead:
```bash
HOST=0.0.0.0
PORT=8000
# ML_MODEL_DIR=models   # only needed if artifacts live somewhere other than ./models
```

**Base URL.** Nothing in this repo reads this -- it's just the conventional env var
name to set in whatever backend consumes this service:

| Backend env var | Points at | Local default |
|---|---|---|
| `ML_SERVICE_URL` | `app.py` | `http://127.0.0.1:8000` (or your LAN/container address) |

**Error contract.** Every endpoint returns JSON with the same shape:
- `200` -- success, body is the endpoint-specific payload documented below.
- `400` -- bad/incomplete input (missing features, unparsable CSV, no treatment column found, dataset too small, ...): `{"error": "<message>"}`. The message is written to be shown to an end user or logged as a validation failure -- it's never a stack trace.
- `500` -- the service itself failed to load its baseline artifacts at startup: `{"error": "ML artifacts not loaded: <reason>"}`. This is an ops problem (bad `ML_MODEL_DIR`, missing `models/` files), not a user-input problem -- alert on it rather than surfacing it to end users.

There is no `401`/`403`/`404`/`409` currently -- if you need auth or
rate-limiting, add it in your backend's proxy layer, not here.

**CORS.** `app.py` calls `CORS(app)` with no origin restriction, so *any* origin can
call it directly from a browser as-is. That's fine for local development; if you ever
let a frontend call this service directly instead of going through your backend,
restrict this first (`CORS(app, origins=["https://your-frontend"])` in `app.py`) --
don't rely on network placement alone.

**Quick endpoint reference:**

| Method & path | Body | Returns |
|---|---|---|
| `GET /health` | -- | baseline status (model type, feature counts) |
| `POST /predict_ps` | JSON `{records}`, one object per row keyed by bfar column name -- partial feature sets OK | `{ps_final, used_baseline, final_features, treatment_column, ...}` |
| `POST /estimate_att` | JSON `{records}` with `treatment`+`outcome` per record | matched-ATT result + adaptation metadata |
| `POST /predict_ps_batch` | multipart CSV upload | per-row `ps` + decision-support quartile table |

Full request/response bodies and examples for each are in the sections below.

**Running in production.** `app.run(...)` is Flask's development server (it prints
its own "do not use in production" warning on startup) -- put a real WSGI server in
front for anything beyond local dev/integration testing, e.g.:
```bash
pip install waitress   # Windows-friendly; use gunicorn on Linux
waitress-serve --host=0.0.0.0 --port=8000 app:app
```

## 1) Install Python dependencies

```bash
pip install -r requirements.txt
```

`models/best_model.pkl`, `models/scaler.pkl`, `models/all_features.json`, `models/core_features.json`, and `models/remaining_features.json` are already committed, so no training step is required for normal use. To regenerate them after changing `bfar.csv`:
```bash
python build_model.py
```
This runs the same 5-fold cross-validated model selection as `updated_psm.ipynb` (Logistic Regression, Random Forest, Gradient Boosting, Neural Network -- picks the lowest-MSE model), fits it on the full dataset, and splits its feature importances into the 30 "core" features (always required for dynamic adaptation) and the 27 "remaining" features (informational baseline ranking only).

## 2) Start the service

```bash
python app.py
```
Reads `HOST`/`PORT` from `.env` (defaults `0.0.0.0:8000` if unset).

## 3) Endpoints

**How treatment detection works** (`psm_core.detect_treatment_column`): looks for a column literally named `treatment`; failing that, scores every column as either an already-binary flag (0/1, Yes/No, True/False) or a "populated only for one group" column (like `Y_BOAT-RE`, non-null only for program participants), favoring the latter since that's the far more common pattern in program/survey datasets, and breaking ties by earliest column position. This is a heuristic over a genuinely ambiguous problem -- always check `treatment_column`/`treatment_detection_method` in the response, and override with a `treatment_column` field/form-field if it's wrong. Only needed (and only attempted) when the request doesn't already cover all 57 baseline features.

### Health
```bash
curl http://localhost:8000/health
```
```json
{
  "status": "ok",
  "psm": {
    "source": "bfar.csv baseline (models/best_model.pkl) -- adapted per-request for uploads that don't cover all baseline features, never persisted",
    "model_type": "GradientBoostingClassifier",
    "n_core_features": 30,
    "n_remaining_features": 27,
    "n_all_features": 57
  }
}
```

### Predict propensity scores (JSON records)
```bash
curl -X POST http://localhost:8000/predict_ps \
  -H "Content-Type: application/json" \
  -d '{
    "records": [
      { "D1.1:A_BIKE": 0, "D1.1-A_QTY": 0, "...": 0 }
    ]
  }'
```
```json
{
  "ps_final": [0.42],
  "used_baseline": true,
  "n_features_used": 57,
  "final_features": ["D1.1:A_BIKE", "..."],
  "treatment_column": null,
  "treatment_detection_method": null
}
```
`used_baseline` is `false` and `treatment_column` is populated when the request didn't cover all 57 features and dynamic adaptation kicked in instead. Optional body fields: `treatment_column` (bypass auto-detection), `n_flex` (default: all 27 "remaining" features).

### Predict propensity scores + decision support (whole CSV)
```bash
curl -X POST http://localhost:8000/predict_ps_batch -F "file=@mydataset.csv"
```
Optional form fields: `treatment_column`, `n_flex`.

Mirrors `predictor_psm.ipynb`'s workflow directly: scores every row and returns a
decision-support table stratified by propensity-score quartile (Low / Med-Low /
Med-High / High), alongside the raw per-row scores:
```json
{
  "rows": 1339,
  "used_baseline": true,
  "n_features_used": 57,
  "treatment_column": "Y_BOAT-RE",
  "treatment_detection_method": "notna_mask",
  "ps": [0.42, "..."],
  "ps_logit": [-0.32, "..."],
  "ps_summary": { "min": 0.04, "max": 0.99, "mean": 0.45, "median": 0.42 },
  "decision_support": [
    { "ps_group": "Low", "count": 335, "mean_ps": 0.37, "interpretation": "Very low likelihood - may need targeted outreach", "...": "..." }
  ]
}
```

### Estimate ATT via matching (JSON records)
```bash
curl -X POST http://localhost:8000/estimate_att \
  -H "Content-Type: application/json" \
  -d '{
    "records": [
      { "D1.1:A_BIKE": 0, "D1.1-A_QTY": 0, "treatment": 1, "outcome": 12000 },
      { "D1.1:A_BIKE": 0, "D1.1-A_QTY": 0, "treatment": 0, "outcome": 9000 }
    ],
    "caliper_ratio": 0.2,
    "n_bootstrap": 200,
    "seed": 42
  }'
```
Records don't need to cover all 57 features -- the same core+flex dynamic adaptation
applies here, using each record's own `treatment`/`outcome` fields to both rank flex
features and compute the matched ATT (the `outcome` column itself is always excluded
from candidate features, so it can never leak into the model). Optional body fields:
`treatment_column`, `outcomeKey` (default `"outcome"`), `n_flex`, `caliper_ratio`,
`n_bootstrap`, `seed`.

## Notebooks

`updated_psm.ipynb` and `predictor_psm.ipynb` (not part of this repo, kept alongside it) contain the fuller research workflow this service's baseline is drawn from -- model selection with ROC/calibration plots, balance diagnostics, SHAP explainability, and IPW-based ATT estimation as a cross-check against the matching-based estimate served here. The live API intentionally exposes only the baseline-scoring and dynamic-adaptation pieces of that workflow; SHAP and IPW remain notebook-only analysis steps.
