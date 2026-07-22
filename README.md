# ML Flask Service (Propensity Score Matching)

A single file, `app.py` (shared logic in `psm_core.py`), running two Flask apps as two
independent servers on two ports -- the dynamic app runs in the main thread, the static
app in a background thread, started together by `python app.py`:

| | Static | Dynamic |
|---|---|---|
| **Port** | `STATIC_PORT` (default `8001`) | `PORT` (default `8000`) |
| **Trigger** | Every request | Request covers all 57 raw bfar.csv columns falls back to baseline; anything else uses the trained dynamic model |
| **Model** | `models/best_model.pkl` -- frozen, produced by `build_model.py` | Baseline (see left) or whatever `POST /train` last produced |
| **`/train` endpoint?** | No -- baseline-only, rejects requests missing any of the 57 features (`409`) | Yes |
| **Retrained on upload?** | Never | Every `/train` call, unconditionally |
| **Persisted?** | Yes, committed to the repo | Yes, `models/dynamic/` (gitignored runtime state) |

The dynamic model works **Teachable-Machine style**: every `POST /train` call deletes
whatever model is currently active and trains a completely fresh one on the new
upload -- ranking every usable column by feature importance and keeping the top 30, no
merging with the previous schema, no reuse-shortcut. See `DYNAMIC_TRAINING.md` for the
full design and why this replaced an earlier index-mapping approach.

Both always start together (one process, `python app.py`) -- there's no flag to run
just one, but each is an independent Flask server on its own port, so callers only
ever need to know about whichever one they integrate against.

If this repo is dropped into another project as a subfolder (commonly named `ml/`), prefix the commands below with that folder (e.g. `pip install -r ml/requirements.txt`, `python ml/app.py`). All paths resolve relative to each file's own location -- `bfar.csv` just needs to stay next to `build_model.py`.

## Integration guide (backend & frontend)

**Architecture.** `app.py` is a plain, unauthenticated Flask HTTP API -- no API key,
session, or user concept. Treat it as an internal ML layer your **backend** calls
server-to-server, not something a frontend talks to directly in production. It binds
`0.0.0.0` by default (LAN/container reachable); it is not meant to be exposed to the
public internet as-is.

```
                    -->  app.py, dynamic app   (HOST:PORT from .env, default 0.0.0.0:8000)
frontend  -->  your backend
                    -->  app.py, static app     (STATIC_HOST:STATIC_PORT, default 0.0.0.0:8001)
```

**Configuration (`.env`).** Loaded on startup via `python-dotenv`:
```bash
HOST=0.0.0.0
PORT=8000
STATIC_HOST=0.0.0.0
STATIC_PORT=8001
# ML_MODEL_DIR=models             # only if baseline artifacts live somewhere other than ./models
# ML_DYNAMIC_STATE_DIR=models/dynamic  # only if the trained dynamic model should live elsewhere
```

**Error contract.** Every endpoint returns JSON with the same shape:
- `200` -- success, body is the endpoint-specific payload documented below.
- `400` -- bad/incomplete input (unparsable CSV, no treatment column found, dataset too small, missing required feature columns, ...): `{"error": "<message>"}` -- shown-to-user quality, never a stack trace.
- `409` -- a scoring endpoint was called but no model applies: the request doesn't cover all 57 baseline features, and nothing has been trained yet. `{"error": "no dynamic model trained yet, ..."}`.
- `500` -- baseline artifacts failed to load at startup: `{"error": "ML artifacts not loaded: <reason>"}`. Ops problem, not user-input problem -- alert on it.

**CORS.** `CORS(app)` currently allows any origin -- fine for local development; if a
frontend ever calls this service directly, restrict it first
(`CORS(app, origins=["https://your-frontend"])`).

**Quick endpoint reference** (dynamic service, `app.py`, port `8000`):

| Method & path | Body | Returns |
|---|---|---|
| `GET /health` | -- | baseline status + current dynamic model status |
| `POST /train` | multipart CSV (`file`, `treatment_column?`) | trained model summary; **replaces** whatever was previously active |
| `POST /predict_ps` | JSON `{records}` | `{ps_final, source, n_features_used}` |
| `POST /estimate_att` | JSON `{records}` with `treatment`+`outcome` per record | matched-ATT result |
| `POST /predict_ps_batch` | multipart CSV (`file`) | per-row `ps` + decision-support quartile table |

The static app (port `8001`, same `app.py` process) exposes the same `GET /health`,
`POST /predict_ps`, `POST /estimate_att`, and `POST /predict_ps_batch` -- no `/train`.
Every request must cover all 57 baseline features or it gets a `409`; `source` in the
response is always `"baseline"`.

**Response fields:** all three scoring endpoints report `source` (`"baseline"` or
`"dynamic"`, telling you which model actually served the request) and
`n_features_used`.

**Running in production.** `app.run(...)` is Flask's dev server -- put a real WSGI
server in front. `app.py` exposes two module-level Flask objects, `app` (dynamic) and
`static_app` (static), so a WSGI server can target either directly instead of going
through the `if __name__ == "__main__"` thread-starting block:
```bash
pip install waitress   # Windows-friendly; use gunicorn on Linux
waitress-serve --host=0.0.0.0 --port=8000 app:app
waitress-serve --host=0.0.0.0 --port=8001 app:static_app
```
Note: the dynamic model is in-process state mirrored to disk. If you scale to
multiple workers, they won't share a freshly `/train`-ed model until each has
independently loaded it from `ML_DYNAMIC_STATE_DIR` on its own startup.

## 1) Install Python dependencies

```bash
pip install -r requirements.txt
```

The baseline artifacts are committed (`models/best_model.pkl`, `scaler.pkl`,
`all_features.json`, `core_features.json`, `remaining_features.json`), so no training
step is required for normal use. To regenerate after changing `bfar.csv`:
```bash
python build_model.py
```
This runs 5-fold cross-validated model selection (Logistic Regression, Random Forest,
Gradient Boosting, Neural Network -- lowest MSE wins) on the raw 57 features and saves
the winner as the frozen baseline.

## 2) Start the service

```bash
python app.py
```
Starts both servers in one process: the dynamic app on `HOST`/`PORT` (default
`0.0.0.0:8000`, main thread) and the static baseline-only app on `STATIC_HOST`/
`STATIC_PORT` (default `0.0.0.0:8001`, background thread).

## 3) Endpoints

**How treatment detection works** (`psm_core.detect_treatment_column`, used by `/train`): looks for a column literally named `treatment`; failing that, scores every column as either an already-binary flag (0/1, Yes/No, True/False) or a "populated only for one group" column (like `Y_BOAT-RE`, non-null only for program participants), favoring the latter, breaking ties by earliest column position. A heuristic over a genuinely ambiguous problem -- always check `treatment_column`/`treatment_detection_method` in the response, and override with the `treatment_column` form field if it's wrong.

### Health
```bash
curl http://localhost:8000/health
```
```json
{
  "status": "ok",
  "baseline": {
    "source": "bfar.csv (baked-in, static, never retrained)",
    "model_type": "GradientBoostingClassifier",
    "n_features_total": 57,
    "top_features": [{"feature": "E1:A_DRINK-H2O", "importance": 0.084}, "..."]
  },
  "dynamic": { "status": "empty", "message": "no dataset trained yet; POST a CSV to /train" }
}
```
After a `/train` call, `dynamic` instead looks like:
```json
{
  "status": "ok",
  "source_filename": "mydataset.csv",
  "rows": 412,
  "trained_at": 1752566400.0,
  "treatment_column": "enrolled",
  "treatment_detection_method": "binary_value",
  "n_features_selected": 30,
  "top_features": [{"feature": "monthly_income", "importance": 0.11}, "..."]
}
```

### Train the dynamic model
```bash
curl -X POST http://localhost:8000/train -F "file=@mydataset.csv"
```
Optional form field: `treatment_column=enrolled_flag` (bypasses auto-detection).

**Deletes whatever dynamic model is currently active and trains a completely fresh
one** -- ranks every usable numeric column by importance for predicting the detected
treatment column (excluding near-perfect treatment proxies), keeps the top 30, fits a
fresh model. Nothing carries over from any previous `/train` call.
```json
{
  "status": "trained",
  "rows": 412,
  "treatment_column": "enrolled",
  "treatment_detection_method": "binary_value",
  "n_features_selected": 30,
  "top_features": [{"feature": "monthly_income", "importance": 0.11}, "..."],
  "excluded_as_leakage": ["group_assignment_code"]
}
```

### Predict propensity scores (JSON records)
```bash
curl -X POST http://localhost:8000/predict_ps \
  -H "Content-Type: application/json" \
  -d '{ "records": [ { "monthly_income": 8000, "household_size": 4, "...": 0 } ] }'
```
```json
{ "ps_final": [0.42], "source": "dynamic", "n_features_used": 30 }
```
Scores against the frozen baseline (`source: "baseline"`) if every record covers all
57 raw bfar features; otherwise against whatever's currently in the dynamic model
(`source: "dynamic"`). `409` if neither applies -- train first, or include all 57
baseline features.

### Predict propensity scores + decision support (whole CSV)
```bash
curl -X POST http://localhost:8000/predict_ps_batch -F "file=@mydataset.csv"
```
Scores every row and adds a decision-support table stratified by propensity-score
quartile (Low / Med-Low / Med-High / High) with per-group interpretations, plus
`ps_summary` (min/max/mean/median) -- mirrors `predictor_psm.ipynb`'s
predict-and-support workflow.

### Estimate ATT via matching (JSON records)
```bash
curl -X POST http://localhost:8000/estimate_att \
  -H "Content-Type: application/json" \
  -d '{
    "records": [
      { "monthly_income": 8000, "...": 0, "treatment": 1, "outcome": 12000 },
      { "monthly_income": 7500, "...": 0, "treatment": 0, "outcome": 9000 }
    ],
    "caliper_ratio": 0.2, "n_bootstrap": 200, "seed": 42
  }'
```
Returns `matched_pairs`, `att_mean`, `ci_95`, `p_value_paired_ttest`, `caliper`, plus
`source`/`n_features_used`. The `outcome` column is never treated as a candidate
feature. Optional body fields: `treatmentKey` (default `"treatment"`), `outcomeKey`
(default `"outcome"`), `caliper_ratio`, `n_bootstrap`, `seed`.

## Notebooks

`updated_psm.ipynb` and `predictor_psm.ipynb` (not part of this repo, kept alongside it) contain the fuller research workflow the baseline model is drawn from -- model selection with ROC/calibration plots, balance diagnostics, SHAP explainability, and IPW-based ATT estimation as a cross-check against the matching-based estimate served here. The live API intentionally exposes only baseline-scoring and dynamic training/scoring; SHAP and IPW remain notebook-only analysis steps.
