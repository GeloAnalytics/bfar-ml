# ML Flask Service (Propensity Score Matching)

Two separate Flask services share the same PSM logic (`psm_core.py`) and the same
frozen **bfar.csv baseline** (`models/best_model.pkl`, `models/scaler.pkl`,
`models/all_features.json`, `models/core_features.json`, `models/remaining_features.json`
-- produced by `build_model.py`). Neither service ever retrains or overwrites that
baseline:

| Service | File | Default bind | Purpose |
|---|---|---|---|
| Static reference | `app.py` | `127.0.0.1:8000` (loopback only) | Always scores against the baseline directly. Known-good reference to check against; not reachable off-box. |
| Dynamic | `app_dynamic.py` | `0.0.0.0:8001` (reachable via the machine's LAN address, e.g. `192.168.x.x`) | Accepts arbitrary uploaded data (JSON records or a whole CSV). If it covers all 57 baseline features, scores it against the baseline directly -- no fitting happens. Otherwise it keeps the 30 "core" baseline features fixed and dynamically fills in extra features from that dataset's own columns, fitting a throwaway model **for that one request only**. Nothing is ever written back to disk or reused by a later request -- every request re-anchors to the same baseline. |

If this repo is dropped into another project as a subfolder (commonly named `ml/`), prefix the commands below with that folder (e.g. `pip install -r ml/requirements.txt`, `python ml/app.py`). All paths resolve relative to each file's own location, so it works the same whether this is the repo root or a nested subfolder -- `bfar.csv` just needs to stay next to `build_model.py`.

## 1) Install Python dependencies

```bash
pip install -r requirements.txt
```

`models/best_model.pkl`, `models/scaler.pkl`, `models/all_features.json`, `models/core_features.json`, and `models/remaining_features.json` are already committed, so no training step is required for normal use. To regenerate them after changing `bfar.csv`:
```bash
python build_model.py
```
This runs the same 5-fold cross-validated model selection as `updated_psm.ipynb` (Logistic Regression, Random Forest, Gradient Boosting, Neural Network -- picks the lowest-MSE model), fits it on the full dataset, and splits its feature importances into the 30 "core" features (always required by the dynamic service) and the 27 "remaining" features (informational baseline ranking only).

## 2) Start the services

```bash
python app.py            # static reference, http://127.0.0.1:8000
python app_dynamic.py     # dynamic, http://0.0.0.0:8001 (also reachable at your LAN IP)
```

Ports are configurable via `STATIC_PORT` and `DYNAMIC_PORT` env vars.

## 3) Static service (`app.py`) -- bfar.csv baseline, always

### Health
```bash
curl http://localhost:8000/health
```
Returns `psm.top_features`: the 30 highest-importance features out of the model's fixed 57, read directly off the already-trained baseline model -- no retraining involved.

### Predict propensity scores
```bash
curl -X POST http://localhost:8000/predict_ps \
  -H "Content-Type: application/json" \
  -d '{
    "records": [
      { "D1.1:A_BIKE": 0, "D1.1-A_QTY": 0, "...": 0 }
    ]
  }'
```

### Estimate ATT via matching
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

**Note:** Every record must include *all* 57 baseline feature keys found in `models/all_features.json`, plus:
- `treatment`: 0 or 1
- `outcome`: numeric outcome value (in the notebooks it was `C5:TOT_INCOME/B`)

`app.py` never adapts to partial feature sets -- that's what the dynamic service is for.

## 4) Dynamic service (`app_dynamic.py`) -- arbitrary datasets, no persistence

Every request is independent and stateless. There's no `/train` step and nothing is
saved to disk -- each call re-anchors to the same `models/` baseline artifacts loaded
once at startup:

- **Upload covers all 57 baseline features** -> scored directly against the frozen
  baseline model. No fitting happens.
- **Upload covers the 30 core features but not all 57** -> the core 30 stay fixed,
  and up to `n_flex` (default 27) of the dataset's own top-ranked numeric columns
  fill the remaining slots. A throwaway model is fit **for this request only** and
  discarded immediately after -- it never affects any other request. This path
  requires a treatment/control column (to rank the extra features against) and at
  least 10 rows.
- **Missing a core feature** -> `400` error.

**How treatment detection works** (`psm_core.detect_treatment_column`): looks for a column literally named `treatment`; failing that, scores every column as either an already-binary flag (0/1, Yes/No, True/False) or a "populated only for one group" column (like `Y_BOAT-RE`, non-null only for program participants), favoring the latter since that's the far more common pattern in program/survey datasets, and breaking ties by earliest column position. This is a heuristic over a genuinely ambiguous problem -- always check `treatment_column`/`treatment_detection_method` in the response, and override with a `treatment_column` field/form-field if it's wrong. Only needed (and only attempted) when the upload doesn't already cover all 57 baseline features.

### Predict propensity scores (JSON records)
```bash
curl -X POST http://localhost:8001/predict_ps \
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
`used_baseline` is `false` and `treatment_column` is populated when the upload didn't cover all 57 features and dynamic adaptation kicked in instead.

### Predict propensity scores + decision support (whole CSV)
```bash
curl -X POST http://localhost:8001/predict_ps_batch -F "file=@mydataset.csv"
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
Same request/response shape as the static service's `/estimate_att` (see section 3), but records don't need to cover all 57 features -- the dynamic core+flex adaptation applies here too, using each record's own `treatment`/`outcome` fields to both rank flex features and compute the matched ATT.

### Health
```bash
curl http://localhost:8001/health
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

## Notebooks

`updated_psm.ipynb` and `predictor_psm.ipynb` (not part of this repo, kept alongside it) contain the fuller research workflow this service's baseline is drawn from -- model selection with ROC/calibration plots, balance diagnostics, SHAP explainability, and IPW-based ATT estimation as a cross-check against the matching-based estimate served here. The live API intentionally exposes only the baseline-scoring and dynamic-adaptation pieces of that workflow; SHAP and IPW remain notebook-only analysis steps.
