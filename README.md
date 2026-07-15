# ML Flask Service (Propensity Score Matching)

Two separate Flask services share the same PSM logic (`psm_core.py`):

| Service | File | Default bind | Purpose |
|---|---|---|---|
| Static reference | `app.py` | `127.0.0.1:8000` (loopback only) | Serves the fixed, pre-trained model baked from `bfar.csv`. Known-good baseline to check against; not reachable off-box. |
| Dynamic | `app_dynamic.py` | `0.0.0.0:8001` (reachable via the machine's LAN address, e.g. `192.168.x.x`) | Accepts an arbitrary CSV upload, auto-detects the treatment column, picks the top 30 predictive features, trains a fresh model in memory, and serves predictions against it. |

If this repo is dropped into another project as a subfolder (commonly named `ml/`), prefix the commands below with that folder (e.g. `pip install -r ml/requirements.txt`, `python ml/app.py`). All paths resolve relative to each file's own location, so it works the same whether this is the repo root or a nested subfolder — `bfar.csv` just needs to stay next to `build_model.py`/`train_model.py`.

## 1) Install Python dependencies

```bash
pip install -r requirements.txt
```

`models/gradient_boosting_ps_model.pkl` and `models/pre_features.json` (used by the static service) are already committed, so no training step is required for normal use. To regenerate them after changing `bfar.csv` or the feature list:
```bash
pip install -r requirements-dev.txt
python build_model.py
```

## 2) Start the services

```bash
python app.py            # static reference, http://127.0.0.1:8000
python app_dynamic.py     # dynamic upload/train, http://0.0.0.0:8001 (also reachable at your LAN IP)
```

Ports are configurable via `STATIC_PORT` and `DYNAMIC_PORT` env vars.

## 3) Static service (`app.py`) — bfar.csv reference

### Health
```bash
curl http://localhost:8000/health
```
Returns `psm.top_features`: the 30 highest-importance features out of the model's fixed 57, read directly off the already-trained model — no retraining involved.

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

**Note:** Every record must include *all* `pre_features` keys found in `models/pre_features.json`, plus:
- `treatment`: 0 or 1
- `outcome`: numeric outcome value (in the notebook it was `C5:TOT_INCOME/B`)

## 4) Dynamic service (`app_dynamic.py`) — arbitrary datasets

The active model + feature schema is persisted to `models/dynamic/` (gitignored — it's runtime state, not a build artifact) and reloaded on startup, so a process restart doesn't lose it.

**`/train` does not necessarily retrain.** Uploading a CSV that already covers at least 90% of the currently active schema's feature columns (by exact or normalized name) reuses the existing model outright — no fitting happens at all. This is the common case: the same survey/export re-uploaded with new or updated rows. A genuinely different dataset, or one missing more than 10% of the current schema, triggers training — but even then, it isn't a blank-slate retrain: `psm_core.select_or_merge_features` keeps whichever of the *previous* schema's features are still present and usable, and only backfills the vacated slots with this upload's own top-ranked features. So the feature set evolves rather than resets on every structural change.

### Train on an uploaded CSV
```bash
curl -X POST http://localhost:8001/train -F "file=@mydataset.csv"
```
Optional form fields:
- `treatment_column=enrolled_flag` — bypasses auto-detection if it picks the wrong column
- `force_retrain=true` — skips the reuse shortcut and retrains even if coverage is high

**Reused** (coverage was high enough — no training happened):
```json
{
  "status": "reused",
  "reason": "upload already covers 30/30 of the active schema's features; reusing the existing model instead of retraining",
  "coverage": 1.0,
  "treatment_column": "Y_BOAT-RE",
  "n_features_selected": 30,
  "unmatched_features": []
}
```

**Trained** (no active schema yet, or coverage was too low):
```json
{
  "status": "trained",
  "rows": 1339,
  "treatment_column": "Y_BOAT-RE",
  "treatment_detection_method": "notna_mask",
  "n_features_selected": 30,
  "top_features": [{"feature": "I5:TFV", "importance": 0.136}, "..."],
  "excluded_as_leakage": ["A2:GROUP", "J1:BOAT_AGREE", "..."],
  "kept_from_previous_schema": ["D3.1:A_CP", "B5:SEX", "..."],
  "added_new": ["C5:TOT_INCOME/B", "E3:A_POWER-SUP", "..."],
  "dropped_from_previous_schema": ["I5:TFV", "B3:AGE", "..."]
}
```
`kept_from_previous_schema` / `added_new` / `dropped_from_previous_schema` are empty/full-30 on the very first-ever `/train` call, since there's no previous schema yet to merge with.

**How treatment detection works** (`psm_core.detect_treatment_column`): looks for a column literally named `treatment`; failing that, scores every column as either an already-binary flag (0/1, Yes/No, True/False) or a "populated only for one group" column (like `Y_BOAT-RE`, non-null only for program participants), favoring the latter since that's the far more common pattern in program/survey datasets, and breaking ties by earliest column position. This is a heuristic over a genuinely ambiguous problem — always check `treatment_column`/`treatment_detection_method` in the response, and override with the `treatment_column` form field if it's wrong.

**How feature selection works** (`psm_core.select_or_merge_features`): fits a `GradientBoostingClassifier` on every numeric column to predict the detected treatment column, excludes candidates that are near-direct proxies for treatment (>0.95 correlated in either raw value or null-pattern — this is what filters out things like a literal treatment/control group column, or a whole block of post-treatment follow-up questions that are only asked of participants), then keeps the top 30 by importance — carrying over as much of the previous schema as still fits before backfilling with new features — and refits the final model on just those.

Only numeric columns are considered as candidate features; categorical/text columns are ignored. Missing values in candidate feature columns are filled with 0.

### Predict propensity scores / estimate ATT
Same request/response shape as the static service's `/predict_ps` and `/estimate_att` (see section 3), but scored against whichever dataset was last trained via `/train`. Returns `409` if nothing has been trained yet.

### Health
```bash
curl http://localhost:8001/health
```
Before training:
```json
{ "status": "empty", "psm": null, "message": "no dataset trained yet; POST a CSV to /train" }
```
After training:
```json
{
  "status": "ok",
  "psm": {
    "source": "mydataset.csv",
    "rows": 1339,
    "trained_at": 1752566400.0,
    "last_action": "trained",
    "treatment_column": "Y_BOAT-RE",
    "treatment_detection_method": "notna_mask",
    "n_features_selected": 30,
    "top_features": [{"feature": "I5:TFV", "importance": 0.136}, "..."]
  }
}
```
`last_action` is `"trained"` or `"reused"` depending on what the most recent `/train` call actually did.
