# ML Flask Service (Propensity Score Matching)

A single Flask service (`app.py`, shared logic in `psm_core.py`) grounded in a frozen
**bfar.csv baseline** produced by `build_model.py`. Two frozen models come out of that
build -- a raw 57-feature model and a 6-composite-index model -- and the service never
retrains or overwrites either. Every request routes into one of three tiers:

| Tier | When | What happens |
|---|---|---|
| **1 — exact schema** | Upload carries bfar's exact 57 raw columns | Scored by the raw baseline model (`models/best_model.pkl`). Most accurate path; no fitting. |
| **2 — registered mapping** | Upload's program has a registered column mapping | Columns folded into 6 universal composite indices, scored by the frozen index model (`models/index_model.pkl`). No fitting, no labels needed. |
| **3 — per-request adaptation** | Everything else | A throwaway model is fit for that one request only (requires a treatment column + ≥10 rows) and discarded. Nothing persisted, nothing reused. Works even when **zero** bfar columns are present -- feature selection falls back to fully data-driven over whatever numeric columns exist. |

**Why tier 2 exists:** different livelihood programs use different column headers for
the same underlying questions. The 57 bfar features compress into 6 indices --
`transport_assets`, `household_durables`, `connectivity`, `utilities_access`,
`housing`, `social_protection` -- and a small JSON mapping tells the service which of
a program's columns mean the same thing as bfar's canonical items. Mapped columns
inherit bfar's standardization, so the frozen index model can score any mapped program
without ever retraining.

## Automatic mapping promotion (how a program graduates to tier 2)

No manual registration step. Every tier-3 request also runs a **confidence-gated
column matcher** (`column_matcher.py`: curated keywords per canonical item, abstains
on anything ambiguous — the bijectivity rule drops both sides of any tie). The
resulting draft mapping advances a state machine (`mapping_store.py`):

1. The identical mapping must appear on **3 consecutive uploads** (`PROMOTION_CONSISTENT_UPLOADS`)
   covering **≥ 4 of the 6 indices** (`MIN_INDICES_COVERED`).
2. Then a **sanity gate**: the index values the mapping produces on the triggering
   upload must look like data the baseline has seen (per-index |mean z| ≤ 3,
   `SANITY_MAX_ABS_MEAN`) — the backstop against a mapping that matched confidently
   and consistently but is still semantically wrong.
3. Pass → the mapping is registered and **future** requests from that program route
   to tier 2. The request that triggered promotion still completes on tier 3.

Program identity = the `program_id` field if the caller sends one, else the **column
signature** (hash of the sorted, normalized column names). A schema change means a new
signature and a fresh draft cycle.

Everything lives as plain JSON under `mappings/` (gitignored runtime metadata — never
model weights, never uploaded data), with an append-only `mappings/audit.log`
recording every draft reset, promotion, sanity rejection, and demotion. To demote a
program back to tier 3: `DELETE /mappings/<program_key>` (or delete its file).

If this repo is dropped into another project as a subfolder (commonly named `ml/`), prefix the commands below with that folder (e.g. `pip install -r ml/requirements.txt`, `python ml/app.py`). All paths resolve relative to each file's own location -- `bfar.csv` just needs to stay next to `build_model.py`.

## Integration guide (backend & frontend)

**Architecture.** `app.py` is a plain, unauthenticated Flask HTTP API -- no API key,
session, or user concept. Treat it as an internal ML layer your **backend** calls
server-to-server, not something a frontend talks to directly in production. It binds
`0.0.0.0` by default (LAN/container reachable); it is not meant to be exposed to the
public internet as-is.

```
frontend  -->  your backend  -->  app.py (HOST:PORT from .env, default 0.0.0.0:8000)
```

**Configuration (`.env`).** Loaded on startup via `python-dotenv`:
```bash
HOST=0.0.0.0
PORT=8000
# ML_MODEL_DIR=models      # only if artifacts live somewhere other than ./models
# ML_MAPPINGS_DIR=mappings # only if mapping state should live elsewhere
```

**Recommended backend practice:** send a stable `program_id` with every request for
the same program. It makes promotion tracking robust to minor schema changes and
gives you a human-readable key in `/mappings` instead of a hash.

**Error contract.** Every endpoint returns JSON with the same shape:
- `200` -- success, body is the endpoint-specific payload documented below.
- `400` -- bad/incomplete input (unparsable CSV, no treatment column found on tier 3, dataset too small, ...): `{"error": "<message>"}` -- shown-to-user quality, never a stack trace.
- `404` -- `DELETE /mappings/<key>` for a key that has no registered mapping or draft.
- `500` -- baseline artifacts failed to load at startup: `{"error": "ML artifacts not loaded: <reason>"}`. Ops problem, not user-input problem -- alert on it.

**CORS.** `CORS(app)` currently allows any origin -- fine for local development; if a
frontend ever calls this service directly, restrict it first
(`CORS(app, origins=["https://your-frontend"])`).

**Quick endpoint reference:**

| Method & path | Body | Returns |
|---|---|---|
| `GET /health` | -- | baseline status, mapping counts, per-tier request counters |
| `POST /predict_ps` | JSON `{records, program_id?, treatment_column?, n_flex?}` | `{ps_final, tier, program_key, used_baseline, final_features, ...}` |
| `POST /estimate_att` | JSON `{records}` with `treatment`+`outcome` per record | matched-ATT result + tier/adaptation metadata |
| `POST /predict_ps_batch` | multipart CSV (`file`, `program_id?`, `treatment_column?`, `n_flex?`) | per-row `ps` + decision-support quartile table |
| `GET /mappings` | -- | all registered mappings + in-flight drafts |
| `DELETE /mappings/<program_key>` | -- | demotes the program back to tier 3 |

**Tier-aware response fields** (all three scoring endpoints): `tier` (1/2/3) and
`program_key` always; tier 2 adds `imputed_indices` (indices with no mapped columns,
filled from bfar's median — a consumer should surface how much of the score rests on
real data); tier 3 adds `core_coverage` (fraction of bfar's 30 core features present)
and `mapping_status` (`matched_items`, `indices_covered`, `draft_consistent_count`,
`promoted`, and `sanity_rejected` when the gate blocked a promotion).

**Running in production.** `app.run(...)` is Flask's dev server -- put a real WSGI
server in front:
```bash
pip install waitress   # Windows-friendly; use gunicorn on Linux
waitress-serve --host=0.0.0.0 --port=8000 app:app
```
Note: the mapping store assumes a single process (waitress default). If you scale to
multiple workers, point `ML_MAPPINGS_DIR` at per-instance storage or add locking.

## 1) Install Python dependencies

```bash
pip install -r requirements.txt
```

All baseline artifacts are committed (`models/best_model.pkl`, `scaler.pkl`,
`index_model.pkl`, `index_scaler.pkl`, `all/core/remaining_features.json`,
`index_taxonomy.json`, `index_stats.json`), so no training step is required for
normal use. To regenerate after changing `bfar.csv`:
```bash
python build_model.py
```
This runs 5-fold cross-validated model selection (Logistic Regression, Random Forest,
Gradient Boosting, Neural Network -- lowest MSE wins) twice: once on the raw 57
features (tier-1 model) and once on the 6 composite indices (tier-2 model). It also
regenerates the taxonomy: per-item bfar mean/std, within-index weights (from feature
importances), and the matching keywords used by the column matcher.

## 2) Start the service

```bash
python app.py
```
Reads `HOST`/`PORT` from `.env` (defaults `0.0.0.0:8000`).

## 3) Endpoints

**How treatment detection works** (`psm_core.detect_treatment_column`, tier 3 only): looks for a column literally named `treatment`; failing that, scores every column as either an already-binary flag (0/1, Yes/No, True/False) or a "populated only for one group" column (like `Y_BOAT-RE`, non-null only for program participants), favoring the latter, breaking ties by earliest column position. A heuristic over a genuinely ambiguous problem -- always check `treatment_column`/`treatment_detection_method` in the response and override with `treatment_column` if wrong.

### Health
```bash
curl http://localhost:8000/health
```
```json
{
  "status": "ok",
  "psm": {
    "source": "bfar.csv baseline (frozen) -- tier 1: raw 57-feature model; tier 2: 6-index model via registered mappings; tier 3: per-request adaptation, never persisted",
    "model_type": "GradientBoostingClassifier",
    "index_model_type": "GradientBoostingClassifier",
    "n_core_features": 30, "n_remaining_features": 27, "n_all_features": 57
  },
  "mappings": { "registered": 1, "drafts": 0 },
  "tier_requests_since_start": { "1": 12, "2": 40, "3": 3 }
}
```

### Predict propensity scores (JSON records)
```bash
curl -X POST http://localhost:8000/predict_ps \
  -H "Content-Type: application/json" \
  -d '{
    "program_id": "coastal-livelihood-2026",
    "records": [ { "owns_motorcycle": 1, "has_electricity": 1, "...": 0 } ]
  }'
```
Tier-3 response while the program is still earning its mapping:
```json
{
  "tier": 3,
  "program_key": "coastal-livelihood-2026",
  "ps_final": [0.42],
  "used_baseline": false,
  "core_coverage": 0.0,
  "n_features_used": 14,
  "treatment_column": "enrolled",
  "treatment_detection_method": "binary_value",
  "mapping_status": { "matched_items": 12, "indices_covered": 5, "draft_consistent_count": 2, "promoted": false }
}
```
After promotion the same request returns `tier: 2` with `imputed_indices` listing any
index none of its columns cover.

### Predict propensity scores + decision support (whole CSV)
```bash
curl -X POST http://localhost:8000/predict_ps_batch \
  -F "file=@mydataset.csv" -F "program_id=coastal-livelihood-2026"
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
      { "owns_motorcycle": 1, "...": 0, "treatment": 1, "outcome": 12000 },
      { "owns_motorcycle": 0, "...": 0, "treatment": 0, "outcome": 9000 }
    ],
    "caliper_ratio": 0.2, "n_bootstrap": 200, "seed": 42
  }'
```
Returns `matched_pairs`, `att_mean`, `ci_95`, `p_value_paired_ttest`, `caliper`, plus
the tier metadata. Works on all three tiers; the `outcome` column is always excluded
from candidate features so it can never leak into the propensity model. Optional:
`program_id`, `treatment_column`, `outcomeKey`, `n_flex`, `caliper_ratio`,
`n_bootstrap`, `seed`.

### Mappings
```bash
curl http://localhost:8000/mappings                  # registered + drafts
curl -X DELETE http://localhost:8000/mappings/coastal-livelihood-2026   # demote
```

## Notebooks

`updated_psm.ipynb` and `predictor_psm.ipynb` (not part of this repo, kept alongside it) contain the fuller research workflow this service's baseline is drawn from -- model selection with ROC/calibration plots, balance diagnostics, SHAP explainability, and IPW-based ATT estimation as a cross-check against the matching-based estimate served here. The live API intentionally exposes only the baseline-scoring, index-mapping, and dynamic-adaptation pieces; SHAP and IPW remain notebook-only analysis steps.
