# Current ML Model — Snapshot

A record of what's actually implemented, written from `app.py`, `psm_core.py`, and
`build_model.py` as they stand right now. Supersedes the pre-pipeline-upgrade version
of this document — retrain-skip, `/train/*` URL nesting, and the covariate-balance /
model-interpretation / decision-support additions to `/train`'s response (steps 6, 7,
9, 10 below) are now live.

## 1. Architecture

One process (`app.py`), two independent Flask servers, sharing `psm_core.py` and the
`models/` baseline artifacts:

| | Static | Dynamic |
|---|---|---|
| Port | `8001` (`STATIC_PORT`) | `8000` (`PORT`) |
| Serves | Frozen bfar.csv baseline only | Baseline (fast path) + trainable dynamic model |
| `/train`? | No | Yes |
| Scoring paths | `/predict_ps`, `/estimate_att`, `/predict_ps_batch` | `/train/predict_ps`, `/train/estimate_att`, `/train/predict_ps_batch` |
| Rejects incomplete input? | `409` if request doesn't cover all 57 baseline features | `409` only if *neither* baseline nor a trained dynamic model applies |

The scoring paths differ by design: the static port never trains anything, so its
paths stay bare; the dynamic port nests them under `/train/` to make explicit that
they read whatever `/train` last produced.

## 2. Baseline model (`build_model.py`)

Unchanged from before. Trained once from `bfar.csv`, committed to the repo, never
retrained by the running service.

- **Features:** a fixed list of 57 "pre-program" columns (asset ownership, utilities,
  housing, government insurance/benefits — see `ALL_FEATURES` in `build_model.py`).
- **Treatment label:** `Y_BOAT-RE` non-null → treatment=1.
- **Imputation:** median (numeric) / mode (object) — `psm_core.impute_dataframe`.
- **Model selection:** 5-fold stratified CV across Logistic Regression, Random Forest,
  Gradient Boosting, Neural Network; lowest MSE wins, refit on the full dataset
  (currently Gradient Boosting).
- **Artifacts:** `models/best_model.pkl`, `scaler.pkl`, `all_features.json`,
  `core_features.json`, `remaining_features.json`.
- No train/test split — the winner is refit on 100% of the data after CV picks it.

## 3. Dynamic model (`POST /train`, dynamic service only)

"Teachable Machine" style, with one exact-match shortcut and a balance-driven retry
loop layered on top.

### Retrain-skip

Before anything else, `/train` compares the uploaded CSV's full column set (sorted)
against `STATE["trained_columns"]` — the columns of whatever dataset trained the
currently active model. Identical → skip training entirely, reuse the existing model,
`"retrained": false`. Any column added/removed/renamed → full retrain. The
`ps_output`/`covariate_balance`/`decision_support` sections are always recomputed
against the new upload's rows regardless, since those describe *this* upload, not
whether the model changed.

### Feature selection — no cap

1. Auto-detect the treatment/control column (`psm_core.detect_treatment_column`;
   override via `treatment_column` form field).
2. Rank every numeric, non-ID-like candidate column by importance for predicting
   treatment (`psm_core.select_top_features`, a throwaway
   `GradientBoostingClassifier`).
3. Exclude near-perfect treatment proxies (`psm_core._leakage_correlated_columns`,
   ≥0.95 correlation with treatment's value or null-pattern).
4. **Fit on every remaining ranked candidate — no top-N cutoff.** The full ranking is
   reported in `feature_selection.selected` / `model_interpretation.feature_contributions`;
   curating that list down to a smaller working set is left to the integrator, not
   decided by this service.

### Covariate-balance re-tune loop (steps 5–7 of the pipeline diagram)

After fitting, `psm_core.covariate_balance`:
- 1-NN caliper-matches treated to control on the fitted model's logit-PS.
- Computes standardized mean difference (SMD) per selected feature, before and after
  matching.
- Computes PS common-support overlap between groups.
- Verdict: `balance_achieved` = mean |SMD after matching| `< 0.1` (falls back to
  pre-match SMD if no pairs matched).

If not achieved, the single worst-balanced feature is dropped
(`dropped_for_rebalancing`) and steps above repeat, up to `MAX_RETRAIN_ATTEMPTS = 3`
total attempts — whichever attempt's result exists when attempts run out becomes
final, balanced or not (`retrain_attempts` reports how many were used).

### Persistence

Model + feature set + treatment column + `trained_columns` (for the next call's
retrain-skip check) + `excluded_as_leakage` + `dropped_for_rebalancing` are all saved
to `models/dynamic/` (`model.pkl` + `meta.json`) so a restart doesn't lose them.

## 4. `POST /train` response shape

```
{
  "status": "trained",
  "retrained": bool,                 # false if retrain-skip fired
  "retrain_attempts": int,           # 0 if skipped
  "rows": int,
  "treatment_column": str,
  "treatment_detection_method": str,
  "feature_selection": {             # pipeline step 3, surfaced explicitly
    "n_features_selected": int,
    "selected": [{"feature": str, "importance": float}, ...],   # every ranked candidate, no cap
    "excluded_as_leakage": [str, ...],
    "dropped_for_rebalancing": [str, ...]
  },
  "ps_output": {                     # step 6 — in-sample, on this upload
    "ps": [float, ...], "ps_logit": [float, ...],
    "ps_summary": {"min", "max", "mean", "median"}
  },
  "covariate_balance": {             # step 7
    "balance_achieved": bool, "mean_abs_smd": float, "balance_threshold": 0.1,
    "matched_pairs": int, "caliper": float,
    "overlap": {"treated_in_control_range_pct", "control_in_treated_range_pct"},
    "per_feature": [{"feature", "smd_before", "smd_after"}, ...],
    "worst_feature": str
  },
  "model_interpretation": {          # step 9 — NOT true SHAP, see below
    "method": "GradientBoostingClassifier.feature_importances_ (not SHAP)",
    "feature_contributions": [{"feature": str, "importance": float}, ...]
  },
  "decision_support": [{"ps_group", "count", "interpretation", "mean_*"}, ...],  # step 10
  # kept for backwards compatibility with pre-upgrade callers:
  "n_features_selected": int, "top_features": [...], "excluded_as_leakage": [...]
}
```

`model_interpretation` deliberately reuses the same `feature_importances_` values
already computed for ranking/selection rather than computing true SHAP values — no
`shap` package dependency was added. If real Shapley values are needed later, this is
the section to swap out (`psm_core` would need a `shap.TreeExplainer` call against the
fitted `GradientBoostingClassifier`).

## 5. Scoring (`/train/predict_ps`, `/train/estimate_att`, `/train/predict_ps_batch`,
   and their static-port equivalents without the `/train` prefix)

Unchanged logic from before, only the dynamic-port paths moved. Resolve which model
applies (baseline if all 57 features present, else dynamic, else `409`), impute/score,
return propensity scores. `/estimate_att` (`psm_core.matched_att`, now built on the
shared `psm_core._match_pairs` helper) does 1-NN caliper matching + paired t-test +
bootstrap CI. `/predict_ps_batch` (`psm_core.decision_support_table`) stratifies into
PS quartiles.

## 6. Current endpoints

**Dynamic — `:8000`**
`GET /` (test UI) · `GET /health` · `POST /train` · `POST /train/predict_ps` ·
`POST /train/estimate_att` · `POST /train/predict_ps_batch`

**Static — `:8001`**
`GET /health` · `POST /predict_ps` · `POST /estimate_att` · `POST /predict_ps_batch`

## 7. Mapped against the 10-step pipeline diagram

| Step | Status |
|---|---|
| 1. Raw data | ✅ `bfar.csv`, or whatever's uploaded to `/train` |
| 2. Preprocessing | Handled upstream of this service (per integrator) — this service only does median/mode imputation and `.fillna(0)` at fit/score time |
| 3. Feature engineering & selection | ✅ Importance-based ranking + leakage exclusion, surfaced in `feature_selection`; no PCA/clustering (handled upstream, per integrator) |
| 4. Stratified train-test split | ❌ Not done — both baseline and dynamic models fit on 100% of their data |
| 5. PS estimation (multi-model) | Baseline compares all 4 candidates via CV; dynamic always uses Gradient Boosting, with the balance re-tune loop as its only iteration mechanism |
| 6. PS output | ✅ `ps_output` in `/train`'s response, `ps_final`/`ps` in scoring responses |
| 7. Covariate balance diagnostics | ✅ `covariate_balance` in `/train`'s response (SMD, overlap, balance_achieved + auto re-tune) |
| 8. Causal estimation (matching/ATT) | ✅ `/train/estimate_att` (and static `/estimate_att`) |
| 9. Model interpretation | ⚠️ `model_interpretation` in `/train`'s response, but `feature_importances_`, not true SHAP (by choice — no new dependency) |
| 10. Decision support system | ⚠️ `decision_support` quartile table (in `/train` and `/train/predict_ps_batch`); no generated reports or visualizations beyond the raw table |

## 8. Known limitations

- Dynamic model calibration can be poor on small/highly-separable uploads.
- No top-N cap means a dataset with many numeric columns and few rows can end up with
  more features than observations; the balance re-tune loop pushes back against this
  by dropping the worst-balanced feature, but doesn't eliminate the risk, and can end
  up discarding a feature that was actually carrying real signal (it optimizes for
  balance, not predictive accuracy).
- No cross-validation or held-out evaluation for the dynamic path.
- `model_interpretation` is feature importance, not SHAP.
