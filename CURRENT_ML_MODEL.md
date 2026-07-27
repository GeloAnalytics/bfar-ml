# Current ML Model — Snapshot

A record of what's actually implemented, written from `app.py`, `psm_core.py`, and
`build_model.py` as they stand right now. Supersedes the pre-pipeline-upgrade version
of this document — retrain-skip, `/train/*` URL nesting, and the covariate-balance /
model-interpretation / decision-support additions to `/train`'s response (steps 6, 7,
9, 10 below) are now live, and step 9 uses real SHAP values (`shap.TreeExplainer`),
not a feature-importances_ stand-in.

A more elaborate demographic-keyword / before-after-wave-pair / low-coverage / manual
override column exclusion system was built, tested, and then reverted — too complex
for the value it added. Re-added in two steps since: the before/after wave-pair
structural detector first (a correctness fix -- post-treatment data as a PS
predictor), then the demographic-keyword and low-coverage filters on top of it (see
below). Still no manual exclude_columns/include_columns overrides.

Latest addition: **livelihood profiling and program impact dashboard**. `/train`
now auto-detects or accepts an explicit `outcome_column` form field, runs matched-pair
ATT estimation against it, and computes pre-post change profiles across every detected
before/after wave-pair column for both the treated group and their matched controls
(`profile_updates`). `test_ui.html` has been redesigned as a dark-mode program impact
dashboard with KPI cards, an outcome-change summary, and categorized tabs (Income,
Assets, Appliances, Gadgets, Utilities, Housing, Insurance) each showing treated vs
control stacked progress bars for every wave-pair feature.

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

### Feature selection — no cap, three exclusion filters

1. Auto-detect the treatment/control column (`psm_core.detect_treatment_column`;
   override via `treatment_column` form field). `test_ui.html`'s train form exposes
   this as a `<select>` dropdown: picking a CSV file parses just its header row
   client-side (`FileReader`, first 8KB) and populates the dropdown with the actual
   column names, defaulting to "Auto-detect" -- no need to already know or type the
   exact column name. A separate **outcome column** dropdown works the same way:
   override via the `outcome_column` form field or let the auto-detector pick
   (`psm_core.detect_outcome_column` -- prefers `C5:TOT_INCOME/B` for BFAR data, then
   keyword-matches "income"/"outcome" in column names, then falls back to the first
   usable numeric non-treatment column).
2. Narrow numeric, non-ID-like candidate columns through three filters, in order (each
   reported separately in `feature_selection`):
   - **Demographic/respondent-identity keyword match**
     (`psm_core._context_excluded_columns`) -- generic survey terms (age, respondent,
     area, sex, marital status, education, ...), not tied to any one program's naming
     scheme. Verified against bfar.csv's 215 raw columns: 5 exact matches (`AREA`,
     `AGE`, `SEX`, `M-STATUS`, `EDUCATION`), zero false positives. Household size
     (`B8:HH_SIZE`) is deliberately *not* included -- treated as livelihood-adjacent
     (household composition affects economic need), not pure demographics.
   - **Before/after wave-pair structural match** (`psm_core._wave_pair_excluded_columns`)
     -- a column that's the "current" half of a pair sharing an identical name except
     for one isolated `A`/`B` token (e.g. `D1.2:A_MOTORC` / `D1.2:B_MOTORC`). Confirmed
     against the actual BFAR beneficiary questionnaire: Parts C/D/E/F/G each ask every
     item twice ("before receiving the boat" / "at present" -- a baseline/endline
     design). A *structural* pattern match (does a same-named counterpart column exist
     with the token swapped?), not a hardcoded word list, so it generalizes to other
     before/after-design datasets. Verified: 71 pairs detected on bfar.csv's 215 raw
     columns, zero false positives (e.g. `I2:A/C_M` -- association/club membership --
     correctly left alone since no `I2:B/C_M` counterpart exists). **Known gap:** some
     current-wave columns have no "before" twin to pair against at all (bfar.csv's
     `C2:INCOME/B/FISH`, `C4:INCOME/B/ALT`, `C5:TOT_INCOME/B` -- current income, but the
     questionnaire never asked a matching "before" breakdown by source) -- no generic
     structural signal can catch these directly, though the covariate-balance re-tune
     loop below sometimes drops them anyway for being poorly balanced.
   - **Leakage correlation with treatment** (`psm_core._leakage_correlated_columns`,
     ≥0.95 correlation with treatment's value or null-pattern) -- unchanged from
     before; this is what catches bfar.csv's entire J-series (boat-repair-specific
     follow-up, populated only for beneficiaries) and `A2:GROUP` automatically.
3. **Fit on every remaining ranked candidate — no top-N cutoff.** The full ranking is
   first attempt uses the full ranked pool; balance retries then keep progressively
   smaller top-ranked subsets.

On the full 215-column `bfar.csv` (not just the 57-feature baseline subset), this
the first attempt narrows candidates to 104 after low-coverage, demographic,
wave-pair, and leakage exclusions. The balance loop then shrinks the active selected
set further when needed; on `bfar.csv` it settles at 7 balanced features.

### Covariate-balance re-tune loop (steps 5–7 of the pipeline diagram)

After fitting, `psm_core.covariate_balance`:
- 1-NN caliper-matches treated to control on the fitted model's logit-PS.
- Computes standardized mean difference (SMD) per selected feature, before and after
  matching.
- Computes PS common-support overlap between groups.
- Verdict: **count-based, not mean-based.** `balance_achieved` = true if either the
  matched-pairs check or the IPW reweighted check stays within the same
  `MAX_UNBALANCED_FEATURE_PCT` (35%) cap on features individually having |SMD| `>=
  0.1`. The matched-pairs verdict is still reported as `matched_balance_achieved`; the
  IPW verdict lives under `ipw.balance_achieved`. Requiring the *average* SMD across
  every candidate to be low turned out to be unrealistically strict once a dataset has
  50-100+ real socioeconomic covariates -- a handful of genuinely different features
  drags the mean up and fails the whole model even when most features are well-matched.
  Mirrors the reference PSM notebook's Cell 9 IPW balance check (tolerates up to 10 of
  57 features, ~17.5%, rather than an aggregate mean threshold), expressed as a
  percentage so it scales with however many features a given dynamic model actually
  selected.

If not achieved, the model is refit on progressively smaller top-ranked subsets
(`BALANCE_SHRINK_FACTOR`, with a `MIN_BALANCE_FEATURES` floor), up to `MAX_RETRAIN_ATTEMPTS`
total attempts — whichever attempt's result exists when attempts run out becomes
final, balanced or not (`retrain_attempts` reports how many were used).

### Persistence

Model + feature set + treatment column + `trained_columns` (for the next call's
retrain-skip check) + `excluded_as_leakage` + `excluded_as_wave_pair` +
`excluded_as_context` + `dropped_for_rebalancing` are all saved to `models/dynamic/`
(`model.pkl` + `meta.json`) so a restart doesn't lose them.

## 4. `POST /train` response shape

**Two independent counts to keep straight:** `rows` (== `ps_output.n_rows_scored`)
is the number of *respondents* in the upload -- one propensity score per row, always,
regardless of feature count. `feature_selection.n_features_selected` (==
`model_interpretation.n_features_ranked`) is the number of *columns* used as
predictors -- unrelated to row count, and can be smaller, larger, or equal by pure
coincidence. Seeing different numbers here (e.g. 1339 rows but only 11 features) is
normal, not a bug -- one counts people, the other counts predictor columns.

```
{
  "status": "trained",
  "retrained": bool,                 # false if retrain-skip fired
  "retrain_attempts": int,           # 0 if skipped
  "rows": int,
  "treatment_column": str,
  "treatment_detection_method": str,
  "outcome_column": str,             # auto-detected or the override passed in the form
  # Describes COLUMNS (features) -- list lengths here are feature counts,
  # unrelated to how many rows were uploaded.
  "feature_selection": {             # pipeline step 3, surfaced explicitly
    "n_features_selected": int,
    "selected": [{"feature": str, "importance": float}, ...],   # every ranked candidate, no cap
    "excluded_as_leakage": [str, ...],
    "excluded_as_wave_pair": [str, ...],
    "excluded_as_context": [str, ...],
    "dropped_for_rebalancing": [str, ...]
  },
  # Describes ROWS/RESPONDENTS -- exactly one propensity score per uploaded
  # row. len(ps) == len(ps_logit) == n_rows_scored == the top-level "rows"
  # field, always, regardless of how many features were used to compute
  # each score.
  "ps_output": {                     # step 6 — in-sample, on this upload
    "n_rows_scored": int,
    "ps": [float, ...], "ps_logit": [float, ...],
    "ps_summary": {"min", "max", "mean", "median"}
  },
  "covariate_balance": {             # step 7
    "balance_achieved": bool,          # count-based: see n_features_over_threshold below
    "mean_abs_smd": float,             # informational only, no longer the deciding metric
    "n_features_over_threshold": int, "max_unbalanced_features_allowed": int,
    "balance_threshold": 0.1,
    "matched_pairs": int, "caliper": float,
    "overlap": {"treated_in_control_range_pct", "control_in_treated_range_pct"},
    "per_feature": [{"feature", "smd_before", "smd_after"}, ...],
    "worst_feature": str
  },
  # ATT estimate from 1-NN caliper matching on the outcome column. Excludes
  # pair_profiles and profiling_summary (kept as top-level fields separately).
  "att_result": {                    # step 8 — in-sample ATT on this upload
    "matched_pairs": int,
    "att_mean": float,
    "ci_95": [float, float],
    "p_value_paired_ttest": float,
    "caliper": float
  },
  # Per-matched-pair outcome detail (row indices + outcome values + status).
  "pair_profiles": [{"treated_index", "control_index", "treated_outcome",
                     "control_outcome", "outcome_difference", "status"}, ...],
  # Aggregate tally of outcome-change direction across all matched pairs.
  "profiling_summary": {"increased_count": int, "decreased_count": int, "no_change_count": int},
  # Pre-post change profile for EVERY detected wave-pair column (psm_core.find_wave_pairs),
  # for both the treated group and their matched controls. Each entry covers one
  # before/after pair, e.g. C1:TOT_INCOME/A -> C5:TOT_INCOME/B.
  "profile_updates": [{
    "feature": str,                  # human-friendly label derived from the pre-column name
    "col_pre": str,                  # original before-program column name
    "col_post": str,                 # original after-program column name
    "treated": {"increased": int, "decreased": int, "no_change": int, "total": int},
    "control": {"increased": int, "decreased": int, "no_change": int, "total": int}
  }, ...],
  # Also describes COLUMNS, not rows -- a ranking of which features drove
  # the model's predictions. n_features_ranked is a different number from
  # ps_output.n_rows_scored above: one counts respondents, the other counts
  # input columns. These two counts being different is expected, not a bug.
  "model_interpretation": {          # step 9 — real SHAP values
    "method": "SHAP (shap.TreeExplainer, exact for tree-ensemble models) ...",
    "n_features_ranked": int,
    "feature_contributions": [
      {"feature": str, "mean_abs_shap": float, "mean_shap": float, "direction": "increases_likelihood"|"decreases_likelihood"}, ...
    ],
    "socioeconomic_insights": [str, ...]   # plain-language, top 5 by default
  },
  "decision_support": [{"ps_group", "count", "interpretation", "mean_*"}, ...],  # step 10
  # kept for backwards compatibility with pre-upgrade callers:
  "n_features_selected": int, "top_features": [...], "excluded_as_leakage": [...]
}
```

`model_interpretation` uses real SHAP values (`psm_core.compute_shap_feature_contributions`,
`shap.TreeExplainer` against the fitted `GradientBoostingClassifier` -- exact, not
approximated, since tree ensembles have a closed-form SHAP computation). Reports the
mean absolute SHAP value per feature across every row in this upload (the standard
"global SHAP importance" view, not a per-row breakdown -- keeps the response a
reasonable size), the signed mean (which direction the feature pushes predictions),
and `socioeconomic_insights`: generic template sentences built from the top-ranked
features' names and directions, not tied to any one program's column-naming scheme.
SHAP values are in the model's raw log-odds (margin) space, not probability space --
not directly comparable in magnitude to a probability difference. Adds `shap` as a
new dependency (`requirements.txt`).

## 5. Scoring (`/train/predict_ps`, `/train/estimate_att`, `/train/predict_ps_batch`,
   and their static-port equivalents without the `/train` prefix)

Dynamic-port paths always resolve to whichever dynamic model `/train` most recently
produced (never the frozen baseline, even if a request covers all 57 raw columns;
`409` if nothing has been trained yet). Static-port equivalents remain baseline-only,
unconditionally, on their own port. `/estimate_att` (`psm_core.matched_att`, built on the
shared `psm_core._match_pairs` helper) does 1-NN caliper matching + paired t-test +
bootstrap CI. Returns `pair_profiles` (detailed stats per matched pair) and a
`profiling_summary` tallying Increased/Decreased/No Change outcomes.

**Livelihood profiling (new, in-train only).** `POST /train` now also runs
`psm_core.detect_outcome_column` to identify or accept the outcome column, computes
`matched_att` against it in-train, and then calls `psm_core.find_wave_pairs` +
`psm_core.compute_wave_pair_profiling` to produce `profile_updates`: a list of
before/after change stats for every detected wave-pair column (58 detected on `bfar.csv`)
for both the matched treated group and their matched control group. This makes it
possible to see, for each life-quality indicator (income, motorcycle ownership, appliance
acquisition, insurance uptake, etc.), how many beneficiaries improved, stayed the same,
or fell back -- relative to what happened to comparable non-beneficiaries over the same
period. `/predict_ps_batch` (`psm_core.decision_support_table`) stratifies into PS quartiles.

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
| 3. Feature engineering & selection | ✅ Importance-based ranking + wave-pair + leakage exclusion, surfaced in `feature_selection`; no PCA/clustering (handled upstream, per integrator) |
| 4. Stratified train-test split | ❌ Not done — both baseline and dynamic models fit on 100% of their data |
| 5. PS estimation (multi-model) | Baseline compares all 4 candidates via CV; dynamic always uses Gradient Boosting, with the balance re-tune loop as its only iteration mechanism |
| 6. PS output | ✅ `ps_output` in `/train`'s response, `ps_final`/`ps` in scoring responses |
| 7. Covariate balance diagnostics | ✅ `covariate_balance` in `/train`'s response (SMD, overlap, balance_achieved + auto re-tune) |
| 8. Causal estimation (matching/ATT) | ✅ `/train/estimate_att` (and static `/estimate_att`) |
| 9. Model interpretation | ✅ `model_interpretation` in `/train`'s response -- real SHAP values (`shap.TreeExplainer`) plus generated socioeconomic insights |
| 10. Decision support system | ✅ `decision_support` quartile table + `profile_updates` (Treated vs Control Increased/Decreased/No Change per wave-pair column) + dark-mode program impact dashboard (`test_ui.html`) with KPI cards and categorized tabs |

## 8. Known limitations

- Dynamic model calibration can be poor on small/highly-separable uploads.
- The first dynamic attempt has no top-N cap, so a dataset with many numeric columns
  and few rows can start with more features than observations. The balance re-tune
  loop pushes back by shrinking to progressively smaller top-ranked subsets, but can
  still discard a feature that was carrying real signal (it optimizes for balance, not
  predictive accuracy).
- No cross-validation or held-out evaluation for the dynamic path.
- Column selection excludes low-coverage columns, leakage-correlated columns,
  before/after wave-pairs, and demographic columns (via a generic keyword list), but
  has no manual exclude/include override.
- The wave-pair detector only catches columns with a genuine "before" counterpart to
  structurally pair against. A "current wave" column with no such counterpart (e.g.
  bfar.csv's `C2:INCOME/B/FISH`) has no generic signal indicating it's post-treatment
  and will still be used as a candidate.
- `profile_updates` is computed only in-train (on `/train`'s own upload); it is not
  available from `/train/estimate_att` (which works on arbitrary JSON records that
  may not contain wave-pair columns). For a per-record ATT call, the matched-pair
  outcome change direction is still available in `pair_profiles` + `profiling_summary`.
- `psm_core.find_wave_pairs` uses a prefix-stripping heuristic that correctly handles
  BFAR's `C1:TOT_INCOME/A` / `C5:TOT_INCOME/B` case (different prefix, matched via
  core-name token swap). If a dataset uses a naming scheme where the section prefix
  itself changes between waves (not just the A/B token), some pairs may not be detected.
