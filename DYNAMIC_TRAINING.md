# Dynamic Scoring: Delete-and-Retrain ("Teachable Machine")

## Why this exists

An earlier revision of this service tried to avoid retraining altogether — a
program's columns got folded into 6 universal composite indices and scored by a
second frozen model, so nothing ever needed to be refit. That approach traded away
resolution (57 raw features compressed into 6 numbers) and needed real upfront domain
modeling (the index taxonomy) for every new kind of program.

The decision was made to go the other direction instead: **the dynamic model behaves
like Google's Teachable Machine.** Every `POST /train` call with a new column set
deletes whatever model is currently active and trains a completely fresh one on the
new upload — no merging with the previous schema, no persisted history beyond "the
most recent model." Feature selection is what it's always been: rank every usable
column in the upload by feature importance -- with no top-N cap, every
non-leakage-correlated candidate is used and reported. Curating that ranked list down
to a smaller working set is the integrator's call, not something this service decides
for them. Any dataset with enough usable columns and a detectable treatment/control
indicator can be trained on, regardless of what its headers are called.

This restores the *original* `app_dynamic.py` design (last present at commit
`e1a2d1e`) with one deliberate simplification: the old version's 90%-coverage
reuse-shortcut and previous-schema merge (`select_or_merge_features`) are gone. The
only reuse-shortcut that exists today is narrower and exact: an upload whose column
set is *identical* to the one that trained the currently active model skips retraining
entirely (see "Retrain-skip" below) — anything else, including an upload that's 99%
the same schema, retrains from scratch.

## How it works

`app.py` serves two models side by side, decided per request by what the request's
columns cover:

| | Baseline | Dynamic |
|---|---|---|
| **Trigger** | Request covers all 57 raw bfar.csv columns | Anything else |
| **Model** | `models/best_model.pkl`, frozen, produced by `build_model.py` | Whatever `POST /train` last produced |
| **Retrained on upload?** | Never | Every `/train` call, unless the upload's columns exactly match what trained the active model |
| **Persisted?** | Yes, committed to the repo | Yes, `models/dynamic/` (gitignored runtime state) |
| **Feature set** | The fixed 57 bfar features | All non-leakage-correlated candidates, ranked by importance, fresh each retrain -- no top-N cap |

The baseline path exists because it's free — bfar.csv's own reference model needs no
training step and is always available, so a request that happens to carry the exact
57 known columns gets the most accurate, already-validated scoring with zero latency
cost. Everything else needs *some* model to score against, and that's the dynamic
one.

### Retrain-skip

Before doing anything else, `/train` compares the uploaded CSV's full column set
(sorted, exact match) against `STATE["trained_columns"]` — the columns of whatever
dataset trained the currently active model. If they're identical, steps 2–5 below are
skipped entirely: the existing model, feature set, and treatment column are reused
as-is, and the response reports `"retrained": false`. This only saves the fit itself
— the response's `ps_output`/`covariate_balance`/`decision_support` sections are still
recomputed against the new upload's rows, since those describe *this* upload, not the
model. Any column added, removed, or renamed forces a full retrain.

### `POST /train` (when it does retrain)

1. Parse the uploaded CSV; reject if fewer than 10 rows.
2. Auto-detect the treatment/control column (`psm_core.detect_treatment_column`;
   override with the `treatment_column` form field if it guesses wrong).
3. Rank every numeric candidate column by importance for predicting treatment
   (`psm_core.select_top_features`), excluding near-perfect treatment proxies
   (`psm_core._leakage_correlated_columns` — a column whose null-pattern or raw
   values correlate ≥0.95 with treatment is almost certainly a renamed copy of the
   group assignment itself, not a genuine covariate).
4. Fit a fresh `GradientBoostingClassifier` on every ranked candidate
   (`psm_core.train_psm_model`) -- no top-N cap; the full ranking is reported
   back so the integrator can trim it further if they want a smaller feature
   set for their own purposes.
5. **Check covariate balance** (`psm_core.covariate_balance`): 1-NN caliper-matches
   treated to control on the fitted model's propensity score, computes standardized
   mean difference per selected feature before/after matching. If the mean |SMD after
   matching| is `>= 0.1`, the single worst-balanced feature is dropped and steps 3–5
   repeat, up to 3 attempts total (`MAX_RETRAIN_ATTEMPTS` in `app.py`) — whichever
   attempt's result is used, balanced or not, becomes final; the response's
   `retrain_attempts` and `feature_selection.dropped_for_rebalancing` show what
   happened.
6. Overwrite whatever was in the dynamic model slot, and persist it (plus
   `trained_columns`, for the next call's retrain-skip check) to `models/dynamic/` so
   a process restart doesn't lose it.

There is no "reuse this if it's similar enough" shortcut and no feature carryover
from the previous model — a `/train` call with a different (not identical) column set
produces a completely independent model, discarding the first one entirely.

### Scoring (`/train/predict_ps`, `/train/estimate_att`, `/train/predict_ps_batch`)

Each request is scored against whichever model applies: the frozen baseline if every
required column is present, otherwise the current dynamic model. If neither applies
(nothing trained yet, and the request doesn't cover all 57 baseline columns), the
response is `409` with a message pointing at `/train`. These three paths live only on
the dynamic port (`:8000`, under the `/train/` prefix) — the static port (`:8001`)
exposes the same three at the bare paths (`/predict_ps` etc.), baseline-only, no
`/train` prefix since it never trains anything.

## What was removed

The index-mapping system built for the previous approach — `column_matcher.py`,
`mapping_store.py`, `psm_indices.py`, the `models/index_*` artifacts, and
`build_model.py`'s second (index-space) model-training pass — is deleted, not kept
alongside this. A program can't simultaneously be "a frozen model, never retrained"
and "delete and retrain," since those are different promises about the same request.

## Known limitation

Because feature selection is fully data-driven per upload, propensity scores from a
freshly trained dynamic model can be poorly calibrated on small or highly separable
datasets — a `GradientBoostingClassifier` fit on a few hundred rows can produce
near-degenerate propensity scores (many values pinned near 0 or 1), which in turn
can produce zero matched pairs for `/estimate_att`. This isn't a bug so much as an
inherent property of fitting a fresh, unvalidated model on whatever's uploaded — the
same tradeoff the original `app_dynamic.py` always had. Use `/health`'s `dynamic.rows`
and treat estimates from very small training sets with proportionate skepticism. With
no top-N cap on feature count, a dataset with many numeric columns and few rows makes
this worse (more features than observations), which is exactly the case the
covariate-balance re-tune loop (dropping the worst-balanced feature and refitting) is
there to push back against, without eliminating the risk.

## Verification performed

- `build_model.py` regenerated and reproduces `bfar_with_ps.csv` predictions to
  floating-point precision (unchanged from the prior baseline).
- Baseline fast path (`/predict_ps` with all 57 raw features) still scores correctly
  with no dynamic model trained.
- A dataset that doesn't cover all 57 features returns `409` before any `/train` call.
- `POST /train` with a CSV produces a `trained` response with every non-leaky
  candidate feature ranked and a leakage-exclusion list; subsequent
  `/train/predict_ps`/`/train/estimate_att` calls against that schema succeed.
- A second `/train` call with a different dataset fully replaces the first model
  (confirmed via `/health`'s `dynamic.source_filename`/`rows`/`trained_at` changing,
  with no features carried over).
- The trained state survives a process restart (`load_state()` on startup).
