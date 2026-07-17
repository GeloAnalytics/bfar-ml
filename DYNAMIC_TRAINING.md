# Dynamic Scoring: Delete-and-Retrain ("Teachable Machine")

## Why this exists

An earlier revision of this service tried to avoid retraining altogether — a
program's columns got folded into 6 universal composite indices and scored by a
second frozen model, so nothing ever needed to be refit. That approach traded away
resolution (57 raw features compressed into 6 numbers) and needed real upfront domain
modeling (the index taxonomy) for every new kind of program.

The decision was made to go the other direction instead: **the dynamic model behaves
like Google's Teachable Machine.** Every `POST /train` call deletes whatever model is
currently active and trains a completely fresh one on the new upload — no merging
with the previous schema, no reuse-shortcut, no persisted history beyond "the most
recent model." Feature selection is what it's always been: rank every usable column
in the upload by feature importance and keep the top 30. "30/30" describes what the
selector always produces, not a fixed list of column names the upload has to match —
any dataset with enough usable columns and a detectable treatment/control indicator
can be trained on, regardless of what its headers are called.

This restores the *original* `app_dynamic.py` design (last present at commit
`e1a2d1e`) with one deliberate simplification: the old version's 90%-coverage
reuse-shortcut and previous-schema merge (`select_or_merge_features`) are gone.
Every `/train` call retrains, full stop.

## How it works

`app.py` serves two models side by side, decided per request by what the request's
columns cover:

| | Baseline | Dynamic |
|---|---|---|
| **Trigger** | Request covers all 57 raw bfar.csv columns | Anything else |
| **Model** | `models/best_model.pkl`, frozen, produced by `build_model.py` | Whatever `POST /train` last produced |
| **Retrained on upload?** | Never | Every `/train` call, unconditionally |
| **Persisted?** | Yes, committed to the repo | Yes, `models/dynamic/` (gitignored runtime state) |
| **Feature set** | The fixed 57 bfar features | Top 30 by importance, fresh each `/train` call |

The baseline path exists because it's free — bfar.csv's own reference model needs no
training step and is always available, so a request that happens to carry the exact
57 known columns gets the most accurate, already-validated scoring with zero latency
cost. Everything else needs *some* model to score against, and that's the dynamic
one.

### `POST /train`

1. Parse the uploaded CSV; reject if fewer than 10 rows.
2. Auto-detect the treatment/control column (`psm_core.detect_treatment_column`;
   override with the `treatment_column` form field if it guesses wrong).
3. Rank every numeric candidate column by importance for predicting treatment
   (`psm_core.select_top_features`), excluding near-perfect treatment proxies
   (`psm_core._leakage_correlated_columns` — a column whose null-pattern or raw
   values correlate ≥0.95 with treatment is almost certainly a renamed copy of the
   group assignment itself, not a genuine covariate).
4. Keep the top 30, fit a fresh `GradientBoostingClassifier`
   (`psm_core.train_psm_model`).
5. **Unconditionally overwrite** whatever was in the dynamic model slot, and persist
   it to `models/dynamic/` so a process restart doesn't lose it.

There is no "reuse this if it's similar enough" shortcut and no feature carryover
from the previous model — a second `/train` call with a different dataset produces a
completely independent model, discarding the first one entirely.

### Scoring (`/predict_ps`, `/estimate_att`, `/predict_ps_batch`)

Each request is scored against whichever model applies: the frozen baseline if every
required column is present, otherwise the current dynamic model. If neither applies
(nothing trained yet, and the request doesn't cover all 57 baseline columns), the
response is `409` with a message pointing at `/train`.

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
and treat estimates from very small training sets with proportionate skepticism.

## Verification performed

- `build_model.py` regenerated and reproduces `bfar_with_ps.csv` predictions to
  floating-point precision (unchanged from the prior baseline).
- Baseline fast path (`/predict_ps` with all 57 raw features) still scores correctly
  with no dynamic model trained.
- A dataset that doesn't cover all 57 features returns `409` before any `/train` call.
- `POST /train` with a CSV produces a `trained` response with 30 selected features
  and a leakage-exclusion list; subsequent `/predict_ps`/`/estimate_att` calls against
  that schema succeed.
- A second `/train` call with a different dataset fully replaces the first model
  (confirmed via `/health`'s `dynamic.source_filename`/`rows`/`trained_at` changing,
  with no features carried over).
- The trained state survives a process restart (`load_state()` on startup).
