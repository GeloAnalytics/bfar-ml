# Portable Dynamic Scoring: Index-Based Mapping with Automatic Promotion

## Why this exists

The service used to require an upload to contain bfar.csv's literal column names --
all 57 for baseline scoring, or at minimum the 30 "core" features for per-request
adaptation. Any livelihood program whose survey used different headers for the same
underlying questions got a `400` immediately, even when the questions were
semantically identical to bfar's. This document describes the fix: bfar's 57 raw
features compress into 6 universal composite indices (the standard proxy-means-testing
approach used in poverty targeting), a second frozen model is trained on that index
space, and new programs get folded into it via a small JSON mapping that is
**discovered and promoted automatically** -- no human registration step, no
retraining, ever.

## Request flow (all three scoring endpoints)

| Step | What happens | Where |
|---|---|---|
| 1 | Parse input -- JSON `records` or uploaded CSV -- into a DataFrame | `app.py` |
| 2 | Resolve program identity: explicit `program_id` if sent, else the column signature (hash of sorted, normalized headers) | `mapping_store.column_signature` |
| 3 | Route to a tier: all 57 raw bfar columns present -> **tier 1**; registered mapping exists for this key -> **tier 2**; else -> **tier 3** | `app._route_tier` |
| 4 | Score against the resolved tier | `psm_core` |
| 5 | Tier 3 only: the column matcher runs on the headers, the draft mapping advances, promotion may fire -- affecting **future** requests only, never the one in flight | `column_matcher` + `mapping_store` |
| 6 | Response assembled with tier metadata (`tier`, `program_key`, coverage/mapping fields) | `app.py` |

## Tier behavior

| | Tier 1 -- exact schema | Tier 2 -- registered mapping | Tier 3 -- ephemeral adapt |
|---|---|---|---|
| **Trigger** | All 57 raw bfar columns present | Program key has a registered mapping | Everything else |
| **Model used** | `models/best_model.pkl` (raw 57 features) | `models/index_model.pkl` (6 composite indices) | Throwaway GradientBoosting fit on this request, discarded after |
| **Fitting per request** | None | None | Yes (request-scoped only, never persisted) |
| **Needs treatment column** | No | No for predict; yes for ATT | Yes, plus >= 10 rows |
| **Feature source** | The 57 raw columns | Mapped columns -> z-scored -> 6 indices | Core-30 + flex features if core is present; otherwise a fully data-driven top-30 selection over whatever numeric columns exist |
| **Extra response fields** | -- | `imputed_indices` | `core_coverage`, `mapping_status` |

Tier 1 always wins over a registered mapping when both would apply -- the raw model is
the more accurate one, so an exact-schema upload never takes the lossier index path.

## The 6 composite indices

| Index | Drawn from (bfar) | Represents |
|---|---|---|
| `transport_assets` | `D1.*` (bike, motorcycle, tricycle, car, jeep, truck + quantities) | Mobility/capital assets |
| `household_durables` | `D2.*` (TV, fridge, washing machine, aircon, fan, stove, furniture) | Consumption capacity |
| `connectivity` | `D3.*` (cellphone, landline, computer) | Communication access |
| `utilities_access` | `E1`-`E5` (water, power, cooking fuel, internet) | Infrastructure quality |
| `housing` | `F1`-`F4` (ownership, acquisition, construction, tenure) | Housing security |
| `social_protection` | `G1`-`G6` (SSS, GSIS, PhilHealth, life/health insurance) | Formal safety-net access |

**Computation** (`psm_indices.compute_indices`): each mapped item is z-scored using
bfar's own mean/std for that item (stored in `index_taxonomy.json` -- the mapping
asserts "this incoming column means the same as bfar canonical item X," so it inherits
X's standardization), then combined as a weighted mean using the raw baseline model's
feature importances, renormalized over whichever items are actually mapped and
present. An index with zero mapped items is imputed with bfar's median value for that
index (from `index_stats.json`) and listed in the response's `imputed_indices`, so a
consumer can see how much of a tier-2 score rests on real data versus imputation.

## Column matching: confidence-gated, abstain by default

bfar's canonical names are opaque survey codes (`D1.2:A_MOTORC`), so string similarity
against them directly is useless. Instead, `build_model.py` hand-curates a keyword
list per canonical item (`motorcycle`, `motorbike`, `motor`, ...), and
`column_matcher.match_columns` matches an incoming header by keyword containment
(quantity variants like `D1.2-A_QTY` require a count/qty/number token; flag variants
require the absence of one, so `owns_motorcycle` and `motorcycle_count` never
cross-match).

**Bijectivity rule:** a match is accepted only if exactly one canonical item claims a
given column and vice versa. Any tie drops both sides. This is the core safety
property of the whole system -- the matcher's only failure mode is *abstaining* on
something it could have matched, never confidently mapping the wrong thing.

## Automatic promotion: draft -> registered

Every tier-3 request runs the matcher and calls `mapping_store.record_observation`,
which advances a small state machine per program key:

1. **Consistency.** The identical mapping must reappear on
   `PROMOTION_CONSISTENT_UPLOADS` (3) consecutive uploads, covering at least
   `MIN_INDICES_COVERED` (4 of 6) indices. A different mapping, or a schema change
   (new column signature), resets the counter to 1 rather than accumulating across
   unrelated uploads.
2. **Sanity gate.** Once consistency is satisfied, the index values the *triggering*
   upload produces are checked against bfar's own distribution: each covered index's
   dataset-wide mean must fall within `SANITY_MAX_ABS_MEAN` (3.0) standard deviations
   of bfar's mean (indices are z-scored, so bfar itself sits at ~0). This is the
   backstop against a mapping that matched keywords confidently and consistently but
   is still reading the wrong thing -- e.g. a column matched as "income" that's
   actually recorded in a currency or scale bfar's stats don't reflect.
3. **Promote or hold.** Passing both gates writes a registered mapping and appends a
   `promoted` event to the audit log; failing the sanity gate withholds promotion,
   appends a `promotion_rejected_sanity` event with the offending index means, and
   the draft counter is left in place (not reset) so a one-off bad upload doesn't
   erase otherwise-consistent progress. Nothing re-triggers the gate automatically --
   the same draft simply gets re-checked against it the next time a consistent
   upload arrives.

Promotion only ever affects requests *after* the one that triggered it -- the
triggering request always completes on tier 3.

### Lifecycle states

| State | Stored at | Enters when | Leaves when |
|---|---|---|---|
| Unmapped | -- | Program first seen | Matcher produces >= 1 confident match on a tier-3 request |
| Draft | `mappings/drafts/<key>.json` | First confident match set | **Promoted** (both gates pass) or **reset** (mapping/schema changes between uploads) |
| Registered | `mappings/registered/<key>.json` | Promotion | `DELETE /mappings/<key>` -- removes the draft too, so a stale counter can't instantly re-promote it |

Every transition (draft reset, promotion, sanity rejection, demotion) is appended to
`mappings/audit.log` as JSONL, so a bad auto-promotion is auditable and reversible
after the fact even though nothing gates it beforehand.

## What persists vs. what doesn't

A mapping is metadata about column names -- `{canonical_item: incoming_column}` -- not
model weights and not any of the program's actual uploaded data. Registering a program
teaches the service how to *read* its columns; it never retrains, and neither frozen
model (`best_model.pkl`, `index_model.pkl`) is ever touched after `build_model.py`
produces them. `mappings/` is therefore small, gitignored, host-local runtime state,
distinct in kind from the persisted-model architecture this service moved away from
earlier.

## Repository structure

| File | Role |
|---|---|
| `app.py` | Flask service: tier routing, all endpoints, tier counters |
| `psm_core.py` | Scoring logic: `predict_dynamic` (tiers 1/3), `predict_with_index_model` (tier 2), matched-ATT, treatment detection, decision-support table |
| `psm_indices.py` | Folds mapped columns into the 6 composite indices; coverage counting |
| `column_matcher.py` | Keyword-based header matching with abstain-on-ambiguity |
| `mapping_store.py` | Draft/registered store, promotion state machine, audit log |
| `build_model.py` | Offline: trains both frozen models + taxonomy from `bfar.csv` |
| `mappings/` | Runtime mapping metadata (gitignored -- never model weights or data) |

## Frozen artifacts (`models/`, committed)

| Artifact | Used by | Contents |
|---|---|---|
| `best_model.pkl` / `scaler.pkl` | Tier 1 | GradientBoosting on the raw 57 features (scaler applied only for model types that need it) |
| `index_model.pkl` / `index_scaler.pkl` | Tier 2 | GradientBoosting on the 6 composite indices |
| `all/core/remaining_features.json` | Tiers 1 & 3 | The 57 features, top-30 core, remaining 27 |
| `index_taxonomy.json` | Tiers 2 & 3 | Per index: items with within-index weight, bfar mean/std, matching keywords |
| `index_stats.json` | Tier 2 + promotion sanity gate | Per-index bfar distribution (mean/std/median/min/max) |

## New API surface

- `GET /mappings` -- lists all registered mappings and in-flight drafts.
- `DELETE /mappings/<program_key>` -- demotes a program back to tier 3.
- `program_id` (optional field/form field on `/predict_ps`, `/estimate_att`,
  `/predict_ps_batch`) -- stable program identity; falls back to the column
  signature when omitted.
- Every scoring response now carries `tier` (1/2/3) and `program_key`; tier 2 adds
  `imputed_indices`; tier 3 adds `core_coverage` and `mapping_status`
  (`matched_items`, `indices_covered`, `draft_consistent_count`, `promoted`, and
  `sanity_rejected` when the gate blocked a promotion).
- `GET /health` additionally reports `mappings: {registered, drafts}` counts and
  `tier_requests_since_start`.

## Known limitation carried over from tier 3

Tier 3's fully data-driven fallback (when core features are absent) has no
leakage filtering -- unlike the old, now-removed `/train` pipeline, which excluded
candidate columns that were near-perfect proxies for the treatment column. A
dataset containing such a proxy can produce an overfit, poorly-matched ephemeral
model on tier 3. Restoring that filter (or porting it into
`select_flex_features`) is a reasonable follow-up if this becomes a real issue in
practice.

## Verification performed

- Raw-model regeneration reproduces `bfar_with_ps.csv` predictions to floating-point
  precision (`3.6e-17` mean absolute difference).
- A renamed-headers synthetic program (12 items, all 6 indices covered) promoted
  automatically on its 3rd upload; its 4th scored on tier 2 with no fitting,
  including a working matched-ATT (599 pairs, consistent effect direction).
- The same headers with values scaled 1000x (same schema, broken distributions)
  correctly triggered the sanity gate and stayed on tier 3, with the rejection
  recorded in the audit log.
- Fifteen fully unrecognizable headers (`q1`...`q15`) plus a treatment column
  scored successfully on the relaxed tier 3 (`core_coverage: 0.0`), where the old
  code returned a `400`.
- `DELETE /mappings/<key>` correctly demoted a promoted program back to tier 3.
- Existing tier-1 JSON `/predict_ps` and `/estimate_att` regression cases (555
  matched pairs, ATT 0.1946 on full bfar) still pass unchanged.
