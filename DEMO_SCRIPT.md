# Integrator Demo Script

A live walkthrough for demoing this service to the integrator. Everything below was
run end-to-end against `demo_training_data.csv` (a real 180-row, 14-column subset of
`bfar.csv` -- committed alongside this script) before writing it down, so it's
copy-pasteable as-is.

## Before you start

```bash
python app.py
```
You should see **two** `Running on...` blocks in the log -- one for `8000`, one for
`8001`. If you only see one, something didn't start; don't proceed until both are up
(`curl http://localhost:8000/health` and `curl http://localhost:8001/health` should
both return `200`).

If this isn't the very first run, the dynamic model may already have something
trained into it from prior testing. Either is fine to demo with, but for a truly
clean "watch it learn from nothing" opening, stop the server and delete
`models/dynamic/` first.

Open `http://localhost:8000/` in a browser too -- that's `test_ui.html`, useful for
the parts of this demo you want to click through instead of type.

## The pitch, in one breath

*"This is one ML service exposed as two ports. Port 8001 is a fixed reference model
we can't break -- point-and-shoot, always available, no training step. Port 8000 is
what makes this reusable across programs that aren't BFAR's boat-repair survey: POST
any CSV with a detectable treatment/control column, it trains a fresh propensity-score
model and hands back not just predictions, but the balance diagnostics and causal
estimate a data scientist would normally have to compute by hand."*

## 1. Two ports, two contracts

```bash
curl http://localhost:8000/health
curl http://localhost:8001/health
```
Point out: both report the same frozen baseline. `:8000`'s response also has a
`dynamic` section (empty until you train); `:8001` doesn't have one at all -- it never
trains anything.

```bash
curl -X POST http://localhost:8001/predict_ps -H "Content-Type: application/json" -d '{"records":[{"foo":1}]}'
```
Static port on a payload that isn't the full 57-feature baseline schema -> clean `409`
telling you exactly what's missing. *"This port has one job and refuses to guess."*

## 2. Train the dynamic model on a dataset it's never seen

`demo_training_data.csv` is real BFAR respondent data, but only 14 of the 215 raw
columns -- deliberately *not* the 57-feature baseline schema, so this forces the
dynamic path instead of the frozen baseline.

```bash
curl -X POST http://localhost:8000/train -F "file=@demo_training_data.csv"
```

Walk through the response section by section -- this is the core of the demo:

- **`treatment_column` / `treatment_detection_method`**: `"Y_BOAT-RE"` /
  `"notna_mask"` -- it found the treatment indicator on its own (populated only for
  boat-repair program participants, blank otherwise) without being told which column
  it was.
- **`feature_selection.selected`**: every usable numeric column, ranked by importance
  for predicting treatment, no arbitrary cutoff -- what to actually use downstream is
  the integrator's call, not baked into the service.
- **`ps_output`**: propensity scores for these 180 rows, plus min/max/mean/median.
- **`covariate_balance`**: standardized mean difference per feature before/after
  matching, PS overlap between groups, and a `balance_achieved` verdict. *"If the
  groups don't look statistically comparable, you'll see it here -- not find out three
  steps later that your estimate was garbage."* (On this real demo data,
  `balance_achieved` often comes back `false` -- that's expected and worth pointing
  out honestly: BFAR program participants and non-participants really are different
  populations, the diagnostic is doing its job by saying so.)
- **`model_interpretation`**: which features actually drove the model, explicitly
  labeled as feature importance, not SHAP.
- **`decision_support`**: rows bucketed into PS quartiles with a plain-English
  interpretation per bucket -- *"this is the table a program officer would actually
  read."*

## 3. Re-upload the exact same file -- show the retrain-skip

```bash
curl -X POST http://localhost:8000/train -F "file=@demo_training_data.csv"
```
Point at `"retrained": false`. *"Same columns as what's already active -- it
recognized that and just re-scored against the existing model instead of refitting
from scratch. Upload a CSV with even one column different and it retrains fully."*

## 4. Manual treatment-column override, from the browser

In `http://localhost:8000/` (test UI): pick `demo_training_data.csv` in the file
input under "1. Train on a new dataset" -- the "Treatment column" dropdown populates
from the file's actual header row. Pick something other than `Y_BOAT-RE` (e.g.
`D2.6:A_FRIDGE`) and hit Train. Response comes back with
`treatment_detection_method: "manual_override"` for that column. *"If the
auto-detection heuristic guesses wrong on someone else's data, this is the escape
hatch -- and you don't have to already know the exact column name to use it."*

Re-select `Y_BOAT-RE` and retrain once more afterward so the active model is back to
the real treatment column before continuing.

## 5. Score against the trained model

```bash
curl -X POST http://localhost:8000/train/predict_ps_batch -F "file=@demo_training_data.csv"
```
`source: "dynamic"` confirms it's scoring against what was just trained, not the
frozen baseline.

```bash
curl -X POST http://localhost:8000/train/estimate_att \
  -H "Content-Type: application/json" \
  -d '{"records": [ /* rows with treatment + outcome per record */ ], "caliper_ratio": 0.2}'
```
Note for the integrator: `estimate_att` needs a real `treatment` + `outcome` per
record and enough rows for matching to find pairs -- a 2-row toy example will
legitimately come back with `matched_pairs: 0`. Use a realistic batch size here, or
point them at the worked example in `README.md`'s "Estimate ATT via matching" section
instead of typing one live.

## 6. Wrap-up talking points

- Error contract is consistent everywhere: `400` bad input, `409` no model applies,
  `500` startup/artifact problem -- see `README.md`'s Integration guide section.
- `CURRENT_ML_MODEL.md` is the living reference for exactly what's implemented right
  now, mapped against the propensity-score pipeline stages.
- `DYNAMIC_TRAINING.md` has the full design rationale for the delete-and-retrain
  approach, retrain-skip, and the covariate-balance re-tune loop, if they want the
  "why," not just the "what."
