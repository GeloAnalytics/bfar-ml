# ML Flask Service (Propensity Score Matching)

This service exposes endpoints that load artifacts generated from `psm.ipynb`:

- `models/pre_features.json`
- `models/gradient_boosting_ps_model.pkl`

It provides:
- `POST /predict_ps` : compute `ps_final` (propensity score) for input records
- `POST /estimate_att` : run propensity-score matching and estimate ATT + CI

If this repo is dropped into another project as a subfolder (commonly named `ml/`), prefix the commands below with that folder (e.g. `pip install -r ml/requirements.txt`, `python ml/app.py`). All paths in `app.py`, `build_model.py`, and `train_model.py` resolve relative to their own file location, so it works the same whether this is the repo root or a nested subfolder — `bfar.csv` just needs to stay next to `build_model.py`/`train_model.py`.

## 1) Model artifacts

`models/gradient_boosting_ps_model.pkl` and `models/pre_features.json` are already committed to this repo, so **you can skip straight to step 2** for normal use.

To regenerate them yourself (e.g. after changing `bfar.csv` or the feature list):
```bash
pip install -r requirements-dev.txt
python build_model.py
```
This reads `bfar.csv` (same directory) and overwrites the two files in `models/`.

Alternatively, export updated artifacts from `psm.ipynb` (or `train_model.py`) into `models/`.

## 2) Install Python dependencies

```bash
pip install -r requirements.txt
```

## 3) Start the Flask service

```bash
python app.py
```

Default port: **8000** (same as controller default).

## 4) API examples

### Health
```bash
curl http://localhost:8000/health
```

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
