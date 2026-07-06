# ML Flask Service (Propensity Score Matching)

This service exposes endpoints that load artifacts generated from `psm.ipynb`:

- `ml/models/pre_features.json`
- `ml/models/gradient_boosting_ps_model.pkl`

It provides:
- `POST /predict_ps` : compute `ps_final` (propensity score) for input records
- `POST /estimate_att` : run propensity-score matching and estimate ATT + CI

## 1) Create / place model artifacts

You can generate the required ML artifacts (`pre_features.json` and `gradient_boosting_ps_model.pkl`) automatically by running the included python scripts:
- `cd ml`
- `python build_model.py`

This script reads `bfar.csv` from the parent directory and saves the generated model and feature list directly into the `ml/models/` directory.

Alternatively, from `psm.ipynb` (or `ml/train_model.py`), export the artifacts into `ml/models/`. If you already saved these somewhere else, copy them into `ml/models/`.

## 2) Install Python dependencies

From repo root, install:
```bash
pip install -r ml/requirements.txt
```

## 3) Start the Flask service

```bash
python ml/app.py
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

**Note:** Every record must include *all* `pre_features` keys found in `ml/models/pre_features.json`, plus:
- `treatment`: 0 or 1
- `outcome`: numeric outcome value (in the notebook it was `C5:TOT_INCOME/B`)
