from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import json

import joblib
import numpy as np
import pandas as pd

import psm_core as core

app = Flask(__name__)
CORS(app)

# Same baseline artifacts as app.py -- this service never trains or persists
# a model of its own. Every request re-anchors to this frozen bfar.csv
# baseline; see psm_core.predict_dynamic for what happens when an upload
# doesn't cover every baseline feature.
MODEL_DIR = os.environ.get("ML_MODEL_DIR", os.path.join(os.path.dirname(__file__), "models"))
MODEL_PATH = os.path.join(MODEL_DIR, "best_model.pkl")
SCALER_PATH = os.path.join(MODEL_DIR, "scaler.pkl")
ALL_FEATURES_PATH = os.path.join(MODEL_DIR, "all_features.json")
CORE_FEATURES_PATH = os.path.join(MODEL_DIR, "core_features.json")
REMAINING_FEATURES_PATH = os.path.join(MODEL_DIR, "remaining_features.json")

KEY_SUPPORT_FEATURES = ["D1.2:A_MOTORC", "D2.1:A_TV", "D2.6:A_FRIDGE", "E3:A_POWER-SUP", "F1:A_HOUSE-OWN"]


def load_artifacts():
    for path in (MODEL_PATH, SCALER_PATH, ALL_FEATURES_PATH, CORE_FEATURES_PATH, REMAINING_FEATURES_PATH):
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing model artifact: {path}")

    with open(ALL_FEATURES_PATH, "r") as f:
        all_features = json.load(f)
    with open(CORE_FEATURES_PATH, "r") as f:
        core_features = json.load(f)
    with open(REMAINING_FEATURES_PATH, "r") as f:
        remaining_features = json.load(f)

    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    return all_features, core_features, remaining_features, model, scaler


try:
    ALL_FEATURES, CORE_FEATURES, REMAINING_FEATURES, BASELINE_MODEL, BASELINE_SCALER = load_artifacts()
    ARTIFACT_LOAD_ERROR = None
except Exception as e:
    ALL_FEATURES, CORE_FEATURES, REMAINING_FEATURES, BASELINE_MODEL, BASELINE_SCALER = None, None, None, None, None
    ARTIFACT_LOAD_ERROR = str(e)


def _resolve_treatment_column(df, override_col=None):
    """Best-effort treatment/control column resolution (see
    psm_core.detect_treatment_column); binarizes it in place on a copy.
    Returns (df, column_name_or_None, method_or_None)."""
    col, binarized, method = core.detect_treatment_column(df, override_col=override_col)
    if col is None:
        return df, None, None
    df = df.copy()
    df[col] = binarized
    return df, col, method


def _default_n_flex():
    return len(REMAINING_FEATURES)


@app.route("/predict_ps", methods=["POST"])
def predict_ps():
    """
    JSON body:
    {
      "records": [ { "<feature_or_column_name>": value, ... }, ... ],
      "treatment_column": "<name>",   # optional -- only needed if records don't
                                       # cover all baseline features, and auto-
                                       # detection picks the wrong column
      "n_flex": 27                    # optional, default = all "remaining" features
    }

    If every record together covers all of the bfar.csv baseline features,
    scores directly against the frozen baseline model (no fitting at all).
    Otherwise requires the 30 core baseline features plus a treatment column
    (to rank this dataset's own extra features against) and at least 10
    records; fits a throwaway model for this request only.
    """
    if BASELINE_MODEL is None:
        return jsonify({"error": f"ML artifacts not loaded: {ARTIFACT_LOAD_ERROR}"}), 500

    body = request.get_json(force=True, silent=True) or {}
    records = body.get("records")
    if not isinstance(records, list) or len(records) == 0:
        return jsonify({"error": "records must be a non-empty array"}), 400

    df = pd.DataFrame(records)
    df, treatment_col, treatment_method = _resolve_treatment_column(df, override_col=body.get("treatment_column"))

    try:
        result = core.predict_dynamic(
            df, CORE_FEATURES, ALL_FEATURES, BASELINE_MODEL, BASELINE_SCALER,
            treatment_col=treatment_col or "treatment",
            n_flex=int(body.get("n_flex", _default_n_flex())),
        )
    except ValueError as e:
        return jsonify({"error": str(e)}), 400

    return jsonify({
        "ps_final": [core.json_safe_float(v) for v in result["ps"]],
        "used_baseline": result["used_baseline"],
        "n_features_used": len(result["final_features"]),
        "final_features": result["final_features"],
        "treatment_column": treatment_col,
        "treatment_detection_method": treatment_method,
    })


@app.route("/estimate_att", methods=["POST"])
def estimate_att():
    """
    JSON body (records must include treatment + outcome):
    {
      "records": [ { "<feature_or_column_name>": value, ..., "treatment": 0/1, "outcome": number }, ... ],
      "treatment_column": "<name>",   # optional -- auto-detected otherwise
      "outcomeKey": "outcome",        # optional
      "n_flex": 27,                   # optional
      "caliper_ratio": 0.2, "n_bootstrap": 500, "seed": 42   # optional
    }
    """
    if BASELINE_MODEL is None:
        return jsonify({"error": f"ML artifacts not loaded: {ARTIFACT_LOAD_ERROR}"}), 500

    body = request.get_json(force=True, silent=True) or {}
    records = body.get("records")
    if not isinstance(records, list) or len(records) == 0:
        return jsonify({"error": "records must be a non-empty array"}), 400

    df = pd.DataFrame(records)
    outcome_key = body.get("outcomeKey", "outcome")
    override_col = body.get("treatment_column") or body.get("treatmentKey")
    df, treatment_col, treatment_method = _resolve_treatment_column(df, override_col=override_col)

    if treatment_col is None:
        return jsonify({"error": "could not auto-detect a treatment/control column in this dataset; retry with a 'treatment_column' field"}), 400
    if outcome_key not in df.columns:
        return jsonify({"error": f"missing required outcome column: {outcome_key}"}), 400

    result, err = core.estimate_att_dynamic(
        df, CORE_FEATURES, ALL_FEATURES, BASELINE_MODEL, BASELINE_SCALER,
        treatment_col=treatment_col, outcome_col=outcome_key,
        n_flex=int(body.get("n_flex", _default_n_flex())),
        caliper_ratio=float(body.get("caliper_ratio", 0.2)),
        n_bootstrap=int(body.get("n_bootstrap", 500)),
        seed=int(body.get("seed", 42)),
    )
    if err:
        return jsonify({"error": err}), 400

    result["treatment_column"] = treatment_col
    result["treatment_detection_method"] = treatment_method
    return jsonify(result)


@app.route("/predict_ps_batch", methods=["POST"])
def predict_ps_batch():
    """
    multipart/form-data:
      file: <CSV file>               required
      treatment_column: <col name>   optional -- bypasses auto-detection
      n_flex: 27                     optional

    Whole-dataset counterpart to /predict_ps (mirrors predictor_psm.ipynb's
    predict_and_support workflow): scores every row and returns a
    decision-support table stratified by propensity-score quartile, in
    addition to the raw per-row scores. Nothing about the baseline is ever
    changed by this call.
    """
    if BASELINE_MODEL is None:
        return jsonify({"error": f"ML artifacts not loaded: {ARTIFACT_LOAD_ERROR}"}), 500

    file = request.files.get("file")
    if file is None:
        return jsonify({"error": "multipart form field 'file' (CSV) is required"}), 400

    try:
        df = pd.read_csv(file)
    except Exception as e:
        return jsonify({"error": f"could not parse CSV: {e}"}), 400
    if len(df) == 0:
        return jsonify({"error": "uploaded CSV has no rows"}), 400

    override_col = request.form.get("treatment_column") or None
    df, treatment_col, treatment_method = _resolve_treatment_column(df, override_col=override_col)

    try:
        result = core.predict_dynamic(
            df, CORE_FEATURES, ALL_FEATURES, BASELINE_MODEL, BASELINE_SCALER,
            treatment_col=treatment_col or "treatment",
            n_flex=int(request.form.get("n_flex", _default_n_flex())),
        )
    except ValueError as e:
        return jsonify({"error": str(e)}), 400

    ps = result["ps"]
    key_features = [f for f in KEY_SUPPORT_FEATURES if f in df.columns]
    support_input = df[key_features].copy() if key_features else pd.DataFrame(index=df.index)
    support_input["ps"] = ps
    support_table = core.decision_support_table(support_input, key_features=key_features, ps_col="ps")

    decision_support = []
    for _, row in support_table.iterrows():
        rec = {"ps_group": str(row["ps_group"]), "count": int(row["Count"]), "interpretation": row["Interpretation"]}
        for col in support_table.columns:
            if col.startswith("Mean_"):
                rec[col.lower()] = core.json_safe_float(row[col])
        decision_support.append(rec)

    return jsonify({
        "rows": len(df),
        "used_baseline": result["used_baseline"],
        "n_features_used": len(result["final_features"]),
        "final_features": result["final_features"],
        "treatment_column": treatment_col,
        "treatment_detection_method": treatment_method,
        "ps": [core.json_safe_float(v) for v in ps],
        "ps_logit": [core.json_safe_float(v) for v in result["ps_logit"]],
        "ps_summary": {
            "min": core.json_safe_float(np.min(ps)),
            "max": core.json_safe_float(np.max(ps)),
            "mean": core.json_safe_float(np.mean(ps)),
            "median": core.json_safe_float(np.median(ps)),
        },
        "decision_support": decision_support,
    })


@app.route("/health", methods=["GET"])
def health():
    if BASELINE_MODEL is None:
        return jsonify({
            "status": "degraded",
            "artifact_error": ARTIFACT_LOAD_ERROR,
            "model_dir": MODEL_DIR,
        }), 500

    return jsonify({
        "status": "ok",
        "model_dir": MODEL_DIR,
        "psm": {
            "source": "bfar.csv baseline (models/best_model.pkl) -- adapted per-request for uploads that don't cover all baseline features, never persisted",
            "model_type": type(BASELINE_MODEL).__name__,
            "n_core_features": len(CORE_FEATURES),
            "n_remaining_features": len(REMAINING_FEATURES),
            "n_all_features": len(ALL_FEATURES),
        },
    })


if __name__ == "__main__":
    # 0.0.0.0: reachable from other machines on the LAN (e.g. http://192.168.x.x:PORT),
    # unlike app.py's loopback-only static reference service.
    port = int(os.environ.get("DYNAMIC_PORT", "8001"))
    app.run(host="0.0.0.0", port=port, debug=False)
