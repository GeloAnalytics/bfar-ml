from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import time

import pandas as pd

import psm_core as core

app = Flask(__name__)
CORS(app)

# In-memory model state for the most recently uploaded dataset. Lost on
# restart -- callers must POST a CSV to /train again after the process
# restarts, or before the first prediction request.
STATE = {
    "model": None,
    "feature_cols": None,
    "importances": None,
    "treatment_col": None,
    "treatment_method": None,
    "source_filename": None,
    "rows": None,
    "trained_at": None,
}

MIN_TRAINING_ROWS = 10
TOP_N_FEATURES = 30


@app.route("/train", methods=["POST"])
def train():
    """
    multipart/form-data:
      file: <CSV file>               required
      treatment_column: <col name>   optional -- bypasses auto-detection if
                                      the heuristic picks the wrong column

    Trains a fresh propensity-score model on the uploaded dataset:
      1. Auto-detects the treatment/control column (see
         psm_core.detect_treatment_column for the heuristic), unless
         `treatment_column` was given explicitly.
      2. Fits a GradientBoostingClassifier on every numeric candidate column
         to rank feature importance for predicting treatment.
      3. Keeps the top 30 features and refits the final PS model on just
         those, replacing any previously trained model on this service.

    predict_ps / estimate_att on this service use whatever was trained by
    the most recent successful /train call.
    """
    file = request.files.get("file")
    if file is None:
        return jsonify({"error": "multipart form field 'file' (CSV) is required"}), 400

    try:
        df = pd.read_csv(file)
    except Exception as e:
        return jsonify({"error": f"could not parse CSV: {e}"}), 400

    if len(df) < MIN_TRAINING_ROWS:
        return jsonify({"error": f"dataset too small to train on (need at least {MIN_TRAINING_ROWS} rows, got {len(df)})"}), 400

    override_col = request.form.get("treatment_column") or None
    try:
        treatment_col, treatment_binarized, method = core.detect_treatment_column(df, override_col=override_col)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    if treatment_col is None:
        return jsonify({"error": "could not auto-detect a treatment/control column in this dataset; retry with a 'treatment_column' form field"}), 400

    try:
        top_features, _, excluded_leakage = core.select_top_features(df, treatment_col, treatment_binarized, top_n=TOP_N_FEATURES)
        model, final_importances = core.train_psm_model(df, treatment_binarized, top_features)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400

    STATE.update({
        "model": model,
        "feature_cols": top_features,
        "importances": final_importances,
        "treatment_col": treatment_col,
        "treatment_method": method,
        "source_filename": file.filename,
        "rows": len(df),
        "trained_at": time.time(),
    })

    ranked = sorted(final_importances.items(), key=lambda kv: kv[1], reverse=True)
    return jsonify({
        "status": "trained",
        "rows": len(df),
        "treatment_column": treatment_col,
        "treatment_detection_method": method,
        "n_features_selected": len(top_features),
        "top_features": [{"feature": name, "importance": imp} for name, imp in ranked],
        "excluded_as_leakage": excluded_leakage,
    })


@app.route("/predict_ps", methods=["POST"])
def predict_ps():
    if STATE["model"] is None:
        return jsonify({"error": "no model trained yet; POST a CSV to /train first"}), 409

    body = request.get_json(force=True, silent=True) or {}
    ps_final, err = core.predict_ps(
        STATE["model"], STATE["feature_cols"], body.get("records"),
        featureMap=body.get("featureMap"),
        auto_infer=core.as_bool(body.get("auto_infer"), default=True),
    )
    if err:
        return jsonify({"error": err}), 400
    return jsonify({"ps_final": ps_final})


@app.route("/estimate_att", methods=["POST"])
def estimate_att():
    if STATE["model"] is None:
        return jsonify({"error": "no model trained yet; POST a CSV to /train first"}), 409

    body = request.get_json(force=True, silent=True) or {}
    result, err = core.estimate_att(
        STATE["model"], STATE["feature_cols"], body.get("records"),
        featureMap=body.get("featureMap"),
        auto_infer=core.as_bool(body.get("auto_infer"), default=True),
        caliper_ratio=float(body.get("caliper_ratio", 0.2)),
        n_bootstrap=int(body.get("n_bootstrap", 500)),
        seed=int(body.get("seed", 42)),
        treatmentKey=body.get("treatmentKey", "treatment"),
        outcomeKey=body.get("outcomeKey", "outcome"),
    )
    if err:
        return jsonify({"error": err}), 400
    return jsonify(result)


@app.route("/health", methods=["GET"])
def health():
    if STATE["model"] is None:
        return jsonify({
            "status": "empty",
            "psm": None,
            "message": "no dataset trained yet; POST a CSV to /train",
        })

    ranked = sorted(STATE["importances"].items(), key=lambda kv: kv[1], reverse=True)
    return jsonify({
        "status": "ok",
        "psm": {
            "source": STATE["source_filename"],
            "rows": STATE["rows"],
            "trained_at": STATE["trained_at"],
            "treatment_column": STATE["treatment_col"],
            "treatment_detection_method": STATE["treatment_method"],
            "n_features_selected": len(STATE["feature_cols"]),
            "top_features": [{"feature": name, "importance": imp} for name, imp in ranked],
        },
    })


if __name__ == "__main__":
    # 0.0.0.0: reachable from other machines on the LAN (e.g. http://192.168.x.x:PORT),
    # unlike app.py's loopback-only static reference service.
    port = int(os.environ.get("DYNAMIC_PORT", "8001"))
    app.run(host="0.0.0.0", port=port, debug=False)
