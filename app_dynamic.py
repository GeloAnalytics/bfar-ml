from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import json
import time

import joblib
import pandas as pd

import psm_core as core

app = Flask(__name__)
CORS(app)

STATE_DIR = os.environ.get("ML_DYNAMIC_STATE_DIR", os.path.join(os.path.dirname(__file__), "models", "dynamic"))
STATE_MODEL_PATH = os.path.join(STATE_DIR, "model.pkl")
STATE_META_PATH = os.path.join(STATE_DIR, "meta.json")

# In-memory model state, mirrored to STATE_DIR on disk so an evergreen
# schema survives process restarts instead of needing a fresh /train call
# every time. See load_state()/save_state().
STATE = {
    "model": None,
    "feature_cols": None,
    "importances": None,
    "treatment_col": None,
    "treatment_method": None,
    "source_filename": None,
    "rows": None,
    "trained_at": None,
    "last_action": None,
}

MIN_TRAINING_ROWS = 10
TOP_N_FEATURES = 30
# If an upload already contains at least this fraction of the current
# schema's feature columns, we reuse the existing model outright instead of
# retraining -- the common case is the same survey/export uploaded again
# with new rows, not a structurally different dataset.
MIN_REUSE_COVERAGE = 0.9


def load_state():
    if not (os.path.exists(STATE_MODEL_PATH) and os.path.exists(STATE_META_PATH)):
        return
    try:
        model = joblib.load(STATE_MODEL_PATH)
        with open(STATE_META_PATH, "r") as f:
            meta = json.load(f)
    except Exception:
        return
    STATE.update(meta)
    STATE["model"] = model


def save_state():
    os.makedirs(STATE_DIR, exist_ok=True)
    joblib.dump(STATE["model"], STATE_MODEL_PATH)
    meta = {k: v for k, v in STATE.items() if k != "model"}
    with open(STATE_META_PATH, "w") as f:
        json.dump(meta, f)


load_state()


@app.route("/train", methods=["POST"])
def train():
    """
    multipart/form-data:
      file: <CSV file>               required
      treatment_column: <col name>   optional -- bypasses auto-detection if
                                      the heuristic picks the wrong column
      force_retrain: "true"          optional -- skip the reuse shortcut below
                                      and retrain even if coverage is high

    Behavior:
      - If a model is already active and this upload contains at least
        MIN_REUSE_COVERAGE of its feature columns (by exact or normalized
        name), no training happens at all -- the existing model is reused
        as-is ("status": "reused"). This is the expected path for repeat
        uploads of the same survey/export with new rows.
      - Otherwise, trains a fresh model: auto-detects the treatment/control
        column (see psm_core.detect_treatment_column), ranks every numeric
        candidate column by importance for predicting it, and selects the
        final feature set via psm_core.select_or_merge_features -- keeping
        whichever features from the *previous* schema are still usable and
        backfilling only the freed-up slots with this dataset's own top
        scorers, rather than discarding the previous schema outright.

    predict_ps / estimate_att on this service use whatever model is
    currently active, persisted under models/dynamic/ so it survives a
    process restart.
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

    force_retrain = core.as_bool(request.form.get("force_retrain"), default=False)

    if STATE["model"] is not None and not force_retrain:
        matches = core.match_feature_columns(STATE["feature_cols"], df.columns)
        matched = [name for name, col in matches.items() if col is not None]
        coverage = len(matched) / len(STATE["feature_cols"])
        if coverage >= MIN_REUSE_COVERAGE:
            STATE["last_action"] = "reused"
            save_state()
            return jsonify({
                "status": "reused",
                "reason": f"upload already covers {len(matched)}/{len(STATE['feature_cols'])} of the active schema's features; reusing the existing model instead of retraining",
                "coverage": core.json_safe_float(coverage),
                "treatment_column": STATE["treatment_col"],
                "n_features_selected": len(STATE["feature_cols"]),
                "unmatched_features": [name for name, col in matches.items() if col is None],
            })

    override_col = request.form.get("treatment_column") or None
    try:
        treatment_col, treatment_binarized, method = core.detect_treatment_column(df, override_col=override_col)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    if treatment_col is None:
        return jsonify({"error": "could not auto-detect a treatment/control column in this dataset; retry with a 'treatment_column' form field"}), 400

    previous_features = STATE["feature_cols"]
    try:
        top_features, _, excluded_leakage, breakdown = core.select_or_merge_features(
            df, treatment_col, treatment_binarized, previous_features=previous_features, top_n=TOP_N_FEATURES,
        )
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
        "last_action": "trained",
    })
    save_state()

    ranked = sorted(final_importances.items(), key=lambda kv: kv[1], reverse=True)
    return jsonify({
        "status": "trained",
        "rows": len(df),
        "treatment_column": treatment_col,
        "treatment_detection_method": method,
        "n_features_selected": len(top_features),
        "top_features": [{"feature": name, "importance": imp} for name, imp in ranked],
        "excluded_as_leakage": excluded_leakage,
        "kept_from_previous_schema": breakdown["kept_from_previous"],
        "added_new": breakdown["added_new"],
        "dropped_from_previous_schema": breakdown["dropped_from_previous"],
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
            "last_action": STATE["last_action"],
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
