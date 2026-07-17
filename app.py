from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import json
import joblib

import psm_core as core

app = Flask(__name__)
CORS(app)

# Paths to model artifacts. Override ML_MODEL_DIR when artifacts are mounted
# outside the repo, such as in production or a notebook export folder.
MODEL_DIR = os.environ.get("ML_MODEL_DIR", os.path.join(os.path.dirname(__file__), "models"))
MODEL_PATH = os.path.join(MODEL_DIR, "best_model.pkl")
SCALER_PATH = os.path.join(MODEL_DIR, "scaler.pkl")
ALL_FEATURES_PATH = os.path.join(MODEL_DIR, "all_features.json")


def load_artifacts():
    for path in (MODEL_PATH, SCALER_PATH, ALL_FEATURES_PATH):
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing model artifact: {path}")

    with open(ALL_FEATURES_PATH, "r") as f:
        all_features = json.load(f)

    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    return all_features, model, scaler


# Load on startup. This service always serves the artifacts already baked
# into models/ (trained on bfar.csv via build_model.py) -- it's the known-good
# baseline to check against, not a target for arbitrary datasets. See
# app_dynamic.py for the service that adapts to uploaded data.
try:
    ALL_FEATURES, MODEL, SCALER = load_artifacts()
    ARTIFACT_LOAD_ERROR = None
    NEEDS_SCALING = core.model_needs_scaling(MODEL)
    FEATURE_IMPORTANCES = {
        name: core.json_safe_float(imp)
        for name, imp in zip(ALL_FEATURES, MODEL.feature_importances_)
    } if hasattr(MODEL, "feature_importances_") else None
except Exception as e:
    ALL_FEATURES, MODEL, SCALER, NEEDS_SCALING, FEATURE_IMPORTANCES = None, None, None, False, None
    ARTIFACT_LOAD_ERROR = str(e)


@app.route("/predict_ps", methods=["POST"])
def predict_ps():
    if MODEL is None:
        return jsonify({"error": f"ML artifacts not loaded: {ARTIFACT_LOAD_ERROR}"}), 500

    body = request.get_json(force=True, silent=True) or {}
    ps_final, err = core.predict_ps(
        MODEL, ALL_FEATURES, body.get("records"),
        featureMap=body.get("featureMap"),
        auto_infer=core.as_bool(body.get("auto_infer"), default=True),
        scaler=SCALER if NEEDS_SCALING else None,
    )
    if err:
        return jsonify({"error": err}), 400
    return jsonify({"ps_final": ps_final})


@app.route("/estimate_att", methods=["POST"])
def estimate_att():
    """
    Input (supports dynamic feature mapping):
    {
      "records": [
        {
          "treatment": 0/1,
          "outcome": number,
          "features": { "<incoming_key>": value, ... }   # optional; if omitted, expects flat payload
        }
      ],
      "featureMap": { "<model_feature>": "<incoming_key>" },  # optional
      "auto_infer": true,                                        # optional default true
      "caliper_ratio": 0.2,                                      # optional default 0.2
      "n_bootstrap": 500,                                        # optional default 500
      "seed": 42,                                                # optional default 42
      "treatmentKey": "treatment",                               # optional
      "outcomeKey": "outcome"                                    # optional
    }
    """
    if MODEL is None:
        return jsonify({"error": f"ML artifacts not loaded: {ARTIFACT_LOAD_ERROR}"}), 500

    body = request.get_json(force=True, silent=True) or {}
    result, err = core.estimate_att(
        MODEL, ALL_FEATURES, body.get("records"),
        featureMap=body.get("featureMap"),
        auto_infer=core.as_bool(body.get("auto_infer"), default=True),
        caliper_ratio=float(body.get("caliper_ratio", 0.2)),
        n_bootstrap=int(body.get("n_bootstrap", 500)),
        seed=int(body.get("seed", 42)),
        treatmentKey=body.get("treatmentKey", "treatment"),
        outcomeKey=body.get("outcomeKey", "outcome"),
        scaler=SCALER if NEEDS_SCALING else None,
    )
    if err:
        return jsonify({"error": err}), 400
    return jsonify(result)


@app.route("/health", methods=["GET"])
def health():
    if MODEL is None:
        return jsonify({
            "status": "degraded",
            "artifact_error": ARTIFACT_LOAD_ERROR,
            "model_dir": MODEL_DIR
        }), 500

    response = {
        "status": "ok",
        "model_dir": MODEL_DIR,
        "psm": {
            "source": "bfar.csv (baked-in, static)",
            "model_type": type(MODEL).__name__,
            "n_features_total": len(ALL_FEATURES),
        },
    }
    if FEATURE_IMPORTANCES is not None:
        top_features = sorted(FEATURE_IMPORTANCES.items(), key=lambda kv: kv[1], reverse=True)[:30]
        response["psm"]["top_features"] = [{"feature": name, "importance": imp} for name, imp in top_features]
    return jsonify(response)


if __name__ == "__main__":
    # Loopback-only: this service is the fixed bfar.csv reference, not meant
    # to be reachable off-box. See app_dynamic.py for the LAN-facing service.
    port = int(os.environ.get("STATIC_PORT", os.environ.get("FLASK_PORT", "8000")))
    app.run(host="127.0.0.1", port=port, debug=False)
