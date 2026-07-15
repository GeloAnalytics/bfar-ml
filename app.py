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
MODEL_PATH = os.path.join(MODEL_DIR, "gradient_boosting_ps_model.pkl")
FEATURES_PATH = os.path.join(MODEL_DIR, "pre_features.json")


def load_artifacts():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Missing model artifact: {MODEL_PATH}")
    if not os.path.exists(FEATURES_PATH):
        raise FileNotFoundError(f"Missing feature list artifact: {FEATURES_PATH}")

    with open(FEATURES_PATH, "r") as f:
        pre_features = json.load(f)

    model = joblib.load(MODEL_PATH)
    return pre_features, model


# Load on startup. This service always serves the artifacts already baked
# into models/ (trained on bfar.csv) -- it's the known-good reference to
# check against, not a target for arbitrary datasets. See app_dynamic.py
# for the service that trains on uploaded data.
try:
    PRE_FEATURES, GB_MODEL = load_artifacts()
    ARTIFACT_LOAD_ERROR = None
    FEATURE_IMPORTANCES = {
        name: core.json_safe_float(imp)
        for name, imp in zip(PRE_FEATURES, GB_MODEL.feature_importances_)
    }
except Exception as e:
    PRE_FEATURES, GB_MODEL, FEATURE_IMPORTANCES = None, None, None
    ARTIFACT_LOAD_ERROR = str(e)


@app.route("/predict_ps", methods=["POST"])
def predict_ps():
    if GB_MODEL is None:
        return jsonify({"error": f"ML artifacts not loaded: {ARTIFACT_LOAD_ERROR}"}), 500

    body = request.get_json(force=True, silent=True) or {}
    ps_final, err = core.predict_ps(
        GB_MODEL, PRE_FEATURES, body.get("records"),
        featureMap=body.get("featureMap"),
        auto_infer=core.as_bool(body.get("auto_infer"), default=True),
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
    if GB_MODEL is None:
        return jsonify({"error": f"ML artifacts not loaded: {ARTIFACT_LOAD_ERROR}"}), 500

    body = request.get_json(force=True, silent=True) or {}
    result, err = core.estimate_att(
        GB_MODEL, PRE_FEATURES, body.get("records"),
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
    if GB_MODEL is None:
        return jsonify({
            "status": "degraded",
            "artifact_error": ARTIFACT_LOAD_ERROR,
            "model_dir": MODEL_DIR
        }), 500

    top_features = sorted(FEATURE_IMPORTANCES.items(), key=lambda kv: kv[1], reverse=True)[:30]
    return jsonify({
        "status": "ok",
        "model_dir": MODEL_DIR,
        "psm": {
            "source": "bfar.csv (baked-in, static)",
            "n_features_total": len(PRE_FEATURES),
            "top_features": [{"feature": name, "importance": imp} for name, imp in top_features],
        },
    })


if __name__ == "__main__":
    # Loopback-only: this service is the fixed bfar.csv reference, not meant
    # to be reachable off-box. See app_dynamic.py for the LAN-facing service.
    port = int(os.environ.get("STATIC_PORT", os.environ.get("FLASK_PORT", "8000")))
    app.run(host="127.0.0.1", port=port, debug=False)
