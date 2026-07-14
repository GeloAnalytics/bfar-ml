from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import json
import joblib
import numpy as np
from sklearn.neighbors import NearestNeighbors
from scipy.stats import ttest_rel

app = Flask(__name__)
CORS(app)

# Paths to model artifacts. Override ML_MODEL_DIR when artifacts are mounted
# outside the repo, such as in production or a notebook export folder.
MODEL_DIR = os.environ.get("ML_MODEL_DIR", os.path.join(os.path.dirname(__file__), "models"))
MODEL_PATH = os.path.join(MODEL_DIR, "gradient_boosting_ps_model.pkl")
FEATURES_PATH = os.path.join(MODEL_DIR, "pre_features.json")


def _artifact_error(message: str, status_code: int = 500):
    return jsonify({"error": message}), status_code


def load_artifacts():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Missing model artifact: {MODEL_PATH}")
    if not os.path.exists(FEATURES_PATH):
        raise FileNotFoundError(f"Missing feature list artifact: {FEATURES_PATH}")

    with open(FEATURES_PATH, "r") as f:
        pre_features = json.load(f)

    model = joblib.load(MODEL_PATH)
    return pre_features, model


# Load on startup
try:
    PRE_FEATURES, GB_MODEL = load_artifacts()
except Exception as e:
    PRE_FEATURES = None
    GB_MODEL = None
    ARTIFACT_LOAD_ERROR = str(e)


def _normalize_key(s: str) -> str:
    import re
    s = str(s).lower()
    s = re.sub(r"[^a-z0-9]+", "", s)
    return s


def _infer_feature_map(incoming_feature_keys):
    """
    Auto-infer mapping from model feature names (PRE_FEATURES)
    to incoming keys using normalized-string similarity.

    Returns: dict model_feature -> incoming_key
    """
    if not PRE_FEATURES:
        return {}

    # Build lookup from normalized incoming keys
    incoming_list = list(incoming_feature_keys)
    incoming_norm = {_normalize_key(k): k for k in incoming_list}
    norm_values = list(incoming_norm.keys())

    from difflib import SequenceMatcher

    mapping = {}
    for mf in PRE_FEATURES:
        mf_norm = _normalize_key(mf)
        # Exact normalized match first
        if mf_norm in incoming_norm:
            mapping[mf] = incoming_norm[mf_norm]
            continue

        # Fuzzy match
        best = None
        best_score = 0.0
        for in_norm in norm_values:
            score = SequenceMatcher(None, mf_norm, in_norm).ratio()
            if score > best_score:
                best_score = score
                best = incoming_norm[in_norm]

        # Threshold to avoid crazy matches
        if best is not None and best_score >= 0.72:
            mapping[mf] = best

    return mapping


def _extract_value(r, key):
    if isinstance(r, dict) and key in r:
        return r[key]
    return None


def _as_bool(value, default=True):
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() not in {"false", "0", "no", "off"}
    return bool(value)


def validate_records(records, require_treatment=False, require_outcome=False, treatmentKey="treatment", outcomeKey="outcome"):
    if not isinstance(records, list) or len(records) == 0:
        return "records must be a non-empty array"

    required_feature_set = set(PRE_FEATURES or [])
    for i, r in enumerate(records):
        if not isinstance(r, dict):
            return f"record at index {i} must be an object"

        if require_treatment and treatmentKey not in r:
            return f"record at index {i} missing required field: {treatmentKey}"
        if require_outcome and outcomeKey not in r:
            return f"record at index {i} missing required field: {outcomeKey}"

        # Feature storage supports:
        # - flat record: { "<model_feature>": value, ... }
        # - nested record.features: { "<incoming_key>": value, ... }
        if "features" in r and isinstance(r["features"], dict):
            incoming_keys = set(r["features"].keys())
        else:
            # treat flat record as incoming keys
            incoming_keys = set(r.keys())

        missing = []
        for mf in required_feature_set:
            if "features" in r and isinstance(r["features"], dict):
                # here we only validate existence by model-feature name,
                # actual mapping will be computed later (featureMap/auto-infer).
                # So only error if neither flat nor features contains mf.
                if mf not in incoming_keys:
                    missing.append(mf)
            else:
                if mf not in r:
                    missing.append(mf)

        # Only hard-error for flat payloads; mapped payloads validated later.
        if missing and not ("features" in r and isinstance(r["features"], dict)):
            return f"record at index {i} missing features: {missing[:5]}{'...' if len(missing) > 5 else ''}"

    return None


def build_X_from_records(records, featureMap=None, auto_infer=True, treatmentKey="treatment", outcomeKey="outcome"):
    """
    Produces X matrix aligned with PRE_FEATURES.

    featureMap: optional dict { model_feature_name: incoming_key_name }
    record payload:
      - flat: record[model_feature_name] = value
      - or nested: record.features[incoming_key_name] = value
    """
    # Determine incoming keys from first record
    first = records[0]
    if isinstance(first, dict) and "features" in first and isinstance(first["features"], dict):
        incoming_feature_keys = set(first["features"].keys())
        nested_features = True
    else:
        incoming_feature_keys = set(first.keys())
        nested_features = False

    fmap = featureMap or {}
    if (not fmap) and auto_infer:
        fmap = _infer_feature_map(incoming_feature_keys)

    # Validate mapping coverage
    missing_model_features = [mf for mf in PRE_FEATURES if mf not in fmap]
    if missing_model_features:
        # Allow flat payload: if nested_features is false, we can fetch directly by mf
        if nested_features:
            return None, f"Could not map model features missing: {missing_model_features[:10]}{'...' if len(missing_model_features) > 10 else ''}"
        # flat payload fallback
        fmap = {mf: mf for mf in PRE_FEATURES}

    X = []
    for r in records:
        row = []
        features_obj = r.get("features", {}) if isinstance(r, dict) else {}
        for mf in PRE_FEATURES:
            incoming_key = fmap[mf]
            if nested_features:
                if incoming_key not in features_obj:
                    return None, f"Missing mapped incoming feature '{incoming_key}' for model feature '{mf}'"
                row.append(float(features_obj[incoming_key]))
            else:
                # flat mode
                row.append(float(r[incoming_key]))
        X.append(row)

    return np.asarray(X, dtype=float), None


def records_to_X(records):
    X = []
    for r in records:
        X.append([float(r[f]) for f in PRE_FEATURES])
    return np.asarray(X, dtype=float)


def logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def _json_safe_float(value):
    """Converts NaN/inf to None so responses stay valid JSON for strict clients."""
    return float(value) if np.isfinite(value) else None


@app.route("/predict_ps", methods=["POST"])
def predict_ps():
    if GB_MODEL is None:
        return _artifact_error(f"ML artifacts not loaded: {ARTIFACT_LOAD_ERROR}", 500)

    body = request.get_json(force=True, silent=True) or {}
    records = body.get("records")

    featureMap = body.get("featureMap")  # optional
    auto_infer = _as_bool(body.get("auto_infer"), default=True)
    # kept for consistency across endpoints; not required for PS prediction
    treatmentKey = body.get("treatmentKey", "treatment")
    outcomeKey = body.get("outcomeKey", "outcome")

    err = validate_records(
        records,
        require_treatment=False,
        require_outcome=False,
        treatmentKey=treatmentKey,
        outcomeKey=outcomeKey
    )
    if err:
        return jsonify({"error": err}), 400

    X, x_err = build_X_from_records(
        records,
        featureMap=featureMap,
        auto_infer=auto_infer,
        treatmentKey=treatmentKey,
        outcomeKey=outcomeKey
    )
    if x_err:
        return jsonify({"error": x_err}), 400

    ps_final = GB_MODEL.predict_proba(X)[:, 1]
    return jsonify({"ps_final": ps_final.tolist()})


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
        return _artifact_error(f"ML artifacts not loaded: {ARTIFACT_LOAD_ERROR}", 500)

    body = request.get_json(force=True, silent=True) or {}
    records = body.get("records")
    caliper_ratio = float(body.get("caliper_ratio", 0.2))

    featureMap = body.get("featureMap")  # optional
    auto_infer = _as_bool(body.get("auto_infer"), default=True)

    treatmentKey = body.get("treatmentKey", "treatment")
    outcomeKey = body.get("outcomeKey", "outcome")

    err = validate_records(
        records,
        require_treatment=True,
        require_outcome=True,
        treatmentKey=treatmentKey,
        outcomeKey=outcomeKey
    )
    if err:
        return jsonify({"error": err}), 400

    treatments = np.asarray([int(r[treatmentKey]) for r in records], dtype=int)
    outcomes = np.asarray([float(r[outcomeKey]) for r in records], dtype=float)

    X, x_err = build_X_from_records(
        records,
        featureMap=featureMap,
        auto_infer=auto_infer,
        treatmentKey=treatmentKey,
        outcomeKey=outcomeKey
    )
    if x_err:
        return jsonify({"error": x_err}), 400

    ps_final = GB_MODEL.predict_proba(X)[:, 1]
    ps_logit_final = logit(ps_final)

    # Matching using logit(PS) with caliper = caliper_ratio * std(logit(PS))
    caliper = caliper_ratio * np.std(ps_logit_final)
    if not np.isfinite(caliper) or caliper <= 0:
        return jsonify({"error": "Invalid caliper computed from input data"}), 400

    control_mask = treatments == 0
    treat_mask = treatments == 1

    if control_mask.sum() == 0 or treat_mask.sum() == 0:
        return jsonify({"error": "Need both treated and control records in input"}), 400

    control_ps = ps_logit_final[control_mask].reshape(-1, 1)
    treat_ps = ps_logit_final[treat_mask].reshape(-1, 1)

    # Nearest neighbor matching with replacement (as notebook)
    # n_neighbors=1 + radius=caliper
    knn = NearestNeighbors(n_neighbors=1, radius=caliper)
    knn.fit(control_ps)

    matched_pairs = []
    control_indices = np.where(control_mask)[0]
    treated_indices = np.where(treat_mask)[0]

    distances, indices = knn.kneighbors(treat_ps)

    for j in range(len(treated_indices)):
        dist = distances[j][0]
        idx_in_control = indices[j][0]
        if dist <= caliper:
            treat_idx = treated_indices[j]
            ctrl_idx = control_indices[idx_in_control]
            matched_pairs.append((treat_idx, ctrl_idx))

    if len(matched_pairs) == 0:
        return jsonify({
            "matched_pairs": 0,
            "att_mean": None,
            "ci_95": None,
            "p_value_paired_ttest": None,
            "caliper": float(caliper)
        }), 200

    # Paired differences
    diffs = []
    treat_outs = []
    ctrl_outs = []
    for treat_idx, ctrl_idx in matched_pairs:
        treat_outs.append(outcomes[treat_idx])
        ctrl_outs.append(outcomes[ctrl_idx])
        diffs.append(outcomes[treat_idx] - outcomes[ctrl_idx])

    diffs = np.asarray(diffs, dtype=float)
    att_mean = float(np.mean(diffs))

    # Paired t-test: control vs treated (as notebook used ttest_rel(matched_control, matched_treatment))
    # We'll mirror: ttest_rel(ctrl_outs, treat_outs)
    t_stat, p_val = ttest_rel(np.asarray(ctrl_outs), np.asarray(treat_outs))
    p_val = float(p_val)

    # Bootstrap CI on ATT (paired diffs resampled)
    n_bootstrap = int(body.get("n_bootstrap", 500))
    rng = np.random.default_rng(int(body.get("seed", 42)))
    boot = []
    m = len(diffs)
    for _ in range(n_bootstrap):
        sample = diffs[rng.integers(0, m, size=m)]
        boot.append(np.mean(sample))
    ci_low, ci_high = np.percentile(np.asarray(boot), [2.5, 97.5])

    return jsonify({
        "matched_pairs": int(len(matched_pairs)),
        "att_mean": _json_safe_float(att_mean),
        "ci_95": [_json_safe_float(ci_low), _json_safe_float(ci_high)],
        "p_value_paired_ttest": _json_safe_float(p_val),
        "caliper": _json_safe_float(caliper)
    })


@app.route("/health", methods=["GET"])
def health():
    if GB_MODEL is None:
        return jsonify({
            "status": "degraded",
            "artifact_error": ARTIFACT_LOAD_ERROR,
            "model_dir": MODEL_DIR
        }), 500
    return jsonify({"status": "ok", "model_dir": MODEL_DIR})


if __name__ == "__main__":
    # Flask dev server
    port = int(os.environ.get("FLASK_PORT", "8000"))
    app.run(host="0.0.0.0", port=port, debug=False)
