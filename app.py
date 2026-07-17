from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv
import os
import json

import joblib
import numpy as np
import pandas as pd

import psm_core as core
import psm_indices
from column_matcher import match_columns
from mapping_store import MappingStore, column_signature

load_dotenv()

app = Flask(__name__)
CORS(app)

# Frozen bfar.csv baseline artifacts (produced by build_model.py). This
# service never retrains or overwrites them -- every request re-anchors to
# the same two frozen models and routes into one of three tiers:
#
#   tier 1: upload carries bfar's exact 57 raw columns -> raw baseline model
#   tier 2: upload's program has a registered column mapping -> folded into
#           6 composite indices -> index-space baseline model (no fitting)
#   tier 3: everything else -> throwaway per-request model; the request also
#           advances the automatic mapping-promotion state machine so a
#           stable program graduates itself to tier 2 (see mapping_store.py)
MODEL_DIR = os.environ.get("ML_MODEL_DIR", os.path.join(os.path.dirname(__file__), "models"))
MAPPINGS_DIR = os.environ.get("ML_MAPPINGS_DIR", os.path.join(os.path.dirname(__file__), "mappings"))

_ARTIFACTS = {
    "model": "best_model.pkl",
    "scaler": "scaler.pkl",
    "index_model": "index_model.pkl",
    "index_scaler": "index_scaler.pkl",
    "all_features": "all_features.json",
    "core_features": "core_features.json",
    "remaining_features": "remaining_features.json",
    "taxonomy": "index_taxonomy.json",
    "index_stats": "index_stats.json",
}

KEY_SUPPORT_FEATURES = ["D1.2:A_MOTORC", "D2.1:A_TV", "D2.6:A_FRIDGE", "E3:A_POWER-SUP", "F1:A_HOUSE-OWN"]


def load_artifacts():
    loaded = {}
    for key, fname in _ARTIFACTS.items():
        path = os.path.join(MODEL_DIR, fname)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing model artifact: {path}")
        loaded[key] = joblib.load(path) if fname.endswith(".pkl") else json.load(open(path, "r"))
    return loaded


try:
    A = load_artifacts()
    ARTIFACT_LOAD_ERROR = None
except Exception as e:
    A = None
    ARTIFACT_LOAD_ERROR = str(e)

STORE = MappingStore(MAPPINGS_DIR)
TIER_COUNTS = {1: 0, 2: 0, 3: 0}


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
    return len(A["remaining_features"])


def _resolve_program_key(df, explicit_id):
    """Explicit program_id wins; otherwise the column signature doubles as
    the implicit program identity (same export re-uploaded -> same key)."""
    signature = column_signature(df.columns)
    return (str(explicit_id) if explicit_id else signature), signature


def _route_tier(df, program_key):
    """Returns (tier, registered_mapping_or_None). Tier 1 (exact raw schema)
    beats a registered mapping -- the raw model is the more accurate one."""
    if all(f in df.columns for f in A["all_features"]):
        return 1, None
    registered = STORE.get_registered(program_key)
    if registered:
        return 2, registered["mapping"]
    return 3, None


def _observe_tier3(df, program_key, signature):
    """Runs the matcher and advances the draft->promotion state machine.
    Promotion only affects FUTURE requests; and a store/matcher failure must
    never break the scoring response it piggybacks on."""
    try:
        mapping, _scores = match_columns(df.columns, A["taxonomy"])
        if not mapping:
            return {"matched_items": 0, "indices_covered": 0, "draft_consistent_count": 0, "promoted": False}
        coverage = psm_indices.indices_covered(mapping, A["taxonomy"], df.columns)
        index_df, imputed = psm_indices.compute_indices(df, mapping, A["taxonomy"], A["index_stats"])
        covered_df = index_df[[c for c in index_df.columns if c not in imputed]]
        return STORE.record_observation(program_key, signature, mapping, coverage, covered_df)
    except Exception as e:
        return {"error": f"mapping observation failed: {e}"}


@app.route("/predict_ps", methods=["POST"])
def predict_ps():
    """
    JSON body:
    {
      "records": [ { "<column_name>": value, ... }, ... ],
      "program_id": "<stable id for this program>",   # optional -- defaults to
                                                       # the column signature
      "treatment_column": "<name>",   # optional -- only used on tier 3
      "n_flex": 27                    # optional, tier 3 only
    }

    Tier 1 (all 57 bfar columns) and tier 2 (registered mapping) score
    against a frozen baseline with no fitting and no labels needed. Tier 3
    requires a treatment column (to rank this dataset's own features
    against) and at least 10 records.
    """
    if A is None:
        return jsonify({"error": f"ML artifacts not loaded: {ARTIFACT_LOAD_ERROR}"}), 500

    body = request.get_json(force=True, silent=True) or {}
    records = body.get("records")
    if not isinstance(records, list) or len(records) == 0:
        return jsonify({"error": "records must be a non-empty array"}), 400

    df = pd.DataFrame(records)
    program_key, signature = _resolve_program_key(df, body.get("program_id"))
    tier, registered_mapping = _route_tier(df, program_key)

    response = {"tier": tier, "program_key": program_key,
                "treatment_column": None, "treatment_detection_method": None}

    if tier == 2:
        result = core.predict_with_index_model(
            df, registered_mapping, A["taxonomy"], A["index_stats"], A["index_model"], A["index_scaler"])
        response["imputed_indices"] = result["imputed_indices"]
    else:
        if tier == 3:
            df, treatment_col, treatment_method = _resolve_treatment_column(df, override_col=body.get("treatment_column"))
            response["treatment_column"] = treatment_col
            response["treatment_detection_method"] = treatment_method
        else:
            treatment_col = None
        try:
            result = core.predict_dynamic(
                df, A["core_features"], A["all_features"], A["model"], A["scaler"],
                treatment_col=treatment_col or "treatment",
                n_flex=int(body.get("n_flex", _default_n_flex())),
            )
        except ValueError as e:
            return jsonify({"error": str(e)}), 400
        if tier == 3:
            response["core_coverage"] = core.json_safe_float(result["core_coverage"])
            response["mapping_status"] = _observe_tier3(df, program_key, signature)

    TIER_COUNTS[tier] += 1
    response.update({
        "ps_final": [core.json_safe_float(v) for v in result["ps"]],
        "used_baseline": result["used_baseline"],
        "n_features_used": len(result["final_features"]),
        "final_features": result["final_features"],
    })
    return jsonify(response)


@app.route("/estimate_att", methods=["POST"])
def estimate_att():
    """
    JSON body (records must include treatment + outcome):
    {
      "records": [ { "<column_name>": value, ..., "treatment": 0/1, "outcome": number }, ... ],
      "program_id": "<stable id>",    # optional
      "treatment_column": "<name>",   # optional -- auto-detected otherwise
      "outcomeKey": "outcome",        # optional
      "n_flex": 27,                   # optional, tier 3 only
      "caliper_ratio": 0.2, "n_bootstrap": 500, "seed": 42   # optional
    }
    """
    if A is None:
        return jsonify({"error": f"ML artifacts not loaded: {ARTIFACT_LOAD_ERROR}"}), 500

    body = request.get_json(force=True, silent=True) or {}
    records = body.get("records")
    if not isinstance(records, list) or len(records) == 0:
        return jsonify({"error": "records must be a non-empty array"}), 400

    df = pd.DataFrame(records)
    program_key, signature = _resolve_program_key(df, body.get("program_id"))
    tier, registered_mapping = _route_tier(df, program_key)

    outcome_key = body.get("outcomeKey", "outcome")
    override_col = body.get("treatment_column") or body.get("treatmentKey")
    df, treatment_col, treatment_method = _resolve_treatment_column(df, override_col=override_col)

    if treatment_col is None:
        return jsonify({"error": "could not auto-detect a treatment/control column in this dataset; retry with a 'treatment_column' field"}), 400
    if outcome_key not in df.columns:
        return jsonify({"error": f"missing required outcome column: {outcome_key}"}), 400

    att_kwargs = dict(
        treatment_col=treatment_col, outcome_col=outcome_key,
        caliper_ratio=float(body.get("caliper_ratio", 0.2)),
        n_bootstrap=int(body.get("n_bootstrap", 500)),
        seed=int(body.get("seed", 42)),
    )
    if tier == 2:
        result, err = core.estimate_att_with_index_model(
            df, registered_mapping, A["taxonomy"], A["index_stats"],
            A["index_model"], A["index_scaler"], **att_kwargs)
    else:
        result, err = core.estimate_att_dynamic(
            df, A["core_features"], A["all_features"], A["model"], A["scaler"],
            n_flex=int(body.get("n_flex", _default_n_flex())), **att_kwargs)
    if err:
        return jsonify({"error": err}), 400

    if tier == 3:
        result["mapping_status"] = _observe_tier3(df, program_key, signature)
    TIER_COUNTS[tier] += 1
    result.update({
        "tier": tier,
        "program_key": program_key,
        "treatment_column": treatment_col,
        "treatment_detection_method": treatment_method,
    })
    return jsonify(result)


@app.route("/predict_ps_batch", methods=["POST"])
def predict_ps_batch():
    """
    multipart/form-data:
      file: <CSV file>               required
      program_id: <stable id>        optional -- defaults to column signature
      treatment_column: <col name>   optional -- tier 3 only
      n_flex: 27                     optional -- tier 3 only

    Whole-dataset counterpart to /predict_ps (mirrors predictor_psm.ipynb's
    predict_and_support workflow): scores every row and returns a
    decision-support table stratified by propensity-score quartile, in
    addition to the raw per-row scores. Nothing about the baseline is ever
    changed by this call.
    """
    if A is None:
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

    program_key, signature = _resolve_program_key(df, request.form.get("program_id"))
    tier, registered_mapping = _route_tier(df, program_key)

    response = {"tier": tier, "program_key": program_key,
                "treatment_column": None, "treatment_detection_method": None}

    if tier == 2:
        result = core.predict_with_index_model(
            df, registered_mapping, A["taxonomy"], A["index_stats"], A["index_model"], A["index_scaler"])
        response["imputed_indices"] = result["imputed_indices"]
    else:
        if tier == 3:
            df, treatment_col, treatment_method = _resolve_treatment_column(df, override_col=request.form.get("treatment_column") or None)
            response["treatment_column"] = treatment_col
            response["treatment_detection_method"] = treatment_method
        else:
            treatment_col = None
        try:
            result = core.predict_dynamic(
                df, A["core_features"], A["all_features"], A["model"], A["scaler"],
                treatment_col=treatment_col or "treatment",
                n_flex=int(request.form.get("n_flex", _default_n_flex())),
            )
        except ValueError as e:
            return jsonify({"error": str(e)}), 400
        if tier == 3:
            response["core_coverage"] = core.json_safe_float(result["core_coverage"])
            response["mapping_status"] = _observe_tier3(df, program_key, signature)

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

    TIER_COUNTS[tier] += 1
    response.update({
        "rows": len(df),
        "used_baseline": result["used_baseline"],
        "n_features_used": len(result["final_features"]),
        "final_features": result["final_features"],
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
    return jsonify(response)


@app.route("/mappings", methods=["GET"])
def list_mappings():
    return jsonify(STORE.list_all())


@app.route("/mappings/<program_key>", methods=["DELETE"])
def delete_mapping(program_key):
    """Demotes a program back to tier 3 (removes its registered mapping AND
    any draft, so a stale counter can't instantly re-promote it)."""
    if STORE.demote(program_key):
        return jsonify({"status": "demoted", "program_key": program_key})
    return jsonify({"error": f"no registered mapping or draft for '{program_key}'"}), 404


@app.route("/health", methods=["GET"])
def health():
    if A is None:
        return jsonify({
            "status": "degraded",
            "artifact_error": ARTIFACT_LOAD_ERROR,
            "model_dir": MODEL_DIR,
        }), 500

    mappings = STORE.list_all()
    return jsonify({
        "status": "ok",
        "model_dir": MODEL_DIR,
        "psm": {
            "source": "bfar.csv baseline (frozen) -- tier 1: raw 57-feature model; tier 2: 6-index model via registered mappings; tier 3: per-request adaptation, never persisted",
            "model_type": type(A["model"]).__name__,
            "index_model_type": type(A["index_model"]).__name__,
            "n_core_features": len(A["core_features"]),
            "n_remaining_features": len(A["remaining_features"]),
            "n_all_features": len(A["all_features"]),
        },
        "mappings": {
            "registered": len(mappings["registered"]),
            "drafts": len(mappings["drafts"]),
        },
        "tier_requests_since_start": {str(k): v for k, v in TIER_COUNTS.items()},
    })


if __name__ == "__main__":
    host = os.environ.get("HOST", "0.0.0.0")
    port = int(os.environ.get("PORT", "8000"))
    app.run(host=host, port=port, debug=False)
