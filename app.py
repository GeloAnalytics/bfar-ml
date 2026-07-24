from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from dotenv import load_dotenv
import os
import json
import socket
import threading
import time

import joblib
import numpy as np
import pandas as pd

import psm_core as core

load_dotenv()

# ---------------------------------------------------------------------------
# Two Flask apps, one process: `app` (dynamic, PORT/HOST) keeps the existing
# auto-detect-baseline-vs-adaptive behavior plus POST /train, unchanged.
# `static_app` (STATIC_PORT/STATIC_HOST) is a second, independent server that
# only ever serves the frozen bfar.csv baseline -- no /train, no dynamic
# fallback, 409 if a request doesn't cover all 57 baseline features. Both are
# started at the bottom of this file, each on its own port/thread.
# ---------------------------------------------------------------------------
app = Flask(__name__)
CORS(app)

static_app = Flask(__name__)
CORS(static_app)

# ---------------------------------------------------------------------------
# Frozen bfar.csv baseline -- produced once by build_model.py, never
# retrained or overwritten by this service. Requests covering all 57 raw
# baseline features score against it directly (no fitting). Shared by both
# apps above.
# ---------------------------------------------------------------------------
MODEL_DIR = os.environ.get("ML_MODEL_DIR", os.path.join(os.path.dirname(__file__), "models"))
MODEL_PATH = os.path.join(MODEL_DIR, "best_model.pkl")
SCALER_PATH = os.path.join(MODEL_DIR, "scaler.pkl")
ALL_FEATURES_PATH = os.path.join(MODEL_DIR, "all_features.json")
CORE_FEATURES_PATH = os.path.join(MODEL_DIR, "core_features.json")
REMAINING_FEATURES_PATH = os.path.join(MODEL_DIR, "remaining_features.json")

KEY_SUPPORT_FEATURES = ["D1.2:A_MOTORC", "D2.1:A_TV", "D2.6:A_FRIDGE", "E3:A_POWER-SUP", "F1:A_HOUSE-OWN"]


def load_baseline_artifacts():
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
    ALL_FEATURES, CORE_FEATURES, REMAINING_FEATURES, BASELINE_MODEL, BASELINE_SCALER = load_baseline_artifacts()
    ARTIFACT_LOAD_ERROR = None
    BASELINE_IMPORTANCES = {
        name: core.json_safe_float(imp)
        for name, imp in zip(ALL_FEATURES, BASELINE_MODEL.feature_importances_)
    } if hasattr(BASELINE_MODEL, "feature_importances_") else None
except Exception as e:
    ALL_FEATURES, CORE_FEATURES, REMAINING_FEATURES = None, None, None
    BASELINE_MODEL, BASELINE_SCALER, BASELINE_IMPORTANCES = None, None, None
    ARTIFACT_LOAD_ERROR = str(e)


# ---------------------------------------------------------------------------
# Dynamic model -- trained from whatever CSV was last POSTed to /train,
# persisted so it survives a restart. Teachable-Machine style: every /train
# call deletes whatever was active and fits a completely fresh model on the
# new upload. No merging with the previous schema, no reuse-shortcut. Only
# `app` (the dynamic service) uses this -- `static_app` never trains.
# ---------------------------------------------------------------------------
STATE_DIR = os.environ.get("ML_DYNAMIC_STATE_DIR", os.path.join(os.path.dirname(__file__), "models", "dynamic"))
STATE_MODEL_PATH = os.path.join(STATE_DIR, "model.pkl")
STATE_META_PATH = os.path.join(STATE_DIR, "meta.json")

STATE = {
    "model": None,
    "feature_cols": None,
    "importances": None,
    "treatment_col": None,
    "treatment_method": None,
    "source_filename": None,
    "rows": None,
    "trained_at": None,
    "trained_columns": None,
    "excluded_as_leakage": None,
    "dropped_for_rebalancing": None,
}

MIN_TRAINING_ROWS = 10
# No cap on selected features -- every non-leakage-correlated numeric candidate
# column is ranked, used to fit the model, and reported with its importance.
# Curating that list down further is left to the integrator, not this service.
TOP_N_FEATURES = None
MAX_RETRAIN_ATTEMPTS = 3


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


def _resolve_scoring_model(df):
    """All 57 raw baseline features present -> the frozen baseline (most
    accurate, always available, needs no training). Otherwise whatever's
    currently in STATE, or (None, None, None, None) if nothing has been
    trained yet."""
    if ALL_FEATURES is not None and all(f in df.columns for f in ALL_FEATURES):
        return BASELINE_MODEL, ALL_FEATURES, BASELINE_SCALER, "baseline"
    if STATE["model"] is not None:
        return STATE["model"], STATE["feature_cols"], None, "dynamic"
    return None, None, None, None


def _score(source, model, feature_cols, scaler, df):
    """Baseline uses build_model.py's median/mode imputation (matches how it
    was trained); the dynamic model uses plain .fillna(0) (matches
    train_psm_model). Returns (ps array, X used)."""
    if source == "baseline":
        X = core.impute_dataframe(df, feature_cols)[feature_cols]
        X_input = scaler.transform(X) if core.model_needs_scaling(model) else X
    else:
        X = df[feature_cols].fillna(0)
        X_input = X
    ps = model.predict_proba(X_input)[:, 1]
    return ps, X


def _no_model_error(df):
    total = len(ALL_FEATURES) if ALL_FEATURES else "?"
    n_covered = len(set(df.columns) & set(ALL_FEATURES)) if ALL_FEATURES else 0
    return (f"no dynamic model trained yet, and this dataset covers only {n_covered}/{total} baseline "
            f"features -- POST a CSV to /train first, or include all {total} baseline features")


def _missing_features_error(df):
    missing = [f for f in ALL_FEATURES if f not in df.columns]
    return (f"this port serves only the frozen baseline and requires all {len(ALL_FEATURES)} "
            f"features; missing: {missing[:5]}{'...' if len(missing) > 5 else ''}")


def _decision_support_payload(df, ps):
    ps_logit = core.logit(ps)
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

    return ps_logit, decision_support


# ===========================================================================
# `app` -- dynamic service (auto-detect baseline-vs-adaptive, POST /train)
# ===========================================================================

@app.route("/", methods=["GET"])
def test_ui():
    """Manual smoke-test page for POST /train and POST /train/predict_ps_batch --
    not part of the production integration path (see README's integration
    guide), just a way to drive an upload through a browser instead of curl."""
    return send_from_directory(os.path.dirname(os.path.abspath(__file__)), "test_ui.html")


@app.route("/train", methods=["POST"])
def train():
    """
    multipart/form-data:
      file: <CSV file>               required
      treatment_column: <col name>   optional -- bypasses auto-detection if
                                      the heuristic picks the wrong column

    If the uploaded CSV's column set exactly matches the columns of whatever
    dataset last trained the active dynamic model, retraining is skipped --
    the existing model is reused as-is and just re-scored against this
    upload. Any other column set retrains from scratch, Teachable-Machine
    style: auto-detects the treatment/control column (see
    psm_core.detect_treatment_column), ranks every usable numeric column by
    importance for predicting it (excluding near-perfect treatment proxies,
    see psm_core._leakage_correlated_columns), and fits a fresh model on
    every ranked candidate -- no top-N cap (psm_core.select_top_features,
    psm_core.train_psm_model). The full ranking ships in the response
    (feature_selection.selected / model_interpretation.feature_contributions)
    so the integrator can curate the list further downstream; this service
    doesn't cut it down for you. If covariate balance isn't achieved (mean
    |SMD| after matching >= 0.1, see psm_core.covariate_balance), the single
    worst-balanced feature is dropped and training retries, up to
    MAX_RETRAIN_ATTEMPTS times.

    Response includes, alongside the trained-model summary: feature_selection
    (selected features + what got excluded and why), ps_output (in-sample
    propensity scores for this upload), covariate_balance (SMD before/after
    matching, PS overlap, balance_achieved verdict), model_interpretation
    (real SHAP values via shap.TreeExplainer -- mean |SHAP value| per feature
    plus plain-language socioeconomic_insights), and decision_support
    (PS-quartile table).

    /train/predict_ps, /train/estimate_att, /train/predict_ps_batch score
    against whichever model currently applies: the frozen baseline if the
    request covers all 57 raw features, otherwise this dynamic model,
    persisted under models/dynamic/ so a restart doesn't lose it.
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

    uploaded_columns = sorted(df.columns)
    override_col = request.form.get("treatment_column") or None
    retrain_skipped = (
        STATE["model"] is not None
        and STATE.get("trained_columns") == uploaded_columns
        # An explicit override naming a *different* column than what's already
        # active is a deliberate request to redo training under that column --
        # honor it instead of silently reusing the old model's treatment column.
        and (override_col is None or override_col == STATE.get("treatment_col"))
    )

    if retrain_skipped:
        try:
            treatment_col, treatment_binarized, _ = core.detect_treatment_column(df, override_col=STATE["treatment_col"])
        except ValueError as e:
            return jsonify({"error": str(e)}), 400
        # Re-binarizing via override_col always reports "manual_override" -- report
        # how the column was *actually* found when the active model was trained,
        # not an artifact of how we're re-deriving the binarized series here.
        method = STATE["treatment_method"]
        model = STATE["model"]
        top_features = STATE["feature_cols"]
        final_importances = STATE["importances"]
        excluded_leakage = STATE.get("excluded_as_leakage") or []
        dropped_for_rebalancing = STATE.get("dropped_for_rebalancing") or []
        retrain_attempts = 0
    else:
        try:
            treatment_col, treatment_binarized, method = core.detect_treatment_column(df, override_col=override_col)
        except ValueError as e:
            return jsonify({"error": str(e)}), 400
        if treatment_col is None:
            return jsonify({"error": "could not auto-detect a treatment/control column in this dataset; retry with a 'treatment_column' form field"}), 400

        extra_exclude = set()
        dropped_for_rebalancing = []
        model, top_features, final_importances, excluded_leakage = None, None, None, None
        balance = None
        for attempt in range(1, MAX_RETRAIN_ATTEMPTS + 1):
            retrain_attempts = attempt
            try:
                top_features, final_importances, excluded_leakage = core.select_top_features(
                    df, treatment_col, treatment_binarized, top_n=TOP_N_FEATURES, extra_exclude=extra_exclude)
                model, _ = core.train_psm_model(df, treatment_binarized, top_features)
            except ValueError as e:
                return jsonify({"error": str(e)}), 400

            ps, _X = _score("dynamic", model, top_features, None, df)
            balance = core.covariate_balance(df, treatment_binarized, top_features, core.logit(ps))
            if balance["balance_achieved"] or not balance.get("worst_feature") or attempt == MAX_RETRAIN_ATTEMPTS:
                break
            dropped_for_rebalancing.append(balance["worst_feature"])
            extra_exclude.add(balance["worst_feature"])

        STATE.update({
            "model": model,
            "feature_cols": top_features,
            "importances": final_importances,
            "treatment_col": treatment_col,
            "treatment_method": method,
            "source_filename": file.filename,
            "rows": len(df),
            "trained_at": time.time(),
            "trained_columns": uploaded_columns,
            "excluded_as_leakage": excluded_leakage,
            "dropped_for_rebalancing": dropped_for_rebalancing,
        })
        save_state()

    ps, X_used = _score("dynamic", model, top_features, None, df)
    ps_logit_arr = core.logit(ps)
    balance = core.covariate_balance(df, treatment_binarized, top_features, ps_logit_arr)
    _, decision_support = _decision_support_payload(df, ps)

    shap_contributions = core.compute_shap_feature_contributions(model, X_used, top_features)
    socioeconomic_insights = core.generate_socioeconomic_insights(shap_contributions)

    ranked = sorted(final_importances.items(), key=lambda kv: kv[1], reverse=True)
    ranked_features = [{"feature": name, "importance": imp} for name, imp in ranked]

    return jsonify({
        "status": "trained",
        "retrained": not retrain_skipped,
        "retrain_attempts": retrain_attempts,
        "rows": len(df),
        "treatment_column": treatment_col,
        "treatment_detection_method": method,
        "feature_selection": {
            "n_features_selected": len(top_features),
            "selected": ranked_features,
            "excluded_as_leakage": excluded_leakage,
            "dropped_for_rebalancing": dropped_for_rebalancing,
        },
        "ps_output": {
            "ps": [core.json_safe_float(v) for v in ps],
            "ps_logit": [core.json_safe_float(v) for v in ps_logit_arr],
            "ps_summary": {
                "min": core.json_safe_float(np.min(ps)),
                "max": core.json_safe_float(np.max(ps)),
                "mean": core.json_safe_float(np.mean(ps)),
                "median": core.json_safe_float(np.median(ps)),
            },
        },
        "covariate_balance": balance,
        "model_interpretation": {
            "method": "SHAP (shap.TreeExplainer, exact for tree-ensemble models) -- mean |SHAP value| "
                      "per feature across all rows in this upload, in the model's raw log-odds space",
            "feature_contributions": shap_contributions,
            "socioeconomic_insights": socioeconomic_insights,
        },
        "decision_support": decision_support,
        # kept for backwards compatibility with existing callers
        "n_features_selected": len(top_features),
        "top_features": ranked_features,
        "excluded_as_leakage": excluded_leakage,
    })


@app.route("/train/predict_ps", methods=["POST"])
def predict_ps():
    """
    JSON body: { "records": [ { "<column_name>": value, ... }, ... ] }

    Scores against the frozen baseline if every record covers all 57 raw
    bfar features, otherwise against whatever's currently in the dynamic
    model (see POST /train). 409 if neither applies.
    """
    body = request.get_json(force=True, silent=True) or {}
    records = body.get("records")
    if not isinstance(records, list) or len(records) == 0:
        return jsonify({"error": "records must be a non-empty array"}), 400
    df = pd.DataFrame(records)

    model, feature_cols, scaler, source = _resolve_scoring_model(df)
    if model is None:
        return jsonify({"error": _no_model_error(df)}), 409

    missing = [f for f in feature_cols if f not in df.columns]
    if missing:
        return jsonify({"error": f"records missing required features: {missing[:5]}{'...' if len(missing) > 5 else ''}"}), 400

    ps, _X = _score(source, model, feature_cols, scaler, df)
    return jsonify({
        "ps_final": [core.json_safe_float(v) for v in ps],
        "source": source,
        "n_features_used": len(feature_cols),
    })


@app.route("/train/estimate_att", methods=["POST"])
def estimate_att():
    """
    JSON body:
    {
      "records": [ { "<column_name>": value, ..., "treatment": 0/1, "outcome": number }, ... ],
      "treatmentKey": "treatment", "outcomeKey": "outcome",  # optional
      "caliper_ratio": 0.2, "n_bootstrap": 500, "seed": 42   # optional
    }
    """
    body = request.get_json(force=True, silent=True) or {}
    records = body.get("records")
    if not isinstance(records, list) or len(records) == 0:
        return jsonify({"error": "records must be a non-empty array"}), 400
    df = pd.DataFrame(records)

    treatment_key = body.get("treatmentKey", "treatment")
    outcome_key = body.get("outcomeKey", "outcome")
    if treatment_key not in df.columns:
        return jsonify({"error": f"missing required field: {treatment_key}"}), 400
    if outcome_key not in df.columns:
        return jsonify({"error": f"missing required field: {outcome_key}"}), 400

    model, feature_cols, scaler, source = _resolve_scoring_model(df)
    if model is None:
        return jsonify({"error": _no_model_error(df)}), 409

    missing = [f for f in feature_cols if f not in df.columns]
    if missing:
        return jsonify({"error": f"records missing required features: {missing[:5]}{'...' if len(missing) > 5 else ''}"}), 400

    ps, _X = _score(source, model, feature_cols, scaler, df)
    ps_logit = core.logit(ps)
    treatments = df[treatment_key].astype(int).to_numpy()
    outcomes = pd.to_numeric(df[outcome_key], errors="coerce").to_numpy(dtype=float)

    result, err = core.matched_att(
        ps_logit, treatments, outcomes,
        caliper_ratio=float(body.get("caliper_ratio", 0.2)),
        n_bootstrap=int(body.get("n_bootstrap", 500)),
        seed=int(body.get("seed", 42)),
    )
    if err:
        return jsonify({"error": err}), 400

    result["source"] = source
    result["n_features_used"] = len(feature_cols)
    return jsonify(result)


@app.route("/train/predict_ps_batch", methods=["POST"])
def predict_ps_batch():
    """
    multipart/form-data: file: <CSV file>   required

    Whole-dataset counterpart to /train/predict_ps: scores every row and
    returns a decision-support table stratified by propensity-score
    quartile, alongside the raw per-row scores.
    """
    file = request.files.get("file")
    if file is None:
        return jsonify({"error": "multipart form field 'file' (CSV) is required"}), 400

    try:
        df = pd.read_csv(file)
    except Exception as e:
        return jsonify({"error": f"could not parse CSV: {e}"}), 400
    if len(df) == 0:
        return jsonify({"error": "uploaded CSV has no rows"}), 400

    model, feature_cols, scaler, source = _resolve_scoring_model(df)
    if model is None:
        return jsonify({"error": _no_model_error(df)}), 409

    missing = [f for f in feature_cols if f not in df.columns]
    if missing:
        return jsonify({"error": f"CSV missing required features: {missing[:5]}{'...' if len(missing) > 5 else ''}"}), 400

    ps, _X = _score(source, model, feature_cols, scaler, df)
    ps_logit, decision_support = _decision_support_payload(df, ps)

    return jsonify({
        "rows": len(df),
        "source": source,
        "n_features_used": len(feature_cols),
        "ps": [core.json_safe_float(v) for v in ps],
        "ps_logit": [core.json_safe_float(v) for v in ps_logit],
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

    response = {
        "status": "ok",
        "model_dir": MODEL_DIR,
        "baseline": {
            "source": "bfar.csv (baked-in, static, never retrained)",
            "model_type": type(BASELINE_MODEL).__name__,
            "n_features_total": len(ALL_FEATURES),
        },
    }
    if BASELINE_IMPORTANCES is not None:
        top_features = sorted(BASELINE_IMPORTANCES.items(), key=lambda kv: kv[1], reverse=True)[:30]
        response["baseline"]["top_features"] = [{"feature": name, "importance": imp} for name, imp in top_features]

    if STATE["model"] is None:
        response["dynamic"] = {"status": "empty", "message": "no dataset trained yet; POST a CSV to /train"}
    else:
        ranked = sorted(STATE["importances"].items(), key=lambda kv: kv[1], reverse=True)
        response["dynamic"] = {
            "status": "ok",
            "source_filename": STATE["source_filename"],
            "rows": STATE["rows"],
            "trained_at": STATE["trained_at"],
            "treatment_column": STATE["treatment_col"],
            "treatment_detection_method": STATE["treatment_method"],
            "n_features_selected": len(STATE["feature_cols"]),
            "top_features": [{"feature": name, "importance": imp} for name, imp in ranked],
        }
    return jsonify(response)


# ===========================================================================
# `static_app` -- baseline-only service, no /train, no dynamic fallback
# ===========================================================================

@static_app.route("/predict_ps", methods=["POST"])
def static_predict_ps():
    """
    JSON body: { "records": [ { "<column_name>": value, ... }, ... ] }

    Scores against the frozen bfar.csv baseline. Every record must cover all
    57 raw baseline features -- this service never trains or adapts.
    """
    if BASELINE_MODEL is None:
        return jsonify({"error": f"ML artifacts not loaded: {ARTIFACT_LOAD_ERROR}"}), 500

    body = request.get_json(force=True, silent=True) or {}
    records = body.get("records")
    if not isinstance(records, list) or len(records) == 0:
        return jsonify({"error": "records must be a non-empty array"}), 400
    df = pd.DataFrame(records)

    if not all(f in df.columns for f in ALL_FEATURES):
        return jsonify({"error": _missing_features_error(df)}), 409

    ps, _X = _score("baseline", BASELINE_MODEL, ALL_FEATURES, BASELINE_SCALER, df)
    return jsonify({
        "ps_final": [core.json_safe_float(v) for v in ps],
        "source": "baseline",
        "n_features_used": len(ALL_FEATURES),
    })


@static_app.route("/estimate_att", methods=["POST"])
def static_estimate_att():
    """
    JSON body:
    {
      "records": [ { "<column_name>": value, ..., "treatment": 0/1, "outcome": number }, ... ],
      "treatmentKey": "treatment", "outcomeKey": "outcome",  # optional
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

    treatment_key = body.get("treatmentKey", "treatment")
    outcome_key = body.get("outcomeKey", "outcome")
    if treatment_key not in df.columns:
        return jsonify({"error": f"missing required field: {treatment_key}"}), 400
    if outcome_key not in df.columns:
        return jsonify({"error": f"missing required field: {outcome_key}"}), 400

    if not all(f in df.columns for f in ALL_FEATURES):
        return jsonify({"error": _missing_features_error(df)}), 409

    ps, _X = _score("baseline", BASELINE_MODEL, ALL_FEATURES, BASELINE_SCALER, df)
    ps_logit = core.logit(ps)
    treatments = df[treatment_key].astype(int).to_numpy()
    outcomes = pd.to_numeric(df[outcome_key], errors="coerce").to_numpy(dtype=float)

    result, err = core.matched_att(
        ps_logit, treatments, outcomes,
        caliper_ratio=float(body.get("caliper_ratio", 0.2)),
        n_bootstrap=int(body.get("n_bootstrap", 500)),
        seed=int(body.get("seed", 42)),
    )
    if err:
        return jsonify({"error": err}), 400

    result["source"] = "baseline"
    result["n_features_used"] = len(ALL_FEATURES)
    return jsonify(result)


@static_app.route("/predict_ps_batch", methods=["POST"])
def static_predict_ps_batch():
    """
    multipart/form-data: file: <CSV file>   required

    Whole-dataset counterpart to /predict_ps: scores every row against the
    frozen baseline and returns a decision-support table stratified by
    propensity-score quartile, alongside the raw per-row scores.
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

    if not all(f in df.columns for f in ALL_FEATURES):
        return jsonify({"error": _missing_features_error(df)}), 409

    ps, _X = _score("baseline", BASELINE_MODEL, ALL_FEATURES, BASELINE_SCALER, df)
    ps_logit, decision_support = _decision_support_payload(df, ps)

    return jsonify({
        "rows": len(df),
        "source": "baseline",
        "n_features_used": len(ALL_FEATURES),
        "ps": [core.json_safe_float(v) for v in ps],
        "ps_logit": [core.json_safe_float(v) for v in ps_logit],
        "ps_summary": {
            "min": core.json_safe_float(np.min(ps)),
            "max": core.json_safe_float(np.max(ps)),
            "mean": core.json_safe_float(np.mean(ps)),
            "median": core.json_safe_float(np.median(ps)),
        },
        "decision_support": decision_support,
    })


@static_app.route("/health", methods=["GET"])
def static_health():
    if BASELINE_MODEL is None:
        return jsonify({
            "status": "degraded",
            "artifact_error": ARTIFACT_LOAD_ERROR,
            "model_dir": MODEL_DIR,
        }), 500

    response = {
        "status": "ok",
        "model_dir": MODEL_DIR,
        "baseline": {
            "source": "bfar.csv (baked-in, static, never retrained)",
            "model_type": type(BASELINE_MODEL).__name__,
            "n_features_total": len(ALL_FEATURES),
        },
    }
    if BASELINE_IMPORTANCES is not None:
        top_features = sorted(BASELINE_IMPORTANCES.items(), key=lambda kv: kv[1], reverse=True)[:30]
        response["baseline"]["top_features"] = [{"feature": name, "importance": imp} for name, imp in top_features]
    return jsonify(response)


def _wait_for_port(check_host, check_port, timeout=10):
    """Werkzeug's own startup banner doesn't reliably reach the console when
    printed from a background thread on every platform/terminal (seen on
    Windows PowerShell: the static app's "Running on..." lines sometimes
    never appear even though it's genuinely up). Poll the socket instead of
    trusting print output, so the confirmation below is always accurate."""
    probe_host = "127.0.0.1" if check_host == "0.0.0.0" else check_host
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with socket.create_connection((probe_host, check_port), timeout=0.5):
                return True
        except OSError:
            time.sleep(0.1)
    return False


if __name__ == "__main__":
    host = os.environ.get("HOST", "0.0.0.0")
    port = int(os.environ.get("PORT", "8000"))
    static_host = os.environ.get("STATIC_HOST", os.environ.get("HOST", "0.0.0.0"))
    static_port = int(os.environ.get("STATIC_PORT", "8001"))

    static_thread = threading.Thread(
        target=lambda: static_app.run(host=static_host, port=static_port, debug=False, use_reloader=False),
        daemon=True,
    )
    static_thread.start()

    if _wait_for_port(static_host, static_port):
        print(f" * Static app confirmed listening on http://{static_host}:{static_port}", flush=True)
    else:
        print(f" * WARNING: static app did not come up on port {static_port} within 10s -- check for a port conflict", flush=True)

    print(f" * Starting dynamic app on http://{host}:{port} (blocking here; Ctrl+C stops both)", flush=True)
    app.run(host=host, port=port, debug=False)
