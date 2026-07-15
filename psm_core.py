"""Shared propensity-score-matching (PSM) logic used by both Flask services:
app.py (static, bfar.csv-only) and app_dynamic.py (arbitrary uploaded datasets).
"""
import re
from difflib import SequenceMatcher

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neighbors import NearestNeighbors
from scipy.stats import ttest_rel


_ID_LIKE_NAME = re.compile(r"(^|_)(id|uuid|guid|index)($|_)", re.IGNORECASE)
_TREATMENT_NAME_HINTS = (
    "treat", "program", "particip", "enroll", "assist", "benefic",
    "recipient", "grant", "subsid", "loan", "interven",
)


def json_safe_float(value):
    """Converts NaN/inf to None so responses stay valid JSON for strict clients."""
    value = float(value)
    return value if np.isfinite(value) else None


def as_bool(value, default=True):
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() not in {"false", "0", "no", "off"}
    return bool(value)


def _normalize_key(s):
    s = str(s).lower()
    return re.sub(r"[^a-z0-9]+", "", s)


def _is_id_like(series, name):
    if _ID_LIKE_NAME.search(str(name)):
        return True
    n = len(series)
    return n > 0 and series.nunique(dropna=True) == n


def _balance_score(balance):
    """Peaks at balance=0.5 (an even split), drops toward the extremes."""
    if balance <= 0 or balance >= 1:
        return -1.0
    return 1.0 - abs(0.5 - balance) * 2


def detect_treatment_column(df, exclude_cols=None, override_col=None):
    """
    Heuristically finds a binary treatment/control indicator in an arbitrary
    dataset. Considers, per column:
      - "binary_value": the column already has exactly 2 distinct values
        (0/1, True/False, Yes/No, ...).
      - "notna_mask": the column is populated only for one group and left
        blank for the other (e.g. bfar.csv's 'Y_BOAT-RE', non-null only for
        program participants) -- treated as notna().astype(int).

    "notna_mask" gets a large tier bonus over "binary_value": in program/
    survey-style datasets (the intended use case here) the treatment marker
    is usually "this intervention-specific field is only populated for
    participants", while merely-balanced binary columns are far more often
    incidental demographic covariates (owns-a-TV, has-insurance, ...) that
    happen to land near a 50/50 split by chance. A column literally named
    "treatment" always wins outright regardless of tier. Ties within a tier
    (e.g. several follow-up columns sharing one skip-logic pattern) are
    broken by earliest column position, since the primary flag conventionally
    precedes its own follow-up detail questions.

    `override_col`, if given, bypasses detection and binarizes that column
    directly -- the escape hatch for when the heuristic guesses wrong.

    Returns (column_name, binarized_series, method) or (None, None, None).
    """
    if override_col is not None:
        if override_col not in df.columns:
            raise ValueError(f"override treatment column '{override_col}' not found in dataset")
        non_null = df[override_col].dropna()
        uniques = non_null.unique()
        if len(uniques) == 2:
            positive = sorted(uniques, key=str)[-1]
            binarized = (df[override_col] == positive).astype(int)
        else:
            binarized = df[override_col].notna().astype(int)
        return override_col, binarized, "manual_override"

    exclude_cols = set(exclude_cols or [])
    candidates = []

    for position, col in enumerate(df.columns):
        if col in exclude_cols or _is_id_like(df[col], col):
            continue

        name_bonus = 0.15 if any(h in col.lower() for h in _TREATMENT_NAME_HINTS) else 0.0
        exact_bonus = 0.5 if col.strip().lower() == "treatment" else 0.0
        position_tiebreak = position * 1e-6  # nudges earlier columns ahead on near-exact ties

        non_null = df[col].dropna()
        uniques = non_null.unique()

        if 0 < len(uniques) <= 2:
            positive = sorted(uniques, key=str)[-1]
            binarized = (df[col] == positive).astype(int)
            score = _balance_score(binarized.mean()) + name_bonus + exact_bonus - position_tiebreak
            candidates.append((score, col, binarized, "binary_value"))

        null_frac = df[col].isna().mean()
        if 0.02 <= null_frac <= 0.98:
            binarized = df[col].notna().astype(int)
            score = _balance_score(binarized.mean()) + name_bonus + exact_bonus - position_tiebreak + 0.3
            candidates.append((score, col, binarized, "notna_mask"))

    if not candidates:
        return None, None, None

    candidates.sort(key=lambda c: c[0], reverse=True)
    _, name, binarized, method = candidates[0]
    return name, binarized, method


def _numeric_candidate_columns(df, exclude):
    return [
        c for c in df.columns
        if c not in exclude
        and pd.api.types.is_numeric_dtype(df[c])
        and not _is_id_like(df[c], c)
    ]


def _leakage_correlated_columns(df, treatment_col, treatment_binarized, candidate_cols, threshold=0.95):
    """
    Excludes candidates that are near-direct proxies for treatment, via two
    checks:
      - null-pattern correlation: when treatment is detected via
        "notna_mask" (populated only for participants), whole blocks of
        follow-up questions are typically skipped for non-participants using
        that same logic -- those columns re-encode "was this question
        reached" rather than a genuine pre-treatment covariate.
      - raw-value correlation: a column whose values almost perfectly
        determine treatment status is very likely a renamed/recoded copy of
        the treatment/control group assignment itself (seen in bfar.csv as
        'A2:GROUP', correlation 1.0 with 'Y_BOAT-RE'). Even setting leakage
        aside, PSM requires overlapping propensity distributions between
        groups ("common support"); a feature that near-perfectly separates
        the groups violates that and shouldn't drive the propensity model.
    """
    treatment_mask = df[treatment_col].isna().astype(int)
    check_null_pattern = treatment_mask.nunique() == 2
    treatment_values = treatment_binarized.to_numpy(dtype=float)

    leaky = set()
    for col in candidate_cols:
        if check_null_pattern:
            col_mask = df[col].isna().astype(int)
            if col_mask.nunique() == 2:
                corr = abs(np.corrcoef(treatment_mask, col_mask)[0, 1])
                if np.isfinite(corr) and corr >= threshold:
                    leaky.add(col)
                    continue

        col_values = df[col].fillna(0).to_numpy(dtype=float)
        if np.std(col_values) > 0:
            corr = abs(np.corrcoef(col_values, treatment_values)[0, 1])
            if np.isfinite(corr) and corr >= threshold:
                leaky.add(col)

    return leaky


def match_feature_columns(feature_cols, available_columns):
    """
    For each name in `feature_cols`, finds an exact or normalized-name match
    in `available_columns`. Returns a dict {feature_name: matched_column_or_None}.
    Used to check how much of an existing model's schema is actually present
    in a freshly uploaded dataset, before deciding whether retraining is
    even necessary.
    """
    available_norm = {_normalize_key(c): c for c in available_columns}
    matches = {}
    for name in feature_cols:
        if name in available_columns:
            matches[name] = name
        else:
            matches[name] = available_norm.get(_normalize_key(name))
    return matches


def select_top_features(df, treatment_col, treatment_binarized, top_n=30):
    """
    Fits a GradientBoostingClassifier on every numeric candidate column
    (minus leakage-correlated ones, see _leakage_correlated_columns) to rank
    importance for predicting `treatment_binarized`. Returns
    (top_n feature names, name->importance dict for every ranked candidate,
    sorted list of columns excluded as leakage-correlated).
    """
    ranked, leaky = _rank_candidate_features(df, treatment_col, treatment_binarized)
    top = ranked[:top_n]
    return [name for name, _ in top], {name: json_safe_float(imp) for name, imp in ranked}, sorted(leaky)


def _rank_candidate_features(df, treatment_col, treatment_binarized):
    candidate_cols = _numeric_candidate_columns(df, exclude={treatment_col})
    leaky = _leakage_correlated_columns(df, treatment_col, treatment_binarized, candidate_cols)
    candidate_cols = [c for c in candidate_cols if c not in leaky]
    if not candidate_cols:
        raise ValueError("no usable feature columns found (all numeric candidates were the treatment column or leakage-correlated with it)")

    X = df[candidate_cols].fillna(0).to_numpy(dtype=float)
    y = treatment_binarized.to_numpy()

    ranker = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
    ranker.fit(X, y)

    ranked = sorted(zip(candidate_cols, ranker.feature_importances_), key=lambda p: p[1], reverse=True)
    return ranked, leaky


def select_or_merge_features(df, treatment_col, treatment_binarized, previous_features=None, top_n=30):
    """
    Like select_top_features, but when `previous_features` (an existing
    model's schema) is given, keeps whichever of those are still present and
    usable in `df`, and only backfills the remaining slots with this
    dataset's own top-ranked features -- rather than discarding a proven
    feature set every time a new file comes in. Falls back to a plain
    top-`top_n` selection when no previous schema is given, or when none of
    it carries over (e.g. a structurally unrelated dataset).

    Returns (final_features, importances_dict_for_ranked, leaky_excluded,
    breakdown) where breakdown = {"kept_from_previous": [...],
    "added_new": [...], "dropped_from_previous": [...]}.
    """
    ranked, leaky = _rank_candidate_features(df, treatment_col, treatment_binarized)
    ranked_names = [name for name, _ in ranked]
    importance_by_name = dict(ranked)

    if not previous_features:
        top = ranked_names[:top_n]
        return top, {name: json_safe_float(imp) for name, imp in ranked}, sorted(leaky), {
            "kept_from_previous": [], "added_new": top, "dropped_from_previous": [],
        }

    usable_previous = [f for f in previous_features if f in importance_by_name]

    # Keep as many previous features as fit, prioritizing the ones that are
    # still most important in this dataset (not just insertion order).
    usable_previous.sort(key=lambda f: importance_by_name[f], reverse=True)
    kept = usable_previous[:top_n]
    dropped = [f for f in previous_features if f not in kept]

    remaining_slots = top_n - len(kept)
    added = [name for name in ranked_names if name not in kept][:max(remaining_slots, 0)]

    final_features = kept + added
    return final_features, {name: json_safe_float(imp) for name, imp in ranked}, sorted(leaky), {
        "kept_from_previous": kept, "added_new": added, "dropped_from_previous": dropped,
    }


def train_psm_model(df, treatment_binarized, feature_cols):
    """Fits the final propensity-score model on just `feature_cols`."""
    X = df[feature_cols].fillna(0).to_numpy(dtype=float)
    y = treatment_binarized.to_numpy()
    model = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
    model.fit(X, y)
    importances = {name: json_safe_float(imp) for name, imp in zip(feature_cols, model.feature_importances_)}
    return model, importances


def logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def _infer_feature_map(feature_cols, incoming_feature_keys):
    incoming_norm = {_normalize_key(k): k for k in incoming_feature_keys}
    norm_values = list(incoming_norm.keys())

    mapping = {}
    for mf in feature_cols:
        mf_norm = _normalize_key(mf)
        if mf_norm in incoming_norm:
            mapping[mf] = incoming_norm[mf_norm]
            continue
        best, best_score = None, 0.0
        for in_norm in norm_values:
            score = SequenceMatcher(None, mf_norm, in_norm).ratio()
            if score > best_score:
                best_score, best = score, incoming_norm[in_norm]
        if best is not None and best_score >= 0.72:
            mapping[mf] = best
    return mapping


def _extract_value(r, key):
    if isinstance(r, dict) and key in r:
        return r[key]
    return None


def validate_records(records, feature_cols, require_treatment=False, require_outcome=False,
                      treatmentKey="treatment", outcomeKey="outcome"):
    if not isinstance(records, list) or len(records) == 0:
        return "records must be a non-empty array"

    required_feature_set = set(feature_cols or [])
    for i, r in enumerate(records):
        if not isinstance(r, dict):
            return f"record at index {i} must be an object"

        if require_treatment and treatmentKey not in r:
            return f"record at index {i} missing required field: {treatmentKey}"
        if require_outcome and outcomeKey not in r:
            return f"record at index {i} missing required field: {outcomeKey}"

        if "features" in r and isinstance(r["features"], dict):
            incoming_keys = set(r["features"].keys())
        else:
            incoming_keys = set(r.keys())

        missing = []
        for mf in required_feature_set:
            if "features" in r and isinstance(r["features"], dict):
                if mf not in incoming_keys:
                    missing.append(mf)
            else:
                if mf not in r:
                    missing.append(mf)

        if missing and not ("features" in r and isinstance(r["features"], dict)):
            return f"record at index {i} missing features: {missing[:5]}{'...' if len(missing) > 5 else ''}"

    return None


def build_X_from_records(records, feature_cols, featureMap=None, auto_infer=True):
    first = records[0]
    if isinstance(first, dict) and "features" in first and isinstance(first["features"], dict):
        incoming_feature_keys = set(first["features"].keys())
        nested_features = True
    else:
        incoming_feature_keys = set(first.keys())
        nested_features = False

    fmap = featureMap or {}
    if (not fmap) and auto_infer:
        fmap = _infer_feature_map(feature_cols, incoming_feature_keys)

    missing_model_features = [mf for mf in feature_cols if mf not in fmap]
    if missing_model_features:
        if nested_features:
            return None, f"could not map model features missing: {missing_model_features[:10]}{'...' if len(missing_model_features) > 10 else ''}"
        fmap = {mf: mf for mf in feature_cols}

    X = []
    for r in records:
        row = []
        features_obj = r.get("features", {}) if isinstance(r, dict) else {}
        for mf in feature_cols:
            incoming_key = fmap[mf]
            if nested_features:
                if incoming_key not in features_obj:
                    return None, f"missing mapped incoming feature '{incoming_key}' for model feature '{mf}'"
                row.append(float(features_obj[incoming_key]))
            else:
                row.append(float(r[incoming_key]))
        X.append(row)

    return np.asarray(X, dtype=float), None


def predict_ps(model, feature_cols, records, featureMap=None, auto_infer=True):
    err = validate_records(records, feature_cols)
    if err:
        return None, err
    X, x_err = build_X_from_records(records, feature_cols, featureMap, auto_infer)
    if x_err:
        return None, x_err
    ps_final = model.predict_proba(X)[:, 1]
    return ps_final.tolist(), None


def estimate_att(model, feature_cols, records, featureMap=None, auto_infer=True,
                  caliper_ratio=0.2, n_bootstrap=500, seed=42,
                  treatmentKey="treatment", outcomeKey="outcome"):
    err = validate_records(records, feature_cols, require_treatment=True, require_outcome=True,
                            treatmentKey=treatmentKey, outcomeKey=outcomeKey)
    if err:
        return None, err

    treatments = np.asarray([int(r[treatmentKey]) for r in records], dtype=int)
    outcomes = np.asarray([float(r[outcomeKey]) for r in records], dtype=float)

    X, x_err = build_X_from_records(records, feature_cols, featureMap, auto_infer)
    if x_err:
        return None, x_err

    ps_final = model.predict_proba(X)[:, 1]
    ps_logit_final = logit(ps_final)

    caliper = caliper_ratio * np.std(ps_logit_final)
    if not np.isfinite(caliper) or caliper <= 0:
        return None, "invalid caliper computed from input data"

    control_mask = treatments == 0
    treat_mask = treatments == 1

    if control_mask.sum() == 0 or treat_mask.sum() == 0:
        return None, "need both treated and control records in input"

    control_ps = ps_logit_final[control_mask].reshape(-1, 1)
    treat_ps = ps_logit_final[treat_mask].reshape(-1, 1)

    knn = NearestNeighbors(n_neighbors=1, radius=caliper)
    knn.fit(control_ps)

    control_indices = np.where(control_mask)[0]
    treated_indices = np.where(treat_mask)[0]
    distances, indices = knn.kneighbors(treat_ps)

    matched_pairs = []
    for j in range(len(treated_indices)):
        if distances[j][0] <= caliper:
            matched_pairs.append((treated_indices[j], control_indices[indices[j][0]]))

    if len(matched_pairs) == 0:
        return {
            "matched_pairs": 0,
            "att_mean": None,
            "ci_95": None,
            "p_value_paired_ttest": None,
            "caliper": json_safe_float(caliper),
        }, None

    diffs, treat_outs, ctrl_outs = [], [], []
    for treat_idx, ctrl_idx in matched_pairs:
        treat_outs.append(outcomes[treat_idx])
        ctrl_outs.append(outcomes[ctrl_idx])
        diffs.append(outcomes[treat_idx] - outcomes[ctrl_idx])

    diffs = np.asarray(diffs, dtype=float)
    att_mean = float(np.mean(diffs))

    _, p_val = ttest_rel(np.asarray(ctrl_outs), np.asarray(treat_outs))
    p_val = float(p_val)

    rng = np.random.default_rng(int(seed))
    boot = []
    m = len(diffs)
    for _ in range(int(n_bootstrap)):
        sample = diffs[rng.integers(0, m, size=m)]
        boot.append(np.mean(sample))
    ci_low, ci_high = np.percentile(np.asarray(boot), [2.5, 97.5])

    return {
        "matched_pairs": int(len(matched_pairs)),
        "att_mean": json_safe_float(att_mean),
        "ci_95": [json_safe_float(ci_low), json_safe_float(ci_high)],
        "p_value_paired_ttest": json_safe_float(p_val),
        "caliper": json_safe_float(caliper),
    }, None
