"""Propensity-score-matching (PSM) logic shared by app.py's endpoints.

app.py is grounded in a frozen bfar.csv baseline (models/best_model.pkl,
models/scaler.pkl, models/all_features.json, models/core_features.json,
models/remaining_features.json -- produced by build_model.py) and never retrains or
overwrites it: requests covering all baseline features score against it directly,
and dynamic feature adaptation for partial requests (see predict_dynamic) fits a
throwaway model per request, scoped to that request only.
"""
import re

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from scipy.stats import ttest_rel


_ID_LIKE_NAME = re.compile(r"(^|_)(id|uuid|guid|index)($|_)", re.IGNORECASE)
_TREATMENT_NAME_HINTS = (
    "treat", "program", "particip", "enroll", "assist", "benefic",
    "recipient", "grant", "subsid", "loan", "interven",
)
# Model types whose training data was standardized -- their predict_proba
# expects scaled input too. Tree/boosting models split on raw thresholds
# learned during training, so scaling them at predict time silently corrupts
# results (verified empirically against bfar_with_ps.csv: applying the saved
# scaler to the saved GradientBoostingClassifier moves predictions off the
# ground truth, while skipping it reproduces it exactly).
_SCALING_REQUIRED_MODELS = {"MLPClassifier"}
# Fitting a classifier to rank flex features on a handful of rows produces
# noise, not a ranking -- mirrors the old /train endpoint's MIN_TRAINING_ROWS
# guard.
MIN_DYNAMIC_ADAPT_ROWS = 10
# Feature-set size for the fully data-driven fallback (dataset shares no core
# baseline columns at all) -- same top-N the core set itself was built with.
FALLBACK_TOP_N = 30


def json_safe_float(value):
    """Converts NaN/inf to None so responses stay valid JSON for strict clients."""
    value = float(value)
    return value if np.isfinite(value) else None


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


def model_needs_scaling(model):
    """Whether `model`'s predict_proba expects standardized input (see
    _SCALING_REQUIRED_MODELS)."""
    return type(model).__name__ in _SCALING_REQUIRED_MODELS


def impute_dataframe(df, columns):
    """Median-impute numeric columns, mode-impute object columns. Returns a
    copy; leaves columns not present in `df` untouched."""
    df = df.copy()
    for col in columns:
        if col not in df.columns:
            continue
        if df[col].dtype == "object":
            mode = df[col].mode()
            df[col] = df[col].fillna(mode.iloc[0] if len(mode) else "")
        else:
            df[col] = df[col].fillna(df[col].median())
    return df


def select_flex_features(df, core_features, treatment_binarized, exclude=(), n_flex=27):
    """
    Ranks numeric columns outside `core_features`/`exclude` by importance for
    predicting `treatment_binarized`, via a throwaway GradientBoostingClassifier
    fit on core+candidates jointly (importances are read relative to that
    joint fit, matching predictor_psm.ipynb's dynamic selection exactly --
    ranking candidates in isolation would give different scores). Returns up
    to `n_flex` candidate column names, highest importance first.
    """
    exclude_set = set(core_features) | set(exclude)
    candidate_cols = [
        c for c in df.columns
        if c not in exclude_set and pd.api.types.is_numeric_dtype(df[c])
    ]
    if not candidate_cols:
        return []

    combined = core_features + candidate_cols
    X_temp = impute_dataframe(df, combined)[combined]
    gb_temp = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
    gb_temp.fit(X_temp, treatment_binarized)

    importances = pd.Series(gb_temp.feature_importances_, index=combined)
    ranked_candidates = importances[candidate_cols].sort_values(ascending=False)
    return ranked_candidates.head(min(n_flex, len(ranked_candidates))).index.tolist()


def predict_dynamic(df, core_features, all_features, baseline_model, baseline_scaler,
                     treatment_col="treatment", n_flex=27, exclude_cols=()):
    """
    Baseline-first propensity score prediction. bfar.csv is the fixed
    baseline (models/best_model.pkl + models/scaler.pkl + models/*_features.json,
    trained once by build_model.py, never touched again by either service):

      - Dataset covers all of `all_features` -> `baseline_model` scores it
        directly. No fitting happens on this request at all.
      - Dataset covers `core_features` but not all of `all_features` -> core
        stays fixed, and up to `n_flex` of the dataset's OWN top-ranked
        numeric columns (by importance for `treatment_col`, see
        select_flex_features) fill out the remaining slots; a throwaway model
        is fit on core+flex for THIS request only and discarded afterwards --
        nothing is ever written back to the baseline or reused by a later
        request. Requires `treatment_col` to be present (there is no labeled
        target to rank candidate features against otherwise). `exclude_cols`
        is subtracted from flex candidates too -- callers with a separate
        outcome column (estimate_att_dynamic) must pass it here, or it'll get
        picked as a "predictor" of treatment, which is nonsense.
      - Core features partially or entirely missing -> fully data-driven
        fallback: every numeric column (minus exclusions) is ranked against
        `treatment_col` and the top FALLBACK_TOP_N form the feature set.
        Same throwaway-model semantics; the result's `core_coverage` tells
        the caller how much of the trusted core actually backed this score.

    The baseline branch does NOT require `treatment_col` -- real-world
    /predict_ps calls are usually scoring someone whose treatment status is
    exactly what's unknown, and the fixed baseline model needs no labels to
    run. This is a deliberate relaxation of predictor_psm.ipynb (which
    requires the column unconditionally); the dynamic-adaptation branches
    still require it, since it can't be avoided there.

    Returns a dict: {ps, ps_logit, used_baseline, final_features,
    core_coverage, model, X}. `model`/`X` (a DataFrame indexed like `df`)
    let callers reuse the exact fitted model/matrix for further work (e.g.
    matched ATT) instead of re-deriving them.
    """
    cols_present = set(df.columns)
    core_present = [f for f in core_features if f in cols_present]
    core_coverage = len(core_present) / len(core_features) if core_features else 0.0

    used_baseline = all(f in cols_present for f in all_features)

    if used_baseline:
        final_features = all_features
        X = impute_dataframe(df, final_features)[final_features]
        model = baseline_model
        X_input = baseline_scaler.transform(X) if model_needs_scaling(model) else X
    else:
        if treatment_col not in df.columns:
            n_covered = len(cols_present & set(all_features))
            raise ValueError(
                f"dataset covers only {n_covered}/{len(all_features)} baseline features; selecting "
                f"extra features from this dataset requires a '{treatment_col}' column to rank them "
                f"against -- include it, or upload all {len(all_features)} baseline features."
            )
        if len(df) < MIN_DYNAMIC_ADAPT_ROWS:
            raise ValueError(
                f"dataset has only {len(df)} row(s); selecting extra features dynamically needs at "
                f"least {MIN_DYNAMIC_ADAPT_ROWS} rows to rank them reliably -- upload all "
                f"{len(all_features)} baseline features instead, or provide a larger dataset."
            )
        treatment_binarized = df[treatment_col].astype(int)
        if core_coverage == 1.0:
            flex = select_flex_features(df, core_features, treatment_binarized,
                                         exclude={treatment_col} | set(exclude_cols), n_flex=n_flex)
            final_features = core_features + flex
        else:
            # Fully data-driven fallback: nothing (or not everything) from
            # the trusted core is present, so rank ALL numeric columns on
            # their own merit -- surviving core columns compete like any
            # other candidate and rank in if they earn it.
            final_features = select_flex_features(df, [], treatment_binarized,
                                                   exclude={treatment_col} | set(exclude_cols),
                                                   n_flex=FALLBACK_TOP_N)
            if not final_features:
                raise ValueError("no usable numeric feature columns found in this dataset")
        X = impute_dataframe(df, final_features)[final_features]
        scaler = StandardScaler()
        X_input = scaler.fit_transform(X)
        model = GradientBoostingClassifier(n_estimators=200, learning_rate=0.1, max_depth=5, random_state=42)
        model.fit(X_input, treatment_binarized)

    ps = model.predict_proba(X_input)[:, 1]
    return {
        "ps": ps,
        "ps_logit": logit(ps),
        "used_baseline": used_baseline,
        "final_features": final_features,
        "core_coverage": core_coverage,
        "model": model,
        "X": X,
    }


def decision_support_table(df_with_ps, key_features=None, ps_col="ps"):
    """
    Stratifies rows into PS quartiles and summarizes each group -- the
    "which beneficiaries look like priority cases" view from
    predictor_psm.ipynb's decision-support step.
    """
    df = df_with_ps.copy()

    # A near-perfectly separating model collapses the PS distribution into
    # fewer than 4 distinct quantile bins (duplicates="drop" merges them),
    # so the quartile labels can't be assumed -- label whatever bins survive.
    try:
        codes = pd.qcut(df[ps_col], q=4, labels=False, duplicates="drop")
        n_bins = int(codes.max()) + 1 if len(codes) else 0
    except ValueError:
        n_bins = 0

    interpretation = {
        "Low": "Very low likelihood - may need targeted outreach",
        "Med-Low": "Below average - consider monitoring",
        "Med-High": "Above average - likely beneficiaries",
        "High": "High likelihood - priority for intervention",
    }
    if n_bins == 4:
        labels = ["Low", "Med-Low", "Med-High", "High"]
    elif n_bins >= 2:
        labels = [f"Group {i + 1} (of {n_bins})" for i in range(n_bins)]
        interpretation = {label: "PS distribution too concentrated for quartile stratification - groups are coarser quantiles" for label in labels}
    else:
        labels = ["All"]
        codes = pd.Series(0, index=df.index)
        interpretation = {"All": "PS distribution has no spread - stratification not meaningful"}

    df["ps_group"] = [labels[int(c)] for c in codes]

    key_features = [f for f in (key_features or []) if f in df.columns]
    agg = {"Count": (ps_col, "count"), "Mean_PS": (ps_col, "mean")}
    agg.update({f"Mean_{f}": (f, "mean") for f in key_features})
    table = df.groupby("ps_group", observed=False).agg(**agg).reset_index()

    table["Interpretation"] = table["ps_group"].map(interpretation)
    # Present groups in PS order, not alphabetical.
    order = {label: i for i, label in enumerate(labels)}
    return table.sort_values("ps_group", key=lambda s: s.map(order)).reset_index(drop=True)


def logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def _matched_att(ps_logit_final, treatments, outcomes, caliper_ratio=0.2, n_bootstrap=500, seed=42):
    """Nearest-neighbor PS matching (within a logit-scale caliper) + paired
    t-test + bootstrap CI for the ATT. Used by estimate_att_dynamic --
    everything upstream of this just needs to produce a ps_logit array plus
    aligned treatment/outcome arrays."""
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


def estimate_att_dynamic(df, core_features, all_features, baseline_model, baseline_scaler,
                          treatment_col="treatment", outcome_col="outcome", n_flex=27,
                          caliper_ratio=0.2, n_bootstrap=500, seed=42):
    """Takes a whole DataFrame (already has treatment+outcome columns),
    resolves baseline-vs-ephemeral-adapt via predict_dynamic, then runs the
    matched-ATT computation (_matched_att)."""
    if treatment_col not in df.columns or outcome_col not in df.columns:
        return None, f"dataset must include '{treatment_col}' and '{outcome_col}' columns"

    try:
        result = predict_dynamic(df, core_features, all_features, baseline_model, baseline_scaler,
                                  treatment_col=treatment_col, n_flex=n_flex, exclude_cols={outcome_col})
    except ValueError as e:
        return None, str(e)

    treatments = df[treatment_col].astype(int).to_numpy()
    outcomes = pd.to_numeric(df[outcome_col], errors="coerce").to_numpy(dtype=float)

    att_result, err = _matched_att(result["ps_logit"], treatments, outcomes, caliper_ratio, n_bootstrap, seed)
    if err:
        return None, err

    att_result["used_baseline"] = result["used_baseline"]
    att_result["final_features"] = result["final_features"]
    att_result["core_coverage"] = result["core_coverage"]
    return att_result, None


def predict_with_index_model(df, mapping, taxonomy, index_stats, index_model, index_scaler):
    """Tier-2 scoring: folds `df` into the 6 composite indices via a
    registered `mapping` (see psm_indices.compute_indices) and scores with
    the frozen index-space baseline model. No fitting, no labels needed --
    same result-dict shape as predict_dynamic so _matched_att and the
    endpoints reuse it unchanged, plus `imputed_indices` (indices with no
    mapped items, filled with bfar's median -- consumers should see how much
    of the score rests on real columns)."""
    from psm_indices import compute_indices

    X, imputed = compute_indices(df, mapping, taxonomy, index_stats)
    X_input = index_scaler.transform(X) if model_needs_scaling(index_model) else X
    ps = index_model.predict_proba(X_input)[:, 1]
    return {
        "ps": ps,
        "ps_logit": logit(ps),
        "used_baseline": True,
        "final_features": list(X.columns),
        "core_coverage": None,
        "imputed_indices": imputed,
        "model": index_model,
        "X": X,
    }


def estimate_att_with_index_model(df, mapping, taxonomy, index_stats, index_model, index_scaler,
                                    treatment_col="treatment", outcome_col="outcome",
                                    caliper_ratio=0.2, n_bootstrap=500, seed=42):
    """Tier-2 counterpart to estimate_att_dynamic: index-space propensity
    scores from the frozen index model, then the same matched-ATT
    computation."""
    if treatment_col not in df.columns or outcome_col not in df.columns:
        return None, f"dataset must include '{treatment_col}' and '{outcome_col}' columns"

    result = predict_with_index_model(df, mapping, taxonomy, index_stats, index_model, index_scaler)

    treatments = df[treatment_col].astype(int).to_numpy()
    outcomes = pd.to_numeric(df[outcome_col], errors="coerce").to_numpy(dtype=float)

    att_result, err = _matched_att(result["ps_logit"], treatments, outcomes, caliper_ratio, n_bootstrap, seed)
    if err:
        return None, err

    att_result["used_baseline"] = True
    att_result["final_features"] = result["final_features"]
    att_result["imputed_indices"] = result["imputed_indices"]
    return att_result, None
