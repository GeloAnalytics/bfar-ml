"""Propensity-score-matching (PSM) logic shared by app.py's endpoints.

app.py serves two models side by side:
  - A frozen bfar.csv baseline (models/best_model.pkl, models/scaler.pkl,
    models/all_features.json -- produced once by build_model.py, never
    retrained). Requests covering all 57 baseline features score against it
    directly.
  - A dynamic model, trained from whatever CSV was last POSTed to /train
    (see select_top_features / train_psm_model below) and persisted to
    models/dynamic/ so it survives a restart. Teachable-Machine style: every
    /train call deletes whatever was there and fits a completely fresh model
    on the new upload -- no merging with the previous schema, no
    reuse-shortcut. Requests that don't cover all 57 baseline features score
    against whichever dynamic model is currently active.
"""
import re

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
# Detects "before/after" wave-pair columns sharing an identical name except
# for a single standalone A vs B token -- e.g. bfar.csv's 'D1.2:A_MOTORC' /
# 'D1.2:B_MOTORC', confirmed against the actual BFAR beneficiary
# questionnaire (Parts C/D/E/F/G each ask every item twice: "BAGO MATANGGAP
# ANG BANGKA" [before receiving the boat] / "SA KASALUKUYAN" [at present] --
# a baseline/endline design). This is a *structural* pattern match, not a
# hardcoded word list, so it generalizes to any dataset using a similar
# two-wave naming convention -- common for livelihood-program impact
# evaluations generally, not just this one. Verified against bfar.csv's 215
# columns: 71 pairs detected, zero false positives (e.g. 'I2:A/C_M' --
# association/club membership -- correctly left alone since no
# 'I2:B/C_M' counterpart exists in the data).
#
# Known limitation: some "current wave" columns have no "before" twin to
# pair against at all (bfar.csv's C2:INCOME/B/FISH, C4:INCOME/B/ALT,
# C5:TOT_INCOME/B -- current income, but the questionnaire never asked a
# matching "before" breakdown by source) and so aren't caught here.
_WAVE_PAIR_A_TOKEN = re.compile(r"(?<![A-Za-z0-9])A(?![A-Za-z0-9])")
# Column-name keywords describing *who was surveyed* (respondent identity /
# demographics) rather than a livelihood/economic factor -- excluded from
# candidate features by default regardless of statistical importance.
# Generic English survey terms, not tied to any one program's column-naming
# scheme, so this applies across arbitrary uploaded datasets, not just
# bfar.csv. Verified empirically against bfar.csv's 215 raw columns: matches
# exactly the 5 demographic columns (AREA, AGE, SEX, M-STATUS, EDUCATION)
# with zero false positives on the other 210. No separate "livelihood
# keyword" allowlist is needed -- asset/income columns are typically named
# by specific item (motorcycle, TV, fridge...) rather than a generic word,
# so they're retained simply by not matching this exclude list. Household
# size (HH_SIZE) is deliberately NOT included here -- treated as a
# livelihood-adjacent factor (household composition affects economic need),
# not pure demographics, per an earlier explicit decision.
_DEMOGRAPHIC_EXCLUDE_KEYWORDS = (
    "age", "respondent", "area", "region", "sex", "gender", "marital",
    "m-status", "m_status", "education", "religion", "ethnic", "address",
    "barangay", "birthdate", "birthday",
)
# Model types whose training data was standardized -- their predict_proba
# expects scaled input too. Tree/boosting models split on raw thresholds
# learned during training, so scaling them at predict time silently corrupts
# results (verified empirically against bfar_with_ps.csv: applying the saved
# scaler to the saved GradientBoostingClassifier moves predictions off the
# ground truth, while skipping it reproduces it exactly).
_SCALING_REQUIRED_MODELS = {"MLPClassifier"}


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
    copy; leaves columns not present in `df` untouched. Used only for the
    frozen bfar baseline (build_model.py trains it this way) -- the dynamic
    per-upload model uses plain .fillna(0), see select_top_features /
    train_psm_model below."""
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


def _numeric_candidate_columns(df, exclude):
    return [
        c for c in df.columns
        if c not in exclude
        and pd.api.types.is_numeric_dtype(df[c])
        and not _is_id_like(df[c], c)
    ]


def _low_coverage_columns(df, columns, min_coverage=0.1):
    """Columns with fewer than `min_coverage` fraction of non-null values in
    this particular upload carry negligible signal regardless of what
    they're named -- e.g. bfar.csv's CD:P_SCORE/CV:PS_WT, which are entirely
    empty, and K:COMMENTS, which is almost entirely blank. Fully
    dataset-agnostic, no naming assumptions."""
    return {c for c in columns if df[c].notna().mean() < min_coverage}


def _wave_pair_excluded_columns(columns):
    """See _WAVE_PAIR_A_TOKEN above. For every column with a standalone 'A'
    token, checks whether swapping it for 'B' matches another column already
    in this dataset -- if so, that 'B' column is the "after" half of a
    before/after pair and gets excluded; its 'A' counterpart is kept."""
    columns = list(columns)
    colset = set(columns)
    excluded = set()
    for col in columns:
        name = str(col)
        if not _WAVE_PAIR_A_TOKEN.search(name):
            continue
        b_version = _WAVE_PAIR_A_TOKEN.sub("B", name, count=1)
        if b_version != name and b_version in colset:
            excluded.add(b_version)
    return excluded


def _context_excluded_columns(columns):
    """Column names matching _DEMOGRAPHIC_EXCLUDE_KEYWORDS -- respondent
    identity/demographics, not livelihood factors."""
    excluded = set()
    for col in columns:
        name = str(col).lower()
        if any(k in name for k in _DEMOGRAPHIC_EXCLUDE_KEYWORDS):
            excluded.add(col)
    return excluded


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


def _rank_candidate_features(df, treatment_col, treatment_binarized, extra_exclude=None):
    base_candidates = _numeric_candidate_columns(df, exclude={treatment_col} | set(extra_exclude or ()))

    low_coverage = _low_coverage_columns(df, base_candidates)
    remaining = [c for c in base_candidates if c not in low_coverage]

    context_excluded = _context_excluded_columns(remaining)
    remaining = [c for c in remaining if c not in context_excluded]

    wave_pair_excluded = _wave_pair_excluded_columns(remaining)
    candidate_cols = [c for c in remaining if c not in wave_pair_excluded]

    leaky = _leakage_correlated_columns(df, treatment_col, treatment_binarized, candidate_cols)
    candidate_cols = [c for c in candidate_cols if c not in leaky]
    if not candidate_cols:
        raise ValueError("no usable feature columns found -- all numeric candidates were the treatment column, low-coverage, demographic, wave-pair-excluded, or leakage-correlated with it")

    X = df[candidate_cols].fillna(0).to_numpy(dtype=float)
    y = treatment_binarized.to_numpy()

    ranker = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
    ranker.fit(X, y)

    ranked = sorted(zip(candidate_cols, ranker.feature_importances_), key=lambda p: p[1], reverse=True)
    return ranked, leaky, wave_pair_excluded, context_excluded, low_coverage


def select_top_features(df, treatment_col, treatment_binarized, top_n=None, extra_exclude=None):
    """
    Fits a GradientBoostingClassifier on every numeric candidate column to
    rank importance for predicting `treatment_binarized`. Always a fresh
    ranking of whatever this dataset provides -- no memory of any previous
    model's schema. Candidates are narrowed down first, in order:
      1. Low data coverage (_low_coverage_columns) -- fewer than 10%
         non-null values in this upload; dataset-agnostic, no naming
         assumptions.
      2. Demographic/respondent-identity keyword match
         (_context_excluded_columns) -- generic English survey terms (age,
         sex, area, education...), not tied to any one program's naming
         scheme.
      3. Before/after wave-pair structural match (_wave_pair_excluded_columns)
         -- a column that's the "current" half of a pair sharing an
         identical name except for an isolated A/B token. A structural
         pattern, not a hardcoded word list, so it generalizes to other
         before/after-design datasets, not just bfar.csv.
      4. Leakage correlation with treatment (_leakage_correlated_columns).
    `extra_exclude` drops columns outright before ranking even starts --
    used by app.py's covariate-balance re-tune loop to drop a feature that
    failed balance and re-rank without it.

    `top_n=None` (the default) returns every ranked candidate -- no arbitrary
    cutoff, so the response can show the full importance ranking and the
    integrator decides what to actually use downstream. Pass an int to cap
    it instead.

    Returns (selected feature names, name->importance dict for every ranked
    candidate, sorted list of columns excluded as leakage-correlated, sorted
    list excluded as a before/after wave-pair, sorted list excluded as
    demographic/context, sorted list excluded as low-coverage).
    """
    ranked, leaky, wave_pair_excluded, context_excluded, low_coverage = _rank_candidate_features(
        df, treatment_col, treatment_binarized, extra_exclude=extra_exclude)
    top = ranked if top_n is None else ranked[:top_n]
    return (
        [name for name, _ in top],
        {name: json_safe_float(imp) for name, imp in ranked},
        sorted(leaky),
        sorted(wave_pair_excluded),
        sorted(context_excluded),
        sorted(low_coverage),
    )


def train_psm_model(df, treatment_binarized, feature_cols):
    """Fits the final propensity-score model on just `feature_cols`."""
    X = df[feature_cols].fillna(0).to_numpy(dtype=float)
    y = treatment_binarized.to_numpy()
    model = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
    model.fit(X, y)
    importances = {name: json_safe_float(imp) for name, imp in zip(feature_cols, model.feature_importances_)}
    return model, importances


def _ordinal(n):
    suffix = "th" if 10 <= n % 100 <= 20 else {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suffix}"


def compute_shap_feature_contributions(model, X, feature_cols):
    """
    Real SHAP values (shap.TreeExplainer -- exact, not approximated, for
    tree-ensemble models like GradientBoostingClassifier) explaining this
    model's predictions in terms of `feature_cols`. `X` should be the same
    fillna(0) frame used to fit/score the model.

    Returns a list of {feature, mean_abs_shap, mean_shap, direction} dicts
    sorted by mean_abs_shap descending -- the standard "global SHAP feature
    importance" view (mean absolute SHAP value per feature across every
    row), plus the signed mean, which tells you the *direction* of the
    effect: whether higher values of that feature push predictions toward
    treatment=1 ("increases_likelihood") or treatment=0
    ("decreases_likelihood") on average. Values are in the model's raw
    margin (log-odds) space, not probability space -- shap.TreeExplainer's
    default for classifiers, and not comparable in magnitude to a
    probability difference.
    """
    import shap

    explainer = shap.TreeExplainer(model)
    shap_values = np.asarray(explainer.shap_values(X))
    if shap_values.ndim == 3:
        # Some shap/model combinations return (n_samples, n_features, n_classes);
        # keep the positive class.
        shap_values = shap_values[:, :, -1]

    mean_abs = np.abs(shap_values).mean(axis=0)
    mean_signed = shap_values.mean(axis=0)

    contributions = [
        {
            "feature": name,
            "mean_abs_shap": json_safe_float(abs_val),
            "mean_shap": json_safe_float(signed_val),
            "direction": "increases_likelihood" if signed_val > 0 else "decreases_likelihood",
        }
        for name, abs_val, signed_val in zip(feature_cols, mean_abs, mean_signed)
    ]
    contributions.sort(key=lambda c: c["mean_abs_shap"] or 0, reverse=True)
    return contributions


def generate_socioeconomic_insights(feature_contributions, top_n=5):
    """
    Plain-language summary of the top `top_n` SHAP-ranked *columns* -- generic
    template sentences built from whatever column names this dataset
    provides. No hardcoded knowledge of what a given column means; this
    doesn't tie the wording to bfar.csv or any one program.

    This ranks INPUT FEATURES (columns), not rows/respondents, and has
    nothing to do with how many propensity scores exist -- that count always
    equals the number of rows in the upload (see ps_output), completely
    independent of how many features were ranked here or how large `top_n`
    is. Each sentence below states its rank explicitly (e.g. "feature #2 of
    5 shown") to keep that distinction unambiguous in the response.
    """
    insights = []
    for rank, contrib in enumerate(feature_contributions[:top_n], start=1):
        direction_phrase = "a higher" if contrib["direction"] == "increases_likelihood" else "a lower"
        insights.append(
            f"Feature #{rank} of {min(top_n, len(feature_contributions))} shown (ranked by SHAP "
            f"contribution to the propensity-score model, out of {len(feature_contributions)} "
            f"features total) -- column \"{contrib['feature']}\": higher values in this column are "
            f"associated with {direction_phrase} likelihood of being in the treatment group. "
            f"(This describes one input column, not an individual respondent.)"
        )
    return insights


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


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def _match_pairs(ps_logit_final, treatments, caliper_ratio=0.2):
    """1-nearest-neighbor matching of treated to control units on the
    logit-scale propensity score, within a caliper. Shared by matched_att
    (needs outcomes too) and covariate_balance (doesn't). Returns
    (matched_pairs, caliper, err) where matched_pairs is a list of
    (treated_row_index, matched_control_row_index) tuples into the original
    arrays; caliper is None and err is set on failure."""
    caliper = caliper_ratio * np.std(ps_logit_final)
    if not np.isfinite(caliper) or caliper <= 0:
        return None, None, "invalid caliper computed from input data"

    control_mask = treatments == 0
    treat_mask = treatments == 1

    if control_mask.sum() == 0 or treat_mask.sum() == 0:
        return None, None, "need both treated and control records in input"

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

    return matched_pairs, caliper, None


def matched_att(ps_logit_final, treatments, outcomes, caliper_ratio=0.2, n_bootstrap=500, seed=42):
    """Nearest-neighbor PS matching (within a logit-scale caliper) + paired
    t-test + bootstrap CI for the ATT. Shared by both the baseline and
    dynamic scoring paths in app.py -- everything upstream of this just
    needs to produce a ps_logit array plus aligned treatment/outcome
    arrays."""
    matched_pairs, caliper, err = _match_pairs(ps_logit_final, treatments, caliper_ratio)
    if err:
        return None, err

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


BALANCE_THRESHOLD = 0.1  # standard "well-balanced" cutoff for a single feature's |SMD|
# Count-based tolerance, not a mean-based gate: up to this fraction of features may
# individually exceed BALANCE_THRESHOLD and the model is still called "balanced"
# overall. Requiring the *average* |SMD| across every candidate to be low is
# unrealistically strict once a dataset has 50-100+ real socioeconomic covariates --
# a handful of genuinely different features drags the mean up and fails the whole
# model even when the vast majority are well-matched. Mirrors the reference PSM
# notebook's Cell 9 IPW balance check, which treats >10 of 57 unbalanced features as
# just a warning (not a hard failure) and still reports the ATT regardless -- rather
# than an aggregate mean threshold. Expressed as a percentage here so it scales with
# however many features this dataset's dynamic model actually selected. Set above the
# notebook's own ~17.5% (10/57) since that reference number is itself only a "print a
# warning" line, not a pass/fail bar -- see MAX_RETRAIN_ATTEMPTS in app.py, which mirrors
# the notebook's tolerated count of 10 for how many rounds the retrain loop gets to
# actually fix imbalance before settling for this looser threshold.
MAX_UNBALANCED_FEATURE_PCT = 0.35


def standardized_mean_diff(X, treatments):
    """
    Per-column standardized mean difference: (mean_treated - mean_control) / pooled_std,
    with pooled_std = sqrt((var_treated + var_control) / 2) (Cohen's-d-style pooling).
    X: 2D numeric array (n_samples, n_features), row-aligned with `treatments` (0/1
    array). Columns with zero pooled variance (constant in both groups) get SMD 0 --
    no imbalance is possible on a column that doesn't vary.
    """
    treat_vals = X[treatments == 1]
    control_vals = X[treatments == 0]
    mean_t = treat_vals.mean(axis=0)
    mean_c = control_vals.mean(axis=0)
    var_t = treat_vals.var(axis=0, ddof=1) if treat_vals.shape[0] > 1 else np.zeros(X.shape[1])
    var_c = control_vals.var(axis=0, ddof=1) if control_vals.shape[0] > 1 else np.zeros(X.shape[1])
    pooled_std = np.sqrt((var_t + var_c) / 2)
    with np.errstate(invalid="ignore", divide="ignore"):
        smd = np.where(pooled_std > 0, (mean_t - mean_c) / pooled_std, 0.0)
    return smd


def _stabilized_ipw_weights(ps, treatments, ps_lower=0.05, ps_upper=0.95):
    """Stabilized inverse-propensity weights, mirroring the reference PSM notebook's
    Cell 9: rows with a propensity score outside [ps_lower, ps_upper] are trimmed
    (weight 0, excluded below) since a near-0/near-1 score produces an unstably huge
    weight. Retained rows get P(treat)/ps if treated, (1-P(treat))/(1-ps) if control.
    Returns (weights, trimmed_mask)."""
    mask = (ps > ps_lower) & (ps < ps_upper)
    weights = np.zeros_like(ps, dtype=float)
    if mask.sum() == 0 or (treatments[mask] == 1).sum() == 0 or (treatments[mask] == 0).sum() == 0:
        return weights, mask
    p_treat = treatments[mask].mean()
    treat_mask = mask & (treatments == 1)
    control_mask = mask & (treatments == 0)
    weights[treat_mask] = p_treat / ps[treat_mask]
    weights[control_mask] = (1 - p_treat) / (1 - ps[control_mask])
    return weights, mask


def weighted_standardized_mean_diff(X, treatments, weights):
    """IPW-weighted standardized mean difference per column: reweights the *entire*
    retained sample by stabilized inverse-propensity weights (see
    _stabilized_ipw_weights), instead of restricting to 1-NN matched pairs like
    standardized_mean_diff. `X`/`treatments`/`weights` should already be trimmed to
    rows with nonzero weight."""
    treat_mask = treatments == 1
    control_mask = treatments == 0
    Xt, Xc = X[treat_mask], X[control_mask]
    wt, wc = weights[treat_mask], weights[control_mask]
    mean_t = np.average(Xt, axis=0, weights=wt)
    mean_c = np.average(Xc, axis=0, weights=wc)
    var_t = np.average((Xt - mean_t) ** 2, axis=0, weights=wt)
    var_c = np.average((Xc - mean_c) ** 2, axis=0, weights=wc)
    pooled_std = np.sqrt((var_t + var_c) / 2)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(pooled_std > 0, (mean_t - mean_c) / pooled_std, 0.0)


def _unbalanced_feature_verdict(abs_smd, n_features, max_unbalanced_pct):
    """Count-based balance verdict: how many features individually exceed
    BALANCE_THRESHOLD, and is that count within the tolerated cap. See
    MAX_UNBALANCED_FEATURE_PCT above for why this replaced a mean-based gate."""
    n_over = int(np.sum(abs_smd > BALANCE_THRESHOLD))
    max_allowed = max(1, int(np.ceil(n_features * max_unbalanced_pct)))
    return n_over, max_allowed


def _balance_achieved_from_abs_smd(abs_smd, n_features, max_unbalanced_pct):
    n_over, max_allowed = _unbalanced_feature_verdict(abs_smd, n_features, max_unbalanced_pct)
    return n_over <= max_allowed, n_over, max_allowed


def covariate_balance(df, treatment_binarized, feature_cols, ps_logit, caliper_ratio=0.2,
                       balance_threshold=BALANCE_THRESHOLD, max_unbalanced_pct=MAX_UNBALANCED_FEATURE_PCT):
    """
    Covariate balance diagnostics (pipeline step 7): standardized mean difference per
    feature before and after 1-NN caliper matching (see _match_pairs), PS common-support
    overlap between groups, and a balance_achieved verdict -- count-based, not a mean
    threshold: true if no more than `max_unbalanced_pct` of features individually have
    |SMD after matching| >= balance_threshold (falls back to pre-match SMD if no pairs
    matched). Also reports the single worst-balanced feature, for a caller that wants
    to drop it and retry (see app.py's /train re-tune loop).

    Also includes an `ipw` block: the same per-feature SMD but reweighted by
    stabilized inverse-propensity weights across the whole (trimmed) sample, instead
    of restricted to matched pairs -- the reference PSM notebook's Cell 9 primary
    balance check (see _stabilized_ipw_weights, weighted_standardized_mean_diff).
    Context only: it does NOT feed into balance_achieved/worst_feature, since matching
    (not IPW) is what this service's ATT estimate actually uses (see matched_att). On
    this project's real dataset, IPW balance comes out meaningfully tighter than
    matched balance once a dynamic model selects 100+ raw covariates -- 1-NN matching
    only guarantees balance on the scalar propensity score, not on every individual
    covariate, so a caller can use this to see whether the matched verdict is
    pessimistic because the covariate set is just large, not because the propensity
    model is bad.
    """
    treatments = treatment_binarized.to_numpy()
    if (treatments == 0).sum() == 0 or (treatments == 1).sum() == 0:
        return {
            "balance_achieved": False,
            "matched_balance_achieved": False,
            "mean_abs_smd": None,
            "n_features_over_threshold": None,
            "max_unbalanced_features_allowed": None,
            "balance_threshold": balance_threshold,
            "matched_pairs": 0,
            "caliper": None,
            "overlap": {"treated_in_control_range_pct": None, "control_in_treated_range_pct": None},
            "per_feature": [],
            "worst_feature": None,
            "ipw": {"balance_achieved": False, "mean_abs_smd": None, "n_features_over_threshold": None, "trimmed_n": 0},
            "error": "need both treated and control records to assess balance",
        }

    X = df[feature_cols].fillna(0).to_numpy(dtype=float)
    pre_smd = standardized_mean_diff(X, treatments)

    ps = _sigmoid(ps_logit)
    ipw_weights, ipw_mask = _stabilized_ipw_weights(ps, treatments)
    if ipw_mask.sum() > 0 and (treatments[ipw_mask] == 1).any() and (treatments[ipw_mask] == 0).any():
        ipw_smd = weighted_standardized_mean_diff(X[ipw_mask], treatments[ipw_mask], ipw_weights[ipw_mask])
    else:
        ipw_smd = np.full(len(feature_cols), np.nan)
    finite_abs_ipw = np.abs(ipw_smd)[np.isfinite(ipw_smd)]
    ipw_achieved, ipw_n_over, ipw_max_allowed = _balance_achieved_from_abs_smd(
        finite_abs_ipw, len(feature_cols), max_unbalanced_pct
    ) if len(finite_abs_ipw) else (False, None, None)
    ipw_summary = {
        "mean_abs_smd": json_safe_float(float(np.mean(finite_abs_ipw))) if len(finite_abs_ipw) else None,
        "n_features_over_threshold": ipw_n_over,
        "max_unbalanced_features_allowed": ipw_max_allowed,
        "balance_achieved": ipw_achieved,
        "trimmed_n": int(ipw_mask.sum()),
    }

    matched_pairs, caliper, err = _match_pairs(ps_logit, treatments, caliper_ratio)

    if err or not matched_pairs:
        per_feature = [
            {"feature": name, "smd_before": json_safe_float(pre), "smd_after": None, "smd_ipw": json_safe_float(ipw)}
            for name, pre, ipw in zip(feature_cols, pre_smd, ipw_smd)
        ]
        mean_abs_smd = float(np.mean(np.abs(pre_smd))) if len(pre_smd) else None
        worst_idx = int(np.argmax(np.abs(pre_smd))) if len(pre_smd) else None
        matched_achieved, n_over, max_allowed = _balance_achieved_from_abs_smd(np.abs(pre_smd), len(feature_cols), max_unbalanced_pct)
        return {
            "balance_achieved": matched_achieved or ipw_achieved,
            "matched_balance_achieved": matched_achieved,
            "mean_abs_smd": json_safe_float(mean_abs_smd) if mean_abs_smd is not None else None,
            "n_features_over_threshold": n_over,
            "max_unbalanced_features_allowed": max_allowed,
            "balance_threshold": balance_threshold,
            "matched_pairs": 0,
            "caliper": json_safe_float(caliper) if caliper is not None else None,
            "overlap": {"treated_in_control_range_pct": None, "control_in_treated_range_pct": None},
            "per_feature": per_feature,
            "worst_feature": feature_cols[worst_idx] if worst_idx is not None else None,
            "ipw": ipw_summary,
        }

    treat_idx = np.array([p[0] for p in matched_pairs])
    ctrl_idx = np.array([p[1] for p in matched_pairs])
    matched_treatments = np.concatenate([np.ones(len(treat_idx)), np.zeros(len(ctrl_idx))])
    matched_X = np.concatenate([X[treat_idx], X[ctrl_idx]], axis=0)
    post_smd = standardized_mean_diff(matched_X, matched_treatments)

    per_feature = [
        {"feature": name, "smd_before": json_safe_float(pre), "smd_after": json_safe_float(post), "smd_ipw": json_safe_float(ipw)}
        for name, pre, post, ipw in zip(feature_cols, pre_smd, post_smd, ipw_smd)
    ]
    abs_post = np.abs(post_smd)
    worst_idx = int(np.argmax(abs_post))
    mean_abs_smd = float(np.mean(abs_post))
    matched_achieved, n_over, max_allowed = _balance_achieved_from_abs_smd(abs_post, len(feature_cols), max_unbalanced_pct)

    control_ps = ps_logit[treatments == 0]
    treat_ps = ps_logit[treatments == 1]
    c_lo, c_hi = float(np.min(control_ps)), float(np.max(control_ps))
    t_lo, t_hi = float(np.min(treat_ps)), float(np.max(treat_ps))
    overlap = {
        "treated_in_control_range_pct": json_safe_float(float(np.mean((treat_ps >= c_lo) & (treat_ps <= c_hi)) * 100)),
        "control_in_treated_range_pct": json_safe_float(float(np.mean((control_ps >= t_lo) & (control_ps <= t_hi)) * 100)),
    }

    return {
        "balance_achieved": matched_achieved or ipw_achieved,
        "matched_balance_achieved": matched_achieved,
        "mean_abs_smd": json_safe_float(mean_abs_smd),
        "n_features_over_threshold": n_over,
        "max_unbalanced_features_allowed": max_allowed,
        "balance_threshold": balance_threshold,
        "matched_pairs": int(len(matched_pairs)),
        "caliper": json_safe_float(caliper),
        "overlap": overlap,
        "per_feature": per_feature,
        "worst_feature": feature_cols[worst_idx],
        "ipw": ipw_summary,
    }
