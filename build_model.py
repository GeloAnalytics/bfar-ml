"""Trains the frozen bfar.csv baseline artifacts served by app.py.

Two frozen models come out of this script:

  1. The raw-feature baseline (best_model.pkl + scaler.pkl + all/core/
     remaining feature lists): 5-fold CV across four candidate classifiers
     (Logistic Regression, Random Forest, Gradient Boosting, Neural
     Network), lowest-MSE model fit on the full dataset -- the tier-1 path,
     used whenever an upload carries bfar's exact 57 columns.

  2. The index-space baseline (index_model.pkl + index_scaler.pkl +
     index_taxonomy.json + index_stats.json): the same 57 features
     compressed into 6 universal composite indices (transport assets,
     household durables, connectivity, utilities access, housing, social
     protection), same CV selection run on that 6-column space -- the
     tier-2 path, used for programs whose own column headers have been
     mapped onto bfar's canonical items. The taxonomy carries each item's
     bfar mean/std (mapped columns inherit them for z-scoring), its
     within-index weight (raw model feature importance), and the
     hand-curated matching keywords column_matcher.py relies on.

Run this whenever bfar.csv changes; app.py never retrains these artifacts
itself.
"""
import json
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, mean_absolute_error, mean_squared_error, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

import psm_core as core
from psm_indices import INDEX_NAMES, compute_indices

# Pre-program features (57) -- present before program enrollment, so safe to
# use as propensity-score predictors without leaking post-treatment info.
ALL_FEATURES = [
    'D1.1:A_BIKE', 'D1.1-A_QTY', 'D1.2:A_MOTORC', 'D1.2-A_QTY',
    'D1.3:A_TRICYCLE', 'D1.3-A_QTY', 'D1.4:A_CAR', 'D1.4-A_QTY',
    'D1.5:A_JEEP', 'D1.5-A_QTY', 'D1.6:A_TRUCK', 'D1.6-A_QTY',
    'D1.7:A_OTHERS', 'D1.7-A_QTY', 'D2.1:A_TV', 'D2.1-A_QTY',
    'D2.2:A_DVD', 'D2.2-A_QTY', 'D2.3:A_WASH-M', 'D2.3-A_QTY',
    'D2.4:A_AC', 'D2.4-A_QTY', 'D2.5:A_E-FAN', 'D2.5-A_QTY',
    'D2.6:A_FRIDGE', 'D2.6-A_QTY', 'D2.7:A_STOVE', 'D2.7-A_QTY',
    'D2.8:A_E-HEATER', 'D2.8-A_QTY', 'D2.9:A_FURNITURE', 'D2.9-A_QTY',
    'D2.10:A_OTHERS', 'D2.10-A_QTY', 'D3.1:A_CP', 'D3.1-A_QTY',
    'D3.2:A_LANDLINE', 'D3.2-A_QTY', 'D3.3:A_COMPUTER', 'D3.3-A_QTY',
    'D3.4:A_OTHERS', 'D3.4-A_QTY', 'E1:A_DRINK-H2O', 'E2:A_DOMESTIC-H2O',
    'E3:A_POWER-SUP', 'E4:A_COOK-FUEL', 'E5:A_NET-SUBS', 'F1:A_HOUSE-OWN',
    'F2:A_HOUSE-ACQ', 'F3:A_HOUSE-BUILT', 'F4:A_OTHER-RP', 'G1:A_SSS',
    'G2:A_GSIS', 'G3:A_PhilHealth', 'G4:A_PN-IN', 'G5:A_LIFE-IN', 'G6:A_HEALTH-IN'
]
CORE_FEATURE_COUNT = 30

# Matching keywords per flag item -- what column_matcher.py compares a new
# program's headers against. Quantity variants (D1.2-A_QTY etc.) inherit
# their flag sibling's keywords; the taxonomy's "quantity" flag plus the
# matcher's qty-token rule keeps "owns_motorcycle" and "num_motorcycles"
# from cross-matching. Curated, not generated: bfar's own codes are opaque,
# so these ARE the semantic bridge to other programs' headers.
KEYWORDS = {
    'D1.1:A_BIKE': ['bike', 'bicycle'],
    'D1.2:A_MOTORC': ['motorcycle', 'motorbike', 'motor'],
    'D1.3:A_TRICYCLE': ['tricycle', 'trike'],
    'D1.4:A_CAR': ['car', 'automobile', 'sedan'],
    'D1.5:A_JEEP': ['jeep', 'jeepney'],
    'D1.6:A_TRUCK': ['truck', 'lorry'],
    'D1.7:A_OTHERS': ['othervehicle', 'othertransport'],
    'D2.1:A_TV': ['tv', 'television'],
    'D2.2:A_DVD': ['dvd'],
    'D2.3:A_WASH-M': ['washingmachine', 'washer'],
    'D2.4:A_AC': ['ac', 'aircon', 'airconditioner', 'airconditioning'],
    'D2.5:A_E-FAN': ['fan', 'electricfan'],
    'D2.6:A_FRIDGE': ['fridge', 'refrigerator', 'ref'],
    'D2.7:A_STOVE': ['stove', 'cooker', 'gasrange'],
    'D2.8:A_E-HEATER': ['heater', 'waterheater'],
    'D2.9:A_FURNITURE': ['furniture'],
    'D2.10:A_OTHERS': ['otherappliance', 'otherdurable'],
    'D3.1:A_CP': ['cp', 'cellphone', 'mobilephone', 'smartphone', 'phone', 'mobile'],
    'D3.2:A_LANDLINE': ['landline', 'telephone'],
    'D3.3:A_COMPUTER': ['computer', 'laptop', 'desktop', 'pc'],
    'D3.4:A_OTHERS': ['othergadget', 'otherdevice'],
    'E1:A_DRINK-H2O': ['drinkingwater', 'drinkwater', 'potablewater'],
    'E2:A_DOMESTIC-H2O': ['domesticwater', 'watersupply', 'watersource'],
    'E3:A_POWER-SUP': ['electricity', 'power', 'powersupply', 'powersource'],
    'E4:A_COOK-FUEL': ['cookingfuel', 'cookfuel', 'fuel', 'lpg'],
    'E5:A_NET-SUBS': ['internet', 'wifi', 'broadband', 'netsubscription'],
    'F1:A_HOUSE-OWN': ['houseownership', 'homeownership', 'ownhouse', 'houseown', 'housingtenure'],
    'F2:A_HOUSE-ACQ': ['houseacquisition', 'houseacquired', 'howacquired'],
    'F3:A_HOUSE-BUILT': ['housebuilt', 'housematerial', 'houseconstruction', 'construction'],
    'F4:A_OTHER-RP': ['otherproperty', 'realproperty', 'otherrealproperty'],
    'G1:A_SSS': ['sss', 'socialsecurity'],
    'G2:A_GSIS': ['gsis'],
    'G3:A_PhilHealth': ['philhealth'],
    'G4:A_PN-IN': ['pension'],
    'G5:A_LIFE-IN': ['lifeinsurance'],
    'G6:A_HEALTH-IN': ['healthinsurance'],
}

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, 'models')


def _index_for(feature):
    if feature.startswith('D1'):
        return 'transport_assets'
    if feature.startswith('D2'):
        return 'household_durables'
    if feature.startswith('D3'):
        return 'connectivity'
    if feature.startswith('E'):
        return 'utilities_access'
    if feature.startswith('F'):
        return 'housing'
    return 'social_protection'


def _keywords_for(feature):
    if '-A_QTY' in feature:
        stem = feature.split('-A_QTY')[0]  # 'D1.2-A_QTY' -> 'D1.2'
        for flag, kws in KEYWORDS.items():
            if flag.startswith(stem + ':'):
                return kws
        raise KeyError(f"no flag sibling with keywords found for quantity item {feature}")
    return KEYWORDS[feature]


def _feature_importance_proxy(X, y):
    """Neural Network has no feature_importances_ -- rank with a throwaway
    Gradient Boosting fit instead, same proxy the notebooks use."""
    proxy = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
    proxy.fit(X, y)
    return proxy.feature_importances_


def _candidate_models():
    return {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42),
        'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42),
        'Neural Network': MLPClassifier(hidden_layer_sizes=(100, 50), activation='relu', solver='adam',
                                         max_iter=500, random_state=42, early_stopping=True, validation_fraction=0.1),
    }


def _run_model_selection(X, y, label):
    """5-fold CV across the candidate set, best by lowest MSE, then fit on
    the full data. Returns (best_name, fitted_model, fitted_scaler)."""
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    candidates = _candidate_models()
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    print(f"=== Model selection on {label} (5-fold CV) ===")
    scored = {}
    for name, model in candidates.items():
        X_cv = X_scaled if name == 'Neural Network' else X
        y_pred = cross_val_predict(model, X_cv, y, cv=cv, method='predict_proba')[:, 1]
        mse = mean_squared_error(y, y_pred)
        scored[name] = {
            'mse': mse,
            'rmse': np.sqrt(mse),
            'mae': mean_absolute_error(y, y_pred),
            'auc': roc_auc_score(y, y_pred),
            'brier': brier_score_loss(y, y_pred),
        }
        print(f"  {name}: MSE={mse:.6f} RMSE={scored[name]['rmse']:.6f} "
              f"MAE={scored[name]['mae']:.6f} AUC={scored[name]['auc']:.6f} Brier={scored[name]['brier']:.6f}")

    best_name = min(scored, key=lambda n: scored[n]['mse'])
    print(f"Best model for {label} (lowest MSE): {best_name}\n")

    best_model = candidates[best_name]
    best_model.fit(X_scaled if best_name == 'Neural Network' else X, y)
    return best_name, best_model, scaler


def main():
    df = pd.read_csv(os.path.join(BASE_DIR, 'bfar.csv'))
    if 'treatment' not in df.columns:
        df['treatment'] = df['Y_BOAT-RE'].notna().astype(int)

    missing = [f for f in ALL_FEATURES if f not in df.columns]
    if missing:
        raise ValueError(f"bfar.csv is missing expected baseline features: {missing}")

    X = core.impute_dataframe(df, ALL_FEATURES)[ALL_FEATURES]
    y = df['treatment']

    # ---- 1. Raw-feature baseline (tier 1) ----
    best_name, best_model, scaler = _run_model_selection(X, y, 'raw 57 features')

    if best_name == 'Neural Network':
        importances = _feature_importance_proxy(X, y)
    elif hasattr(best_model, 'feature_importances_'):
        importances = best_model.feature_importances_
    else:
        importances = np.abs(best_model.coef_[0])

    ranked = pd.Series(importances, index=ALL_FEATURES).sort_values(ascending=False)
    core_features = ranked.index[:CORE_FEATURE_COUNT].tolist()
    remaining_features = ranked.index[CORE_FEATURE_COUNT:].tolist()

    # ---- 2. Taxonomy: item stats + within-index weights + keywords ----
    importance_by_feature = dict(zip(ALL_FEATURES, importances))
    taxonomy = {'indices': {name: {'items': {}} for name in INDEX_NAMES}}
    for feat in ALL_FEATURES:
        taxonomy['indices'][_index_for(feat)]['items'][feat] = {
            'weight': float(importance_by_feature[feat]),
            'mean': float(X[feat].mean()),
            'std': float(X[feat].std()),
            'keywords': _keywords_for(feat),
            'quantity': '-A_QTY' in feat,
        }
    for index_def in taxonomy['indices'].values():
        total = sum(item['weight'] for item in index_def['items'].values())
        for item in index_def['items'].values():
            # Equal weights if the raw model put zero importance on a whole
            # section -- an all-zero index would silently drop its items.
            item['weight'] = item['weight'] / total if total > 0 else 1.0 / len(index_def['items'])

    # ---- 3. Index-space baseline (tier 2) ----
    identity_mapping = {f: f for f in ALL_FEATURES}
    index_df, imputed = compute_indices(df, identity_mapping, taxonomy)
    assert not imputed, f"bfar itself produced imputed indices: {imputed}"

    index_best_name, index_model, index_scaler = _run_model_selection(index_df, y, '6 composite indices')

    index_stats = {
        name: {
            'mean': float(index_df[name].mean()),
            'std': float(index_df[name].std()),
            'median': float(index_df[name].median()),
            'min': float(index_df[name].min()),
            'max': float(index_df[name].max()),
        }
        for name in INDEX_NAMES
    }

    # ---- 4. Save everything ----
    os.makedirs(MODELS_DIR, exist_ok=True)
    joblib.dump(best_model, os.path.join(MODELS_DIR, 'best_model.pkl'))
    joblib.dump(scaler, os.path.join(MODELS_DIR, 'scaler.pkl'))
    joblib.dump(index_model, os.path.join(MODELS_DIR, 'index_model.pkl'))
    joblib.dump(index_scaler, os.path.join(MODELS_DIR, 'index_scaler.pkl'))
    for fname, values in (
        ('all_features.json', ALL_FEATURES),
        ('core_features.json', core_features),
        ('remaining_features.json', remaining_features),
        ('index_taxonomy.json', taxonomy),
        ('index_stats.json', index_stats),
    ):
        with open(os.path.join(MODELS_DIR, fname), 'w') as f:
            json.dump(values, f, indent=2)

    print(f"Raw model: {best_name} | Index model: {index_best_name}")
    print(f"Top {CORE_FEATURE_COUNT} core features: {core_features}")
    print(f"\nSaved baseline artifacts to {MODELS_DIR}{os.sep} "
          f"(best_model.pkl, scaler.pkl, index_model.pkl, index_scaler.pkl, "
          f"all/core/remaining_features.json, index_taxonomy.json, index_stats.json)")


if __name__ == '__main__':
    main()
