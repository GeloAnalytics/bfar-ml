"""Trains the frozen bfar.csv baseline artifacts served by app.py.

Mirrors the model-selection methodology from updated_psm.ipynb: 5-fold CV across
four candidate classifiers (Logistic Regression, Random Forest, Gradient Boosting,
Neural Network), picks the lowest-MSE model, fits it on the full dataset, and
splits its feature importances into the top-30 "core" features (informational --
shown in app.py's /health) and the remaining 27.

This produces app.py's fixed bfar.csv reference baseline. The separate dynamic
model (trained per-upload via POST /train, see psm_core.select_top_features /
train_psm_model) is unrelated to this script and is never touched by it.
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

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, 'models')


def _feature_importance_proxy(X, y):
    """Neural Network has no feature_importances_ -- rank with a throwaway
    Gradient Boosting fit instead, same proxy the notebooks use."""
    proxy = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
    proxy.fit(X, y)
    return proxy.feature_importances_


def main():
    df = pd.read_csv(os.path.join(BASE_DIR, 'bfar.csv'))
    if 'treatment' not in df.columns:
        df['treatment'] = df['Y_BOAT-RE'].notna().astype(int)

    missing = [f for f in ALL_FEATURES if f not in df.columns]
    if missing:
        raise ValueError(f"bfar.csv is missing expected baseline features: {missing}")

    X = core.impute_dataframe(df, ALL_FEATURES)[ALL_FEATURES]
    y = df['treatment']

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    candidates = {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42),
        'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42),
        'Neural Network': MLPClassifier(hidden_layer_sizes=(100, 50), activation='relu', solver='adam',
                                         max_iter=500, random_state=42, early_stopping=True, validation_fraction=0.1),
    }
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    print("=== Model selection on bfar.csv (5-fold CV) ===")
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
    print(f"\nBest model (lowest MSE): {best_name}")
    for metric, val in scored[best_name].items():
        print(f"  {metric}: {val:.6f}")

    best_model = candidates[best_name]
    if best_name == 'Neural Network':
        best_model.fit(X_scaled, y)
        importances = _feature_importance_proxy(X, y)
    else:
        best_model.fit(X, y)
        importances = (
            best_model.feature_importances_ if hasattr(best_model, 'feature_importances_')
            else np.abs(best_model.coef_[0])
        )

    ranked = pd.Series(importances, index=ALL_FEATURES).sort_values(ascending=False)
    core_features = ranked.index[:CORE_FEATURE_COUNT].tolist()
    remaining_features = ranked.index[CORE_FEATURE_COUNT:].tolist()

    print(f"\nTop {CORE_FEATURE_COUNT} core features:")
    print(core_features)
    print(f"\nRemaining {len(remaining_features)} features:")
    print(remaining_features)

    os.makedirs(MODELS_DIR, exist_ok=True)
    joblib.dump(best_model, os.path.join(MODELS_DIR, 'best_model.pkl'))
    joblib.dump(scaler, os.path.join(MODELS_DIR, 'scaler.pkl'))
    for fname, values in (
        ('all_features.json', ALL_FEATURES),
        ('core_features.json', core_features),
        ('remaining_features.json', remaining_features),
    ):
        with open(os.path.join(MODELS_DIR, fname), 'w') as f:
            json.dump(values, f, indent=2)

    print(f"\nSaved baseline artifacts to {MODELS_DIR}{os.sep} "
          f"(best_model.pkl, scaler.pkl, all_features.json, core_features.json, remaining_features.json)")


if __name__ == '__main__':
    main()
