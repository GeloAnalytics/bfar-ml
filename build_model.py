import numpy as np
import pandas as pd
import json
import joblib
import os
from sklearn.ensemble import GradientBoostingClassifier

# Pre-Program Features
pre_features = [
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

df = pd.read_csv('../bfar.csv')
df['treatment'] = df['Y_BOAT-RE'].notna().astype(int)

X = df[pre_features]
y = df['treatment']

gb = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
gb.fit(X, y)

os.makedirs('models', exist_ok=True)
joblib.dump(gb, 'models/gradient_boosting_ps_model.pkl')

with open('models/pre_features.json', 'w') as f:
    json.dump(pre_features, f)

print("Model and features saved successfully.")
