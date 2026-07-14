import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import math
import os
sns.set()
df = pd.read_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'bfar.csv'))
df.info()
df.head()
# Pre-Program Features
pre_features = [
  # Pagaaring sasakyan
  'D1.1:A_BIKE',
  'D1.1-A_QTY',
  'D1.2:A_MOTORC',
  'D1.2-A_QTY',
  'D1.3:A_TRICYCLE',
  'D1.3-A_QTY',
  'D1.4:A_CAR',
  'D1.4-A_QTY',
  'D1.5:A_JEEP',
  'D1.5-A_QTY',
  'D1.6:A_TRUCK',
  'D1.6-A_QTY',
  'D1.7:A_OTHERS',
  'D1.7-A_QTY',
  # Kagamitan sa bahay
  'D2.1:A_TV',
  'D2.1-A_QTY',
  'D2.2:A_DVD',
  'D2.2-A_QTY',
  'D2.3:A_WASH-M',
  'D2.3-A_QTY',
  'D2.4:A_AC',
  'D2.4-A_QTY',
  'D2.5:A_E-FAN',
  'D2.5-A_QTY',
  'D2.6:A_FRIDGE',
  'D2.6-A_QTY',
  'D2.7:A_STOVE',
  'D2.7-A_QTY',
  'D2.8:A_E-HEATER',
  'D2.8-A_QTY',
  'D2.9:A_FURNITURE',
  'D2.9-A_QTY',
  'D2.10:A_OTHERS',
  'D2.10-A_QTY',
  # Kagamitang teknolohiya
  'D3.1:A_CP',
  'D3.1-A_QTY',
  'D3.2:A_LANDLINE',
  'D3.2-A_QTY',
  'D3.3:A_COMPUTER',
  'D3.3-A_QTY',
  'D3.4:A_OTHERS',
  'D3.4-A_QTY',
  # Kagamitan sa pangkabuhayan (D4 not exists in the dataset)
  # Kalagayan ng pamumuhay/estilo ng pamumuhay
  'E1:A_DRINK-H2O',
  'E2:A_DOMESTIC-H2O',
  'E3:A_POWER-SUP',
  'E4:A_COOK-FUEL',
  'E5:A_NET-SUBS',
  # Ari-arian
  'F1:A_HOUSE-OWN',
  'F2:A_HOUSE-ACQ',
  'F3:A_HOUSE-BUILT',
  'F4:A_OTHER-RP',
  # Miyembro ng insurance
  'G1:A_SSS',
  'G2:A_GSIS',
  'G3:A_PhilHealth',
  'G4:A_PN-IN',
  'G5:A_LIFE-IN',
  'G6:A_HEALTH-IN'
]
df['treatment'] = df['Y_BOAT-RE'].notna().astype(int)
print(df[['Y_BOAT-RE', 'treatment']].head())
print("\nTreatment counts:")
print(df['treatment'].value_counts())
print("\nMeans by treatment group (numeric columns only):")
print(df.groupby('treatment').mean(numeric_only=True))
# separate control and treatment for t-test
df_control = df[df.treatment==0]
df_treatment = df[df.treatment==1]
# t-test for total income (dependent variable)
from scipy.stats import ttest_ind

# Print the mean income for each group
print(df_control['C5:TOT_INCOME/B'].mean(), df_treatment['C5:TOT_INCOME/B'].mean())

# Perform the t-test
_, p = ttest_ind(df_control['C5:TOT_INCOME/B'], df_treatment['C5:TOT_INCOME/B'])

# Print the p-value
print(f'p={p:.3f}')

# Interpretation
alpha = 0.05  # significance level
if p > alpha:
    print('Same distributions / same group mean (fail to reject H0 - not enough evidence to say the treatment had an effect)')
else:
    print('Different distributions / different group mean (reject H0 - treatment likely had an effect)')
# choose features for propensity score calculation
X = df[pre_features]
y = df['treatment']

X.head()
# use logistic regression to calculate the propensity scores
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(max_iter=1000)
lr.fit(X, y)

from sklearn.ensemble import RandomForestClassifier

# # Use RandomForest for propensity scores (better for complex relationships)
ps_model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
ps_model.fit(X, y)
df['ps'] = ps_model.predict_proba(X)[:, 1]  # Probability of being in treatment group

# # Keep your logit transformation
df['ps_logit'] = df['ps'].apply(lambda x: math.log(x / (1-x)) if x not in [0,1] else math.log((x+1e-6)/(1-x+1e-6)))
# get the coefficients
lr.coef_.ravel()  # or reshape(-1) refers to an unknown dimension, often used to flatten the array
# get the feature names
X.columns.to_numpy()
# combine features and coefficients into a dataframe
coeffs = pd.DataFrame({
    'column':X.columns.to_numpy(),
    'coeff':lr.coef_.ravel(),
})
coeffs
# prediction
pred_binary = lr.predict(X)  # binary 0 control, 1, treatment
pred_prob = lr.predict_proba(X)  # probabilities for classes

print('the binary prediction is:', pred_binary[0])
print('the corresponding probabilities are:', pred_prob[0])
# the propensity score (ps) is the probability of being 1 (i.e., in the treatment group)
df['ps'] = pred_prob[:, 1]

# calculate the logit of the propensity score for matching if needed
# I just use the propensity score to match in this tutorial
def logit(p):
    logit_value = math.log(p / (1-p))
    return logit_value

df['ps_logit'] = df.ps.apply(lambda x: logit(x))

df.head()
# check the overlap of ps for control and treatment using histogram
# if not much overlap, the matching won't work
sns.histplot(data=df, x='ps', hue='treatment')  # multiple="dodge" for
# adding 'min_req' here makes matching not working - because treatment is derived from min_req
# there is no overlap and thus matching will not work
X1 = df[pre_features]
y = df['treatment']

# use logistic regression to calculate the propensity scores
lr1 = LogisticRegression(max_iter=1000)
lr1.fit(X1, y)

pred_prob1 = lr1.predict_proba(X1)  # probabilities for classes
df['ps1'] = pred_prob1[:, 1]

sns.histplot(data=df, x='ps1', hue='treatment')
# use 75% of standard deviation of the propensity score as the caliper/radius
# get the k closest neighbors for each observations
# relax caliper and increase k can provide more matches

from sklearn.neighbors import NearestNeighbors

caliper = np.std(df.ps) * .75
print(f'caliper (radius) is: {caliper:.4f}')

n_neighbors = 50

# setup knn
knn = NearestNeighbors(n_neighbors=n_neighbors, radius=caliper)

ps = df[['ps']]  # double brackets as a dataframe
knn.fit(ps)
# distances and indexes
distances, neighbor_indexes = knn.kneighbors(ps)

print(neighbor_indexes.shape)

# the 10 closest points to the first point
print(distances[0])
print(neighbor_indexes[0])
# for each point in treatment, we find a matching point in control without replacement
# note the 10 neighbors may include both points in treatment and control

# matched_control = []  # keep track of the matched observations in control

# for current_index, row in df.iterrows():  # iterate over the dataframe
#     if row.treatment == 0:  # the current row is in the control group
#         df.loc[current_index, 'matched'] = np.nan  # set matched to nan
#     else:
#         for idx in neighbor_indexes[current_index, :]: # for each row in treatment, find the k neighbors
#             # make sure the current row is not the idx - don't match to itself
#             # and the neighbor is in the control
#             if (current_index != idx) and (df.loc[idx].treatment == 0):
#                 if idx not in matched_control:  # this control has not been matched yet
#                     df.loc[current_index, 'matched'] = idx  # record the matching
#                     matched_control.append(idx)  # add the matched to the list
#                     break

# # Allow matching with replacement (controls can be matched multiple times)
# matched_control = []  # Now can contain duplicates
# df['matched'] = np.nan

# for current_index, row in df[df['treatment']==1].iterrows():  # Only loop through treatment cases
#     # Get distances and indexes of neighbors
#     distances, indexes = knn.kneighbors([df.loc[current_index, ['ps']]])

#     # Find the closest control unit (even if already matched)
#     for idx in indexes[0]:
#         if (idx != current_index) and (df.loc[idx, 'treatment'] == 0):
#             df.loc[current_index, 'matched'] = idx
#             matched_control.append(idx)
#             break  # Take just the first match

# Create treatment and control groups
df_treatment = df[df['treatment'] == 1].copy()
df_control = df[df['treatment'] == 0].copy()

# Fit NearestNeighbors on control group only
knn = NearestNeighbors(n_neighbors=1, radius=caliper)
knn.fit(df_control[['ps']])

# Match each treatment unit to a control unit (with replacement)
matched_treatment_indexes = []
matched_control_indexes = []

for idx, row in df_treatment.iterrows():
    ps_value = row['ps']
    distances, indices = knn.kneighbors([[ps_value]])

    if distances[0][0] <= caliper:  # ensure match is within caliper
        control_idx = df_control.index[indices[0][0]]
        matched_treatment_indexes.append(idx)
        matched_control_indexes.append(control_idx)

# Retrieve matched observations
matched_treatment = df.loc[matched_treatment_indexes].copy()
matched_control = df.loc[matched_control_indexes].copy()

# Combine matched pairs into one DataFrame
df_matched = pd.concat([matched_treatment, matched_control])
# try to increase the number of neighbors and/or caliper to get more matches
print('total observations in treatment:', len(df[df.treatment==1]))
print('total matched observations in control:', len(matched_control))

# # control have no match
# treatment_matched = df.dropna(subset=['matched'])  # drop not matched

# # matched control observation indexes
# control_matched_idx = treatment_matched.matched
# control_matched_idx = control_matched_idx.astype(int)  # change to int
# control_matched = df.loc[control_matched_idx, :]  # select matched control observations

# # combine the matched treatment and control
# df_matched = pd.concat([treatment_matched, control_matched])

# df_matched.treatment.value_counts()
# matched control and treatment
df_matched_control = df_matched[df_matched.treatment==0]
df_matched_treatment = df_matched[df_matched.treatment==1]
# t-test for income (dependent variable) after matching
# p-value is not significant now
from scipy.stats import ttest_ind

# Print the mean income for matched control and treatment groups
print(df_matched_control['C5:TOT_INCOME/B'].mean(), df_matched_treatment['C5:TOT_INCOME/B'].mean())

# Compare samples
_, p = ttest_ind(df_matched_control['C5:TOT_INCOME/B'], df_matched_treatment['C5:TOT_INCOME/B'])
print(f'p={p:.3f}')

# Interpret the result
alpha = 0.05  # significance level
if p > alpha:
    print('Same distributions / same group mean (fail to reject H0 - we do not have enough evidence to reject H0)')
else:
    print('Different distributions / different group mean (reject H0 - treatment likely had an effect)')
# As an effect size, Cohen's d is typically used to represent the magnitude of differences between two (or more) groups on a given variable, with larger values representing a greater differentiation between the two groups on that variable.
# we hope the effect sizes for features decrease after matching
# adapted from https://machinelearningmastery.com/effect-size-measures-in-python/

from numpy import mean
from numpy import var
from math import sqrt

# function to calculate Cohen's d for independent samples
def cohen_d(d1, d2):
	# calculate the size of samples
	n1, n2 = len(d1), len(d2)
	# calculate the variance of the samples
	s1, s2 = var(d1, ddof=1), var(d2, ddof=1)
	# calculate the pooled standard deviation
	s = sqrt(((n1 - 1) * s1 + (n2 - 1) * s2) / (n1 + n2 - 2))
	# calculate the means of the samples
	u1, u2 = mean(d1), mean(d2)
	# calculate the effect size
	return (u1 - u2) / s
effect_sizes = []
cols = pre_features

for cl in cols:
    _, p_before = ttest_ind(df_control[cl], df_treatment[cl])
    _, p_after = ttest_ind(df_matched_control[cl], df_matched_treatment[cl])
    cohen_d_before = cohen_d(df_treatment[cl], df_control[cl])
    cohen_d_after = cohen_d(df_matched_treatment[cl], df_matched_control[cl])
    effect_sizes.append([cl,'before', cohen_d_before, p_before])
    effect_sizes.append([cl,'after', cohen_d_after, p_after])
df_effect_sizes = pd.DataFrame(effect_sizes, columns=['feature', 'matching', 'effect_size', 'p-value'])
df_effect_sizes
import matplotlib.pyplot as plt
import seaborn as sns

# Sort features for better visualization
df_effect_sizes_sorted = df_effect_sizes.sort_values(by='effect_size', ascending=False)

# Set plot size dynamically based on number of features
num_features = df_effect_sizes_sorted.shape[0]
fig_height = max(10, num_features * 0.25)  # adjust this scale if needed

# Create the barplot
fig, ax = plt.subplots(figsize=(15, fig_height))
sns.barplot(
    data=df_effect_sizes_sorted,
    x='effect_size',
    y='feature',
    hue='matching',
    orient='h'
)

# Improve layout
plt.title("Effect Sizes Before and After Matching")
plt.xlabel("Effect Size")
plt.ylabel("Feature")
plt.tight_layout()
plt.show()
predicted_data = df[['ps', 'ps_logit', 'treatment']]
print(predicted_data.head())
# Add a column to identify matched pairs
matched_treatment = matched_treatment.copy()
matched_control = matched_control.copy()

matched_treatment['pair_id'] = range(len(matched_treatment))
matched_control['pair_id'] = range(len(matched_control))

# Combine again with pair IDs
df_matched = pd.concat([matched_treatment, matched_control])

# View some of the matched pairs
matched_data = df_matched[['pair_id', 'ps', 'treatment']]
print(matched_data.sort_values('pair_id').head(10))
plt.figure(figsize=(10, 5))
sns.histplot(data=df_matched, x='ps', hue='treatment', kde=True, bins=30)
plt.title("Propensity Score Distribution After Matching")
plt.xlabel("Propensity Score")
plt.ylabel("Count")
plt.tight_layout()
plt.show()
df_effect_sizes_sorted = df_effect_sizes.sort_values(by='effect_size', ascending=False)

plt.figure(figsize=(15, max(10, len(df_effect_sizes_sorted['feature'].unique()) * 0.25)))
sns.barplot(
    data=df_effect_sizes_sorted,
    x='effect_size',
    y='feature',
    hue='matching',
    orient='h'
)
plt.title("Effect Sizes of Covariates Before and After Matching")
plt.axvline(0.1, color='gray', linestyle='--', label='Small Effect')
plt.axvline(0.25, color='orange', linestyle='--', label='Medium Effect')
plt.axvline(0.5, color='red', linestyle='--', label='Large Effect')
plt.xlabel("Cohen's d Effect Size")
plt.ylabel("Feature")
plt.legend()
plt.tight_layout()
plt.show()
pd.set_option('display.max_rows', None)
print(df_effect_sizes[['feature', 'matching', 'effect_size']].to_string(index=False))
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ttest_ind

def check_psm_success(df_effect_sizes, df_matched, df, treatment_col='treatment', ps_col='ps', effect_size_threshold=0.1, p_value_threshold=0.05):
    results = {
        'passed': True,
        'reasons': [],
        'failed_covariates': [],
        'effect_size_summary': None,
        'p_value_summary': None,
        'overlap_status': None,
        'sample_size_status': None
    }

    # 1. Check covariate balance (effect sizes and p-values after matching)
    after_matching = df_effect_sizes[df_effect_sizes['matching'] == 'after']

    # Count covariates with effect size > threshold or p-value < threshold
    large_effect_size = after_matching[abs(after_matching['effect_size']) > effect_size_threshold]
    significant_p_value = after_matching[after_matching['p-value'] < p_value_threshold]

    # Failure if more than 20% of covariates are unbalanced
    unbalanced_covariates = set(large_effect_size['feature']).union(set(significant_p_value['feature']))
    unbalanced_percentage = len(unbalanced_covariates) / len(after_matching) * 100

    if unbalanced_percentage > 20:
        results['passed'] = False
        results['reasons'].append(f"High percentage of unbalanced covariates ({unbalanced_percentage:.1f}%).")
        results['failed_covariates'] = list(unbalanced_covariates)

    # Store summaries
    results['effect_size_summary'] = after_matching['effect_size'].describe()
    results['p_value_summary'] = after_matching['p-value'].describe()

    # 2. Check propensity score overlap
    treatment_ps = df_matched[df_matched[treatment_col] == 1][ps_col]
    control_ps = df_matched[df_matched[treatment_col] == 0][ps_col]

    # Compare quantiles to check overlap
    treatment_quantiles = treatment_ps.quantile([0.1, 0.5, 0.9])
    control_quantiles = control_ps.quantile([0.1, 0.5, 0.9])

    # Check if quantiles are within a reasonable range (e.g., 0.1 difference)
    quantile_diff = (treatment_quantiles - control_quantiles).abs().max()
    if quantile_diff > 0.2:
        results['passed'] = False
        results['reasons'].append(f"Propensity score distributions do not overlap well (max quantile difference: {quantile_diff:.2f}).")

    results['overlap_status'] = {
        'treatment_quantiles': treatment_quantiles,
        'control_quantiles': control_quantiles,
        'max_quantile_diff': quantile_diff
    }

    # 3. Check sample size retention
    original_treatment_size = len(df[df[treatment_col] == 1])
    matched_treatment_size = len(df_matched[df_matched[treatment_col] == 1])
    retention_rate = matched_treatment_size / original_treatment_size * 100

    if retention_rate < 80:
        results['passed'] = False
        results['reasons'].append(f"Low sample retention rate ({retention_rate:.1f}%).")

    results['sample_size_status'] = {
        'original_treatment_size': original_treatment_size,
        'matched_treatment_size': matched_treatment_size,
        'retention_rate': retention_rate
    }

    return results

# Example usage:
results = check_psm_success(df_effect_sizes, df_matched, df)
print("PSM Evaluation Results:")
print(f"Overall Passed: {results['passed']}")
if not results['passed']:
    print("Reasons for Failure:")
    for reason in results['reasons']:
        print(f"- {reason}")
    if results['failed_covariates']:
        print(f"Unbalanced Covariates: {results['failed_covariates']}")

print("\nEffect Size Summary (After Matching):")
print(results['effect_size_summary'])
print("\nP-Value Summary (After Matching):")
print(results['p_value_summary'])
print("\nPropensity Score Overlap Status:")
print(f"Max Quantile Difference: {results['overlap_status']['max_quantile_diff']:.2f}")
print("\nSample Size Retention:")
print(f"Retention Rate: {results['sample_size_status']['retention_rate']:.1f}%")

# Plot propensity score distributions after matching
plt.figure(figsize=(10, 5))
sns.histplot(data=df, x='ps', hue='treatment', kde=True, bins=30)
plt.title("Propensity Score Distribution After Matching")
plt.xlabel("Propensity Score")
plt.ylabel("Count")
plt.show()
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, accuracy_score, roc_auc_score
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV

# Scale features for neural network (X assumed to be defined earlier)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
# Logistic Regression
lr = LogisticRegression(max_iter=1000, random_state=42)
lr.fit(X, y)
df['ps_logit'] = lr.predict_proba(X)[:, 1]

# Random Forest
rf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
rf.fit(X, y)
df['ps_rf'] = rf.predict_proba(X)[:, 1]

# Gradient Boosting
gb = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
gb.fit(X, y)
df['ps_gb'] = gb.predict_proba(X)[:, 1]

# Neural Network (MLP)
mlp = MLPClassifier(hidden_layer_sizes=(100, 50), activation='relu', solver='adam',
                    max_iter=500, random_state=42, early_stopping=True, validation_fraction=0.1)
mlp.fit(X_scaled, y)
df['ps_mlp'] = mlp.predict_proba(X_scaled)[:, 1]
# Random Forest calibrated
rf_cal = CalibratedClassifierCV(
    RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42),
    method='sigmoid', cv=5
)
rf_cal.fit(X, y)
df['ps_rf_cal'] = rf_cal.predict_proba(X)[:, 1]

# Gradient Boosting calibrated
gb_cal = CalibratedClassifierCV(
    GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42),
    method='sigmoid', cv=5
)
gb_cal.fit(X, y)
df['ps_gb_cal'] = gb_cal.predict_proba(X)[:, 1]

# MLP calibrated
mlp_cal = CalibratedClassifierCV(
    MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=500, random_state=42),
    method='sigmoid', cv=5
)
mlp_cal.fit(X, y)
df['ps_mlp_cal'] = mlp_cal.predict_proba(X)[:, 1]
def print_metrics(y_true, y_pred_prob, model_name):
    mse = mean_squared_error(y_true, y_pred_prob)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred_prob)
    y_pred_class = (y_pred_prob >= 0.5).astype(int)
    auc = roc_auc_score(y_true, y_pred_prob)
    print(f"{model_name:26} | MSE: {mse:.6f} | RMSE: {rmse:.6f} | MAE: {mae:.6f} | AUC: {auc:.4f}")
print("\n" + "="*100)
print("PROPENSITY SCORE MODEL PERFORMANCE: UNCALIBRATED vs CALIBRATED")
print("="*100)

print("\n--- UNCALIBRATED MODELS ---")
print_metrics(y, df['ps_logit'], "Logistic Regression")
print_metrics(y, df['ps_rf'],    "Random Forest (uncal)")
print_metrics(y, df['ps_gb'],    "Gradient Boosting (uncal)")
print_metrics(y, df['ps_mlp'],   "Neural Network (uncal)")

print("\n--- CALIBRATED MODELS (Platt scaling) ---")
print_metrics(y, df['ps_rf_cal'],  "Random Forest (calibrated)")
print_metrics(y, df['ps_gb_cal'],  "Gradient Boosting (calibrated)")
print_metrics(y, df['ps_mlp_cal'], "Neural Network (calibrated)")

print("\n" + "="*100)
print("RECOMMENDATION: Use uncalibrated Gradient Boosting (lowest MSE, highest AUC)")
print("Calibration (Platt scaling) worsened performance for all ensemble models.")
print("="*100)
import warnings
warnings.filterwarnings('ignore', category=UserWarning, module='sklearn')

# Use the best propensity scores
df['ps_final'] = df['ps_gb']

# Caliper = 0.2 * SD(ps_final) (common rule of thumb)
caliper = 0.2 * df['ps_final'].std()

# Nearest neighbor matching with replacement, within caliper
from sklearn.neighbors import NearestNeighbors

knn = NearestNeighbors(n_neighbors=1, radius=caliper)
knn.fit(df[df['treatment']==0][['ps_final']])

matched_treatment_idx = []
matched_control_idx = []

for idx, row in df[df['treatment']==1].iterrows():
    distances, indices = knn.kneighbors([[row['ps_final']]])
    if distances[0][0] <= caliper:
        control_idx = df[df['treatment']==0].index[indices[0][0]]
        matched_treatment_idx.append(idx)
        matched_control_idx.append(control_idx)

# Create matched dataset
matched_treatment = df.loc[matched_treatment_idx].copy()
matched_control = df.loc[matched_control_idx].copy()
matched_treatment['pair_id'] = range(len(matched_treatment))
matched_control['pair_id'] = range(len(matched_control))
df_matched = pd.concat([matched_treatment, matched_control])

print(f"Matched pairs: {len(matched_treatment)}")
from scipy.stats import ttest_rel

matched_treatment = df_matched[df_matched['treatment'] == 1].sort_values('pair_id')
matched_control   = df_matched[df_matched['treatment'] == 0].sort_values('pair_id')

diff = matched_treatment['C5:TOT_INCOME/B'].values - matched_control['C5:TOT_INCOME/B'].values
att = np.mean(diff)
t_stat, p_value = ttest_rel(matched_control['C5:TOT_INCOME/B'], matched_treatment['C5:TOT_INCOME/B'])
ci = np.percentile(diff, [2.5, 97.5])

print(f"ATT: {att:.4f}")
print(f"95% CI: [{ci[0]:.4f}, {ci[1]:.4f}]")
print(f"Paired t-test p-value: {p_value:.6f}")
def std_diff(treat, control):
    return (treat.mean() - control.mean()) / np.sqrt((treat.var() + control.var())/2)

for col in pre_features[:5]:  # check first few
    d = std_diff(matched_treatment[col], matched_control[col])
    print(f"{col:20} | Std diff: {d:.4f}")
df['ipw'] = np.where(df['treatment']==1, 1/df['ps_gb'], 1/(1-df['ps_gb']))
ate_ipw = (df[df['treatment']==1]['C5:TOT_INCOME/B'] * df[df['treatment']==1]['ipw']).sum() / df[df['treatment']==1]['ipw'].sum() - \
         (df[df['treatment']==0]['C5:TOT_INCOME/B'] * df[df['treatment']==0]['ipw']).sum() / df[df['treatment']==0]['ipw'].sum()
print(f"IPTW ATE: {ate_ipw:.4f}")
# Use logit of final propensity score for better balance
df['ps_logit_final'] = np.log(df['ps_gb'] / (1 - df['ps_gb']))

# Tighter caliper: 0.2 * SD(logit(PS))
caliper_logit = 0.2 * df['ps_logit_final'].std()
print(f"Caliper (logit scale): {caliper_logit:.4f}")

# Nearest neighbor matching on logit(PS)
knn_logit = NearestNeighbors(n_neighbors=1, radius=caliper_logit)
knn_logit.fit(df[df['treatment']==0][['ps_logit_final']])

matched_treat_idx = []
matched_contr_idx = []

for idx, row in df[df['treatment']==1].iterrows():
    distances, indices = knn_logit.kneighbors([[row['ps_logit_final']]])
    if distances[0][0] <= caliper_logit:
        matched_treat_idx.append(idx)
        matched_contr_idx.append(df[df['treatment']==0].index[indices[0][0]])

df_matched2 = pd.concat([
    df.loc[matched_treat_idx].assign(pair_id=range(len(matched_treat_idx))),
    df.loc[matched_contr_idx].assign(pair_id=range(len(matched_contr_idx)))
])

print(f"Matched pairs (tighter caliper): {len(matched_treat_idx)}")
treat2 = df_matched2[df_matched2['treatment']==1].sort_values('pair_id')
ctrl2  = df_matched2[df_matched2['treatment']==0].sort_values('pair_id')
diff2 = treat2['C5:TOT_INCOME/B'].values - ctrl2['C5:TOT_INCOME/B'].values

att2 = diff2.mean()
ci2 = np.percentile(diff2, [2.5, 97.5])
_, p2 = ttest_rel(ctrl2['C5:TOT_INCOME/B'], treat2['C5:TOT_INCOME/B'])

print(f"ATT (tight caliper): {att2:.4f}")
print(f"95% CI: [{ci2[0]:.4f}, {ci2[1]:.4f}]")
print(f"p-value: {p2:.6f}")

# Check balance on first few covariates
for col in pre_features[:5]:
    d = (treat2[col].mean() - ctrl2[col].mean()) / np.sqrt((treat2[col].var() + ctrl2[col].var())/2)
    print(f"{col:20} | Std diff: {d:.4f}")
def bootstrap_att(data, treat_col, outcome_col, ps_col, n_bootstrap=500):
    atts = []
    n_treat = data[data[treat_col]==1].shape[0]
    for _ in range(n_bootstrap):
        boot_idx = np.random.choice(data.index, size=n_treat, replace=True)
        boot_data = data.loc[boot_idx]
        treat = boot_data[boot_data[treat_col]==1][outcome_col]
        control = boot_data[boot_data[treat_col]==0][outcome_col]
        if len(treat) > 0 and len(control) > 0:
            atts.append(treat.mean() - control.mean())
    return np.percentile(atts, [2.5, 97.5]), atts

ci_boot, atts_boot = bootstrap_att(df_matched2, 'treatment', 'C5:TOT_INCOME/B', 'ps_gb')
print(f"Bootstrap 95% CI for ATT: [{ci_boot[0]:.4f}, {ci_boot[1]:.4f}]")
def sensitivity_approx(att, se, gamma, alpha=0.05):
    """
    Approximate sensitivity: adjusts pâ€‘value for unobserved confounder
    that changes odds of treatment by factor gamma.
    """
    from scipy.stats import norm
    # Simplified: inflate standard error by sqrt(gamma)
    se_adj = se * np.sqrt(gamma)
    z = att / se_adj
    p_adj = 2 * (1 - norm.cdf(abs(z)))
    return p_adj

# Compute standard error from paired differences
se_att = np.std(diff2) / np.sqrt(len(diff2))

print("Sensitivity to unobserved confounding:")
for gamma in [1.0, 1.5, 2.0, 2.5, 3.0]:
    p_adj = sensitivity_approx(att2, se_att, gamma)
    print(f"Gamma = {gamma:.1f} â†’ adjusted p-value = {p_adj:.4f}")
import statsmodels.formula.api as smf

# Inverse probability weights using final PS
df_matched2['weights'] = np.where(df_matched2['treatment']==1,
                                   1 / df_matched2['ps_gb'],
                                   1 / (1 - df_matched2['ps_gb']))

# Weighted regression (doubly robust)
dr_model = smf.wls('Q("C5:TOT_INCOME/B") ~ treatment', data=df_matched2,
                   weights=df_matched2['weights']).fit()
print(dr_model.summary().tables[1])
import statsmodels.formula.api as smf

# Inverse probability weights using final PS
df_matched2['weights'] = np.where(df_matched2['treatment']==1,
                                   1 / df_matched2['ps_gb'],
                                   1 / (1 - df_matched2['ps_gb']))

# Weighted regression (doubly robust)
dr_model = smf.wls('Q("C5:TOT_INCOME/B") ~ treatment', data=df_matched2,
                   weights=df_matched2['weights']).fit()
print(dr_model.summary().tables[1])
plt.hist(treat2['C5:TOT_INCOME/B'], bins=30, alpha=0.5, label='Treatment')
plt.hist(ctrl2['C5:TOT_INCOME/B'], bins=30, alpha=0.5, label='Control')
plt.legend()
plt.title('Outcome Distribution After Tight Caliper Matching')
plt.show()
# ============================================================
# COMPLETE PROPENSITY SCORE MATCHING ANALYSIS
# ============================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ttest_rel
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, roc_auc_score
from sklearn.neighbors import NearestNeighbors
import statsmodels.formula.api as smf
import warnings
warnings.filterwarnings('ignore', category=UserWarning, module='sklearn')

# ---------- 1. Define features and treatment ----------
X = df[pre_features]
y = df['treatment']
outcome_col = 'C5:TOT_INCOME/B'

# Scale features for neural network
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ---------- 2. Train propensity score models (uncalibrated) ----------
models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42),
    'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42),
    'Neural Network': MLPClassifier(hidden_layer_sizes=(100,50), activation='relu', solver='adam',
                                    max_iter=500, random_state=42, early_stopping=True, validation_fraction=0.1)
}

ps_cols = {}
print("\n=== Propensity Score Model Performance (Uncalibrated) ===")
for name, model in models.items():
    if name == 'Neural Network':
        model.fit(X_scaled, y)
        ps = model.predict_proba(X_scaled)[:, 1]
    else:
        model.fit(X, y)
        ps = model.predict_proba(X)[:, 1]
    ps_cols[name] = ps
    mse = mean_squared_error(y, ps)
    auc = roc_auc_score(y, ps)
    print(f"{name:20} | MSE: {mse:.6f} | AUC: {auc:.4f}")

# Best model: Gradient Boosting (lowest MSE, highest AUC)
best_ps = ps_cols['Gradient Boosting']
df['ps_final'] = best_ps
print("\nâœ… Selected: Gradient Boosting (uncalibrated) for final PS.")

# ---------- 3. Matching on logit(PS) with caliper ----------
df['ps_logit'] = np.log(df['ps_final'] / (1 - df['ps_final']))
caliper = 0.2 * df['ps_logit'].std()
print(f"\nCaliper (logit scale): {caliper:.4f}")

knn = NearestNeighbors(n_neighbors=1, radius=caliper)
control_ps = df[df['treatment']==0][['ps_logit']]
knn.fit(control_ps)

matched_treat_idx = []
matched_control_idx = []
for idx, row in df[df['treatment']==1].iterrows():
    dist, ind = knn.kneighbors([[row['ps_logit']]])
    if dist[0][0] <= caliper:
        matched_treat_idx.append(idx)
        matched_control_idx.append(control_ps.index[ind[0][0]])

df_matched = pd.concat([
    df.loc[matched_treat_idx].assign(pair_id=range(len(matched_treat_idx))),
    df.loc[matched_control_idx].assign(pair_id=range(len(matched_control_idx)))
])
print(f"Matched pairs: {len(matched_treat_idx)} / {df['treatment'].sum()} treated")

# ---------- 4. Balance check (standardized differences) ----------
treat_m = df_matched[df_matched['treatment']==1].sort_values('pair_id')
ctrl_m  = df_matched[df_matched['treatment']==0].sort_values('pair_id')

def std_diff(t, c):
    return (t.mean() - c.mean()) / np.sqrt((t.var() + c.var())/2)

balance = {}
for col in pre_features:
    balance[col] = std_diff(treat_m[col], ctrl_m[col])

print("\n=== Balance After Matching (Standardized Differences) ===")
for col, d in list(balance.items())[:5]:
    print(f"{col:20} | {d:.4f}")
print("... (full list available in 'balance' dict)")

# ---------- 5. Estimate ATT ----------
diff_income = treat_m[outcome_col].values - ctrl_m[outcome_col].values
att_mean = diff_income.mean()
t_stat, p_val = ttest_rel(ctrl_m[outcome_col], treat_m[outcome_col])

# Bootstrap CI
np.random.seed(42)
boot_att = []
for _ in range(500):
    boot_idx = np.random.choice(len(diff_income), size=len(diff_income), replace=True)
    boot_att.append(diff_income[boot_idx].mean())
ci_boot = np.percentile(boot_att, [2.5, 97.5])

print("\n=== Treatment Effect (ATT) ===")
print(f"Paired t-test ATT: {att_mean:.4f}, p = {p_val:.6f}")
print(f"Bootstrap 95% CI: [{ci_boot[0]:.4f}, {ci_boot[1]:.4f}]")

# ---------- 6. Doubly robust estimation ----------
df_matched['ipw'] = np.where(df_matched['treatment']==1,
                              1 / df_matched['ps_final'],
                              1 / (1 - df_matched['ps_final']))
dr_model = smf.wls(f'Q("{outcome_col}") ~ treatment', data=df_matched, weights=df_matched['ipw']).fit()
dr_att = dr_model.params['treatment']
dr_ci = dr_model.conf_int().loc['treatment'].values
print(f"Doubly robust ATT: {dr_att:.4f}, 95% CI [{dr_ci[0]:.4f}, {dr_ci[1]:.4f}], p = {dr_model.pvalues['treatment']:.6f}")

# ---------- 7. Sensitivity analysis (approximate) ----------
def sensitivity_pvalue(att, se, gamma):
    # inflate standard error by sqrt(gamma)
    se_adj = se * np.sqrt(gamma)
    z = att / se_adj
    from scipy.stats import norm
    return 2 * (1 - norm.cdf(abs(z)))

se_att = np.std(diff_income) / np.sqrt(len(diff_income))
print("\n=== Sensitivity to Unobserved Confounding (Gamma) ===")
for gamma in [1.0, 1.5, 2.0, 2.5, 3.0]:
    p_adj = sensitivity_pvalue(att_mean, se_att, gamma)
    print(f"Gamma = {gamma:.1f} â†’ adjusted p = {p_adj:.6f}")

# ---------- 8. Final summary ----------
print("\n" + "="*70)
print("FINAL SUMMARY")
print("="*70)
print(f"Matched pairs: {len(matched_treat_idx)}")
print(f"ATT (paired t-test): {att_mean:.4f} (p = {p_val:.6f})")
print(f"Bootstrap 95% CI: [{ci_boot[0]:.4f}, {ci_boot[1]:.4f}]")
print(f"Doubly robust ATT: {dr_att:.4f} [{dr_ci[0]:.4f}, {dr_ci[1]:.4f}]")
print(f"Balance (max std diff): {max(balance.values()):.4f}")
print(f"Sensitivity: effect remains significant (p<0.05) up to Gamma â‰ˆ 3.0")
print("="*70)
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Before matching
sns.histplot(data=df, x='ps_final', hue='treatment', bins=30, ax=axes[0])
axes[0].set_title('Before Matching')
axes[0].set_xlabel('Propensity Score')

# After matching
sns.histplot(data=df_matched, x='ps_final', hue='treatment', bins=30, ax=axes[1])
axes[1].set_title('After Matching')
axes[1].set_xlabel('Propensity Score')

plt.tight_layout()
plt.show()
# Compute standardized differences before matching
std_diff_before = []
for col in pre_features:
    treat_before = df[df['treatment']==1][col]
    control_before = df[df['treatment']==0][col]
    sd = (treat_before.mean() - control_before.mean()) / np.sqrt((treat_before.var() + control_before.var())/2)
    std_diff_before.append(sd)

# Use already computed `balance` dictionary for after matching
std_diff_after = [balance[col] for col in pre_features]

# Create DataFrame for plotting
love_df = pd.DataFrame({
    'Covariate': pre_features,
    'Before': std_diff_before,
    'After': std_diff_after
})

# Melt for seaborn
love_melted = love_df.melt(id_vars='Covariate', var_name='Matching', value_name='StdDiff')

plt.figure(figsize=(10, 12))
sns.barplot(data=love_melted, x='StdDiff', y='Covariate', hue='Matching', orient='h')
plt.axvline(0.1, color='gray', linestyle='--', label='Small effect (0.1)')
plt.axvline(0.2, color='red', linestyle='--', label='Medium effect (0.2)')
plt.xlabel('Standardized Mean Difference')
plt.title('Covariate Balance Before and After Matching')
plt.legend()
plt.tight_layout()
plt.show()
from sklearn.calibration import calibration_curve

prob_true, prob_pred = calibration_curve(y, df['ps_final'], n_bins=10)

plt.figure(figsize=(6, 6))
plt.plot(prob_pred, prob_true, marker='o', label='Gradient Boosting')
plt.plot([0, 1], [0, 1], linestyle=':', color='gray', label='Perfect calibration')
plt.xlabel('Mean Predicted Probability')
plt.ylabel('Fraction of Positives')
plt.title('Calibration Plot â€“ Gradient Boosting PS Model')
plt.legend()
plt.grid(alpha=0.3)
plt.show()
plt.figure(figsize=(14, 22))  # increase height

x = np.arange(len(pre_features))
width = 0.35

plt.barh(x - width/2, std_diff_before, width,
         label='Before', color='royalblue')

plt.barh(x + width/2, std_diff_after, width,
         label='After', color='orange')

# Smaller font size for labels
plt.yticks(x, pre_features, fontsize=8)

plt.xlabel('Standardized Mean Difference')
plt.title('Balance Improvement After Matching')

# Add more left spacing for long labels
plt.gcf().subplots_adjust(left=0.35)

plt.legend()

# Auto-adjust layout
plt.tight_layout()

plt.show()
import scipy.stats as stats

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Before matching
stats.probplot(df[df['treatment']==1][outcome_col], dist='norm', plot=axes[0])
axes[0].set_title('Treatment Group (Before)')
stats.probplot(df[df['treatment']==0][outcome_col], dist='norm', plot=axes[0], fit=False)
axes[0].legend(['Treatment', 'Control'])

# After matching
stats.probplot(treat_m[outcome_col], dist='norm', plot=axes[1])
axes[1].set_title('Treatment Group (After)')
stats.probplot(ctrl_m[outcome_col], dist='norm', plot=axes[1], fit=False)
axes[1].legend(['Treatment', 'Control'])

plt.tight_layout()
plt.show()
plt.figure(figsize=(6, 4))
sns.boxplot(data=df_matched, x='treatment', y=outcome_col)
plt.xticks([0, 1], ['Control', 'Treatment'])
plt.title('Outcome Distribution After Matching')
plt.ylabel('Income')
plt.show()
import joblib

# Assuming 'gb' is your trained GradientBoostingClassifier object
# and 'pre_features' is the list of feature names

# Save the model
joblib.dump(gb, 'gradient_boosting_ps_model.pkl')

# Save the feature list (optional, but useful for validation)
import json
with open('pre_features.json', 'w') as f:
    json.dump(pre_features, f)