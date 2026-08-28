
"""
---
**Note:**
- Data paths and sensitive details are removed for sharing.

"""

#!/usr/bin/env python3

# ====================== IMPORTS ======================
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from sklearn.metrics import (confusion_matrix, classification_report, ConfusionMatrixDisplay)
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import hmac
import hashlib
import secrets


# ====================== CONFIGURATION ======================
RANDOM_STATE = 42
TEST_SIZE_REAL_SPLIT = 0.5          # For splitting real data into train/val and test
TEST_SIZE_VAL_SPLIT = 0.2           # For splitting simulated+real into train/val


# ====================== DIRECTORIES ======================

RES_DIR = os.path.join("/Users", "sol", "_courses", "project_work", "thesis", "results", "reRun2Cl", "22_1Hashing_FullLoc_31")
os.makedirs(RES_DIR, exist_ok=True)
DATA_REAL = os.path.join("/Users", "sol", "_courses", "project_work", "thesis", "data", "KMC", "real_260421", "realCombinedUnmerged31.csv.xz")
DATA_SIM = os.path.join("/Users", "sol", "_courses", "project_work", "thesis", "data", "KMC", "sim2", "sim2c200UnmergedDip", "sim2c100Hapc200UnmergedDip31.csv.xz")

print("Real dataset: ", DATA_REAL)
print("Simulatated dataset: ", DATA_SIM)
print("Result directory: ", RES_DIR)


# ====================== 1. DATA PREP ======================
# --- Load Real (Test) Data ---
dfR = pd.read_csv(DATA_REAL, compression="xz")
print("Shape of the real data:", dfR.shape)
print("Number of samples in the real data:", len(dfR["ID"].unique()))

assert dfR.isnull().sum().sum() == 0, "Missing values found in real data!"
assert dfR.duplicated().sum() == 0, "Duplicates found in real data!"

# --- Split Real Data into Train/Val and Test ---
unique_ids = dfR['ID'].unique()
sampled_ids, _ = train_test_split(
    unique_ids,
    test_size=TEST_SIZE_REAL_SPLIT,
    stratify=dfR.drop_duplicates('ID')['type'],
    random_state=RANDOM_STATE
)
train_df = dfR[dfR['ID'].isin(sampled_ids)]  # For mixing with simulated
test_df = dfR[~dfR['ID'].isin(sampled_ids)]  # For final test

# Pivot to wide format
test_df_wide = pd.pivot_table(
    test_df, index=["ID", "type"], columns=["kmer_seq"],
    values="count", fill_value=0
).reset_index()
print("Shape of the pivot test dataset:", test_df_wide.shape)

# Test data
X_test = test_df_wide.drop(['ID', 'type'], axis=1)
y_test = test_df_wide['type']

# --- Load Simulated (Training) Data ---
dfS = pd.read_csv(DATA_SIM, compression="xz")
print("Shape of the simulated data:", dfS.shape)
print("Number of samples in the simulated data:", len(dfS["ID"].unique()))

assert dfS.isnull().sum().sum() == 0, "Missing values found in simulated data!"
assert dfS.duplicated().sum() == 0, "Duplicates found in simulated data!"

# Add origin column
dfS['origin'] = 'simulated'
train_df['origin'] = 'real'

# Concatenate simulated + selected real
df1 = pd.concat([dfS, train_df], axis=0)
df_sim_wide = pd.pivot_table(
    df1, index=["ID", "type", "origin"], columns=["kmer_seq"],
    values="count", fill_value=0
).reset_index()
print(f"Shape of the simulated+real data after pivot: {df_sim_wide.shape}")

df_sim_wide = df_sim_wide[df_sim_wide['ID'] != 'NIST']
print(f"Shape of the pivot simulated+real data after removing 'NIST': {df_sim_wide.shape}")

# Split into features, labels, and origin
X = df_sim_wide.drop(['ID', 'type', 'origin'], axis=1)
y = df_sim_wide['type']
origin = df_sim_wide['origin']

# Split into train/val (stratified)
X_train_orig, X_val_orig, y_train_orig, y_val_orig, origin_train, origin_val = train_test_split(
    X, y, origin,
    test_size=TEST_SIZE_VAL_SPLIT,
    stratify=y,
    random_state=RANDOM_STATE
)


# ====================== 2. FEATURE ALIGNMENT ======================
print(f"Number of features in training dataset before alignment: {len(X_train_orig.columns)}")
print(f"Number of features in test dataset before alignment: {len(X_test.columns)}")

# Align features
features = X_train_orig.columns

##

# key_path = os.path.join(RES_DIR, "secret.key")

# if not os.path.exists(key_path):
#     with open(key_path, "wb") as f:
#         f.write(secrets.token_bytes(32))

# with open(key_path, "rb") as f:
#     SECRET_KEY = f.read()
    
##    
        
with open(os.path.join(RES_DIR, "secret.key"), "wb") as f:
    f.write(secrets.token_bytes(32))

with open(os.path.join(RES_DIR, "secret.key"), "rb") as f:
    SECRET_KEY = f.read()
        
def hash_kmer(kmer):
    return hmac.new(
        SECRET_KEY,
        kmer.encode("utf-8"),
        hashlib.sha256
    ).hexdigest()

feature_map = {
    k: hash_kmer(k)
    for k in features
}

X_train_orig = X_train_orig.rename(columns=feature_map)
X_val_orig = X_val_orig.rename(columns=feature_map)

test_map = {
    k: hash_kmer(k)
    for k in X_test.columns
}

X_test = X_test.rename(columns=test_map)

hashed_features = list(feature_map.values())

X_test_orig = X_test.reindex(columns=hashed_features, fill_value=0)

assert list(X_train_orig.columns) == list(X_test_orig.columns), "Train and test feature order/columns do not match!"

print("Raw Reduced Train data shape after alignment and without scaling:", X_train_orig.shape)
print("Raw Reduced Validation data shape after alignment and without scaling:", X_val_orig.shape)
print("Raw Reduced Real data shape after alignment and without scaling:", X_test_orig.shape)

print(f"Number of 'positive' and 'negative' samples in the raw training data: {y_train_orig.value_counts()}")
print(f"Number of 'positive' and 'negative' samples in the raw validation data: {y_val_orig.value_counts()}")
print(f"Number of 'positive' and 'negative' samples in the raw testing data: {y_test.value_counts()}")


# ====================== 12. HASHED DATA ======================

threshold = 0.5

full_pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("classifier", LogisticRegression(
        penalty="l2",
        solver="lbfgs",
        class_weight={"pos": 5, "neg": 1},
        C=5.0,
        max_iter=1000,
        random_state=RANDOM_STATE
    ))
])
full_pipeline.fit(X_train_orig, y_train_orig)

# Test prediction
y_test_prob = full_pipeline.predict_proba(X_test_orig)[:, 1]
y_test_pred = np.where(y_test_prob >= threshold, "pos", "neg")

prediction_results = pd.DataFrame({
    "sample_ID": X_test_orig.index,
    "true_label": y_test.values,
    "probability_pos": y_test_prob,
    "prediction": y_test_pred
})
prediction_results.to_csv(os.path.join(RES_DIR, "s12_test_sample_predictions.csv"), index=False)

print("\nThreshold used: ", threshold)
print("\nFull features classification report:\n", classification_report(y_test, y_test_pred))

report_dict_full = classification_report(y_test, y_test_pred, output_dict=True)
report_df_full = pd.DataFrame(report_dict_full).transpose()
report_df_full["Model"] = "Logistic_Regression_lbfgs"
cols = report_df_full.columns.tolist()
cols = cols[-1:] + cols[:-1]
report_df_full = report_df_full[cols]
report_df_full.loc['threshold'] = threshold
report_df_full.to_csv(os.path.join(RES_DIR, "s12_lr_classification_report_test.csv"), index=True)

cnf_matrix_full = confusion_matrix(y_test, y_test_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cnf_matrix_full)
disp.plot(cmap="Blues")
plt.title("Confusion Matrix (Full features)")
plt.savefig(os.path.join(RES_DIR, "s12_lr_full_feature_confMat_test.png"), dpi=300, bbox_inches="tight")
plt.show()
plt.close()


# len(set(hashed_features)) == len(hashed_features)
# Out[13]: True