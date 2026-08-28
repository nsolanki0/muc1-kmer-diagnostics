
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
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import (confusion_matrix, classification_report, ConfusionMatrixDisplay,
                             make_scorer, accuracy_score, precision_score, recall_score, f1_score)
from sklearn.preprocessing import StandardScaler
from scipy.stats import t
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns


# ====================== CONFIGURATION ======================
RANDOM_STATE = 42
TEST_SIZE_REAL_SPLIT = 0.5          # For splitting real data into train/val and test
TEST_SIZE_VAL_SPLIT = 0.2           # For splitting simulated+real into train/val
TEST_SIZE_REAL_VAL_SPLIT = 0.4      # For splitting real train/val into train/val for tuning
N_REPEATS = 30
VALIDATION_SIZE = 10
N_BOOTSTRAP = 100
FEATURE_COUNTS = [100, 500, 750, 1000, 2000, 4000, 6000, 9000, 12000]
CANDIDATE_FEATURES = [100, 500, 750, 1000, 2000, 4000]

# ====================== DIRECTORIES ======================

RES_DIR = os.path.join("/Users", "sol", "_courses", "project_work", "thesis", "results", "reRun2", "21_3_DomainAdaptation2_PIP_ZScoreC_31_260801")
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

#train_df = df[df['label'] == 1].copy()
#train_df['origin'] = 'real'

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
common_kmer_columns = X_train_orig.columns.intersection(X_test.columns)
print(f"Number of common k-mers: {len(common_kmer_columns)}")

unique_to_X_train = X_train_orig.columns.difference(X_test.columns)
print(f"Number of features in X_train, not in X_test: {len(unique_to_X_train)}")

USE_VARIANCE_FILTER = False
if USE_VARIANCE_FILTER:
    selected_features = X_train_orig.columns[X_train_orig.var(axis=0) > 0.001]
else:
    selected_features = X_train_orig.columns

X_train_orig = X_train_orig[selected_features]
X_val_orig = X_val_orig[selected_features]
X_test = X_test.reindex(columns=selected_features, fill_value=0)

assert list(X_train_orig.columns) == list(X_test.columns), "Train and test feature order/columns do not match!"

print("Raw Train data shape after alignment and without scaling:", X_train_orig.shape)
print("Raw Validation data shape after alignment and without scaling:", X_val_orig.shape)
print("Raw Real data shape after alignment and without scaling:", X_test.shape)

print(f"Number of 'positive' and 'negative' samples in the raw training data: {y_train_orig.value_counts()}")
print(f"Number of 'positive' and 'negative' samples in the raw validation data: {y_val_orig.value_counts()}")
print(f"Number of 'positive' and 'negative' samples in the raw testing data: {y_test.value_counts()}")


# ====================== 3. DATA PREP FOR EXPERIMENTS ======================
# ---------------------- (i) DATA FOR LOOCV, TUNING ----------------------
# Extract indices for real/simulated samples
real_train_indices = np.where(origin_train == 'real')[0]
real_val_indices = np.where(origin_val == 'real')[0]
sim_train_indices = np.where(origin_train == 'simulated')[0]
sim_val_indices = np.where(origin_val == 'simulated')[0]

# Extract subsets (use DataFrames)
X_real_train = X_train_orig.iloc[real_train_indices]
y_real_train = y_train_orig.iloc[real_train_indices]
X_real_val = X_val_orig.iloc[real_val_indices]
y_real_val = y_val_orig.iloc[real_val_indices]

X_sim_train = X_train_orig.iloc[sim_train_indices]
y_sim_train = y_train_orig.iloc[sim_train_indices]
X_sim_val = X_val_orig.iloc[sim_val_indices]
y_sim_val = y_val_orig.iloc[sim_val_indices]

# Concatenate all real-world samples
X_real_train_val = pd.concat([X_real_train, X_real_val], axis=0).copy()
y_real_train_val = pd.concat([y_real_train, y_real_val], axis=0)

assert len(X_real_train_val) == 42, f"Expected 42 real train/val samples, got {len(X_real_train_val)}"

# Split into 24 (train) + 16 (val)
X_real_train_new, X_real_val_new, y_real_train_new, y_real_val_new = train_test_split(
    X_real_train_val, y_real_train_val,
    test_size=TEST_SIZE_REAL_VAL_SPLIT,
    stratify=y_real_train_val,
    random_state=RANDOM_STATE
)

# ---------------------- (ii) SCALING FOR BATCH TUNING ----------------------
X_train_tun = pd.concat([X_sim_train, X_real_train_new], axis=0).copy()
y_train_tun = pd.concat([y_sim_train, y_real_train_new], axis=0)

# Final validation set: 16 real-world samples
X_val_tun = X_real_val_new
y_val_tun = y_real_val_new

# Final test set: untouched
X_test_tun = X_test
y_test_tun = y_test

print("Simulated+Real samples in training set:", X_train_orig.shape)
print("Simulated samples in training set:", X_sim_train.shape)
print("Real samples in training set:", X_real_train.shape)
print("Real samples in validation set:", X_real_val.shape)
print("Real samples in training+validation set:", X_real_train_val.shape)

print("Tuning train data shape:", X_train_tun.shape)
print("Tuning validation data shape:", X_val_tun.shape)
print("Tuning test data shape:", X_test_tun.shape)

print(f"Number of 'positive' and 'negative' samples in the tuning training data: {y_train_tun.value_counts()}")
print(f"Number of 'positive' and 'negative' samples in the tuning validation data: {y_val_tun.value_counts()}")
print(f"Number of 'positive' and 'negative' samples in the tuning test data: {y_test_tun.value_counts()}")

tun_pipeline = Pipeline([
    ("scaler", StandardScaler())
])
X_train_tun_scaled = tun_pipeline.fit_transform(X_train_tun)
X_val_tun_scaled = tun_pipeline.transform(X_val_tun)
X_test_tun_scaled = tun_pipeline.transform(X_test_tun)

# X_train_orig.shape
# Out[11]: (5284, 14368)

# X_sim_train.shape
# Out[10]: (5250, 14368)

# X_real_train.shape
# Out[6]: (34, 14368)

# X_real_val.shape
# Out[8]: (8, 14368)

# X_real_train_val.shape
# Out[9]: (42, 14368)


# X_train_tun.shape
# Out[12]: (5275, 14368)

# X_val_tun.shape
# Out[13]: (17, 14368)


# ====================== 4. LOOCV ======================
# Initialize lists to store LOOCV predictions
y_true_loocv = []
y_pred_loocv = []

# --- LOOCV LOOP ---
for i in range(len(X_real_train_val)):
    X_left_out = X_real_train_val.iloc[[i]]
    y_left_out = y_real_train_val.iloc[i]

    X_train_loocv = pd.concat([X_sim_train, X_real_train_val.drop(index=i)])
    y_train_loocv = pd.concat([y_sim_train, y_real_train_val.drop(index=i)])

    # Use Pipeline for scaling + model
    loocv_pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(
            penalty='l2',
            solver='lbfgs',
            class_weight={'pos': 5, 'neg': 1},
            C=0.8,
            max_iter=1000,
            random_state=RANDOM_STATE
        ))
    ])
    loocv_pipeline.fit(X_train_loocv, y_train_loocv)
    y_pred = loocv_pipeline.predict(X_left_out)[0]
    y_true_loocv.append(y_left_out)
    y_pred_loocv.append(y_pred)

# --- EVALUATE LOOCV RESULTS ---
print("\nLOOCV Classification Report:\n", classification_report(y_true_loocv, y_pred_loocv, target_names=['neg', 'pos']))

report_dict = classification_report(y_true_loocv, y_pred_loocv, target_names=['neg', 'pos'], output_dict=True)
report_df = pd.DataFrame(report_dict).transpose()
report_df.to_csv(os.path.join(RES_DIR, "s4_loocv_testSet_classification_report.csv"))

# --- Save LOOCV Confusion Matrix ---
#disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['neg', 'pos'])
cm = confusion_matrix(y_true_loocv, y_pred_loocv)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap=plt.cm.Blues)
plt.title("LOOCV Confusion Matrix")
plt.savefig(os.path.join(RES_DIR, "s4_loocv_confusion_matrix.png"), dpi=300, bbox_inches='tight')
plt.show()
plt.close()


# ====================== 5. EVALUATE LOOCV MODEL ON TEST SET ======================
# Train final model on: Simulated + all 42 real train/val samples
X_train_loocv_evl = pd.concat([X_sim_train, X_real_train_val]).copy()
y_train_loocv_evl = pd.concat([y_sim_train, y_real_train_val])

loocv_evl_pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("classifier", LogisticRegression(
        penalty='l2',
        solver='lbfgs',
        class_weight={'pos': 5, 'neg': 1},
        C=0.8,
        max_iter=1000,
        random_state=RANDOM_STATE
    ))
])
loocv_evl_pipeline.fit(X_train_loocv_evl, y_train_loocv_evl)
y_pred_test_loocv = loocv_evl_pipeline.predict(X_test)

# --- Evaluate on test set ---
print("\nLOOCV EVALUATION Classification Report:\n", classification_report(y_test, y_pred_test_loocv, target_names=['neg', 'pos']))

report_dict = classification_report(y_test, y_pred_test_loocv, target_names=['neg', 'pos'], output_dict=True)
report_df = pd.DataFrame(report_dict).transpose()
report_df.to_csv(os.path.join(RES_DIR, "s5_loocv_evl_classification_report.csv"))

# --- Save Confusion Matrix ---
#disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['neg', 'pos'])
cm = confusion_matrix(y_test, y_pred_test_loocv)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap=plt.cm.Blues)
plt.title("LOOCV Evaluation Confusion Matrix")
plt.savefig(os.path.join(RES_DIR, "s5_loocv_evl_confusion_matrix.png"), dpi=300, bbox_inches='tight')
plt.show()
plt.close()


# ====================== 6. MANUAL BATCH TUNING ======================
print("Unique labels in y_train_tun:", y_train_tun.unique())
print("Unique labels in y_val_tun:", y_val_tun.unique())

# Define parameter grid
C_values = [0.1, 0.3, 0.5, 0.8, 1.0, 5.0, 10.0]
class_weights = [None, {'pos': 1, 'neg': 1}, {'pos': 3, 'neg': 1}, {'pos': 5, 'neg': 1}, {'pos': 10, 'neg': 1}, {'pos': 20, 'neg': 1}]
thresholds = [0.2, 0.3, 0.4, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75]

# Initialize variables to track best parameters
best_recall = 0
best_params = {}
best_threshold = 0.2

# List to store all results
tuning_results = []

# Loop over all parameter combinations
for C in C_values:
    for class_weight in class_weights:
        for threshold in thresholds:
            tun_model_pipeline = Pipeline([
                ("scaler", StandardScaler()),
                ("classifier", LogisticRegression(
                    penalty='l2',
                    solver='lbfgs',
                    C=C,
                    class_weight=class_weight,
                    max_iter=1000,
                    random_state=RANDOM_STATE
                ))
            ])
            tun_model_pipeline.fit(X_train_tun, y_train_tun)

            # Predict on validation set
            y_val_proba = tun_model_pipeline.predict_proba(X_val_tun)[:, 1]
            y_val_pred = (y_val_proba >= threshold).astype(int)

            y_val_pred_labels = np.where(y_val_pred == 1, 'pos', 'neg')

            # Calculate metrics
            cm = confusion_matrix(y_val_tun, y_val_pred_labels, labels=['neg', 'pos'])
            TN, FP, FN, TP = cm[0, 0], cm[0, 1], cm[1, 0], cm[1, 1]
            precision = precision_score(y_val_tun, y_val_pred_labels, pos_label='pos')
            recall = recall_score(y_val_tun, y_val_pred_labels, pos_label='pos')
            f1 = f1_score(y_val_tun, y_val_pred_labels, pos_label='pos')

            # Append results to the list
            tuning_results.append({
                'C': C,
                'class_weight': str(class_weight),
                'threshold': threshold,
                'TN': TN, 'FP': FP, 'FN': FN, 'TP': TP, 'CM': cm,
                'precision': precision,
                'recall': recall,
                'f1': f1,
            })

            # Update best parameters if current recall is better
            if recall > best_recall:
                best_recall = recall
                best_params = {'C': C, 'class_weight': class_weight}
                best_threshold = threshold

# Print best results
print("Best parameters for recall:", best_params)
print("Best recall:", best_recall)
print("Best threshold:", best_threshold)

# Convert results to DataFrame and save as CSV
tuning_results_df = pd.DataFrame(tuning_results)
tuning_results_df.to_csv(
    os.path.join(RES_DIR, "s6_tuning_results.csv"),
    index=False
)
print(f"Saved tuning results to {os.path.join(RES_DIR, 's6_tuning_results.csv')}")


# ====================== 7. SAMPLE WEIGHT AND PROPORTIONS ======================
SPLIT_DIR = os.path.join(RES_DIR, "strategy_splits")
os.makedirs(SPLIT_DIR, exist_ok=True)

print("Development real dataset:")
print(X_real_train_val.shape)
print(y_real_train_val.value_counts())

# Generate splits once
splits = []
SEEDS = list(range(N_REPEATS))

for seed in SEEDS:
    train_idx, val_idx = train_test_split(
        X_real_train_val.index,
        test_size=VALIDATION_SIZE,
        stratify=y_real_train_val,
        random_state=seed
    )

    splits.append({
        "seed": seed,
        "train_idx": list(train_idx),
        "val_idx": list(val_idx)
    })

    # Save individual split information
    split_df = pd.DataFrame({
        "seed": seed,
        "set": ["train"] * len(train_idx) + ["validation"] * len(val_idx),
        "index": list(train_idx) + list(val_idx)
    })
    split_df.to_csv(os.path.join(SPLIT_DIR, f"split_seed_{seed}.csv"), index=False)

print(f"Generated {len(splits)} reproducible splits")

# Save combined split table
all_splits = []
for s in splits:
    temp = pd.DataFrame({
        "seed": [s["seed"]],
        "train_idx": [s["train_idx"]],
        "val_idx": [s["val_idx"]]
    })
    all_splits.append(temp)
all_splits = pd.concat(all_splits, ignore_index=True)
all_splits.to_csv(os.path.join(SPLIT_DIR, "all_splits.csv"), index=False)
print("Saved split information to:", SPLIT_DIR)

# ---------------------------------------------------------------------
# STRATEGY C: Fixed real sample number (24), Vary real sample weighting
# ---------------------------------------------------------------------
N_REAL_TRAIN = 24
REAL_SAMPLE_WEIGHTS = [1, 2, 5, 10, 20]
RESULT_DIR_C = os.path.join(RES_DIR, "Strategy_C_real_weight")
os.makedirs(RESULT_DIR_C, exist_ok=True)

strategy_C_results = []

for split in splits:
    seed = split["seed"]
    print("\nRunning Strategy C seed:", seed)

    # Retrieve fixed validation set
    X_real_pool = X_real_train_val.loc[split["train_idx"]]
    y_real_pool = y_real_train_val.loc[split["train_idx"]]
    X_real_val = X_real_train_val.loc[split["val_idx"]]
    y_real_val = y_real_train_val.loc[split["val_idx"]]

    # Select 24 real samples for training
    X_real_train, _, y_real_train, _ = train_test_split(
        X_real_pool, y_real_pool,
        train_size=N_REAL_TRAIN,
        stratify=y_real_pool,
        random_state=seed
    )

    # Combine simulated + real (n=24)
    X_train_strategy_C = pd.concat([X_sim_train, X_real_train], axis=0).copy()
    y_train_strategy_C = pd.concat([y_sim_train, y_real_train], axis=0)

    X_val_strategy_C = X_real_val.copy()

    # Explicitly track origin for sample weights
    origin_train_strategy_C = pd.Series(
        ['simulated'] * len(X_sim_train) + ['real'] * len(X_real_train),
        index=X_train_strategy_C.index
    )

    for real_weight in REAL_SAMPLE_WEIGHTS:
        sample_weights = np.ones(len(X_train_strategy_C))
        sample_weights[origin_train_strategy_C == 'real'] = real_weight

        strategy_C_pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression(
                penalty="l2",
                solver="lbfgs",
                C=5,
                class_weight={"pos": 5, "neg": 1},
                max_iter=1000,
                random_state=RANDOM_STATE
            ))
        ])
        strategy_C_pipeline.fit(X_train_strategy_C, y_train_strategy_C, classifier__sample_weight=sample_weights)
        #strategy_C_pipeline.fit(X_train_strategy_C, y_train_strategy_C, sample_weight=sample_weights)
        y_pred = strategy_C_pipeline.predict(X_val_strategy_C)

        cm = confusion_matrix(y_real_val, y_pred, labels=["neg", "pos"])
        TN, FP, FN, TP = cm.ravel()
        precision = precision_score(y_real_val, y_pred, pos_label="pos", zero_division=0)
        recall = recall_score(y_real_val, y_pred, pos_label="pos", zero_division=0)
        f1 = f1_score(y_real_val, y_pred, pos_label="pos", zero_division=0)

        strategy_C_results.append({
            "seed": seed,
            "real_samples": N_REAL_TRAIN,
            "real_weight": real_weight,
            "validation_size": len(y_real_val),
            "TN": TN, "FP": FP, "FN": FN, "TP": TP,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "training_real_IDs": list(X_real_train.index),
            "validation_IDs": list(X_real_val.index)
        })

# Convert to dataframe and save as CSV
strategy_C_df = pd.DataFrame(strategy_C_results)
strategy_C_df.to_csv(os.path.join(RESULT_DIR_C, "Strategy_C_all_results.csv"), index=False)
strategy_C_summary = strategy_C_df.groupby("real_weight")[["precision", "recall", "f1", "TN", "FP", "FN", "TP"]].agg(["mean", "std"])
strategy_C_summary.to_csv(os.path.join(RESULT_DIR_C, "Strategy_C_summary.csv"))
print("\nStrategy C summary:")
print(strategy_C_summary)

# ---------------------------------------------------
# STRATEGY D: Vary number of real samples in training
# ---------------------------------------------------
REAL_SAMPLE_SIZES = [10, 15, 20, 24, 30]
RESULT_DIR_D = os.path.join(RES_DIR, "Strategy_D_real_number")
os.makedirs(RESULT_DIR_D, exist_ok=True)

strategy_D_results = []

for split in splits:
    seed = split["seed"]
    print("\nRunning Strategy D seed:", seed)

    X_real_pool = X_real_train_val.loc[split["train_idx"]]
    y_real_pool = y_real_train_val.loc[split["train_idx"]]
    X_real_val = X_real_train_val.loc[split["val_idx"]]
    y_real_val = y_real_train_val.loc[split["val_idx"]]

    for n_real in REAL_SAMPLE_SIZES:
        X_real_train, _, y_real_train, _ = train_test_split(
            X_real_pool, y_real_pool,
            train_size=n_real,
            stratify=y_real_pool,
            random_state=seed
        )

        # Combine simulated + real (n=10-30)
        X_train_strategy_D = pd.concat([X_sim_train, X_real_train], axis=0).copy()
        y_train_strategy_D = pd.concat([y_sim_train, y_real_train], axis=0)

        X_val_strategy_D = X_real_val.copy()

        strategy_D_pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression(
                penalty="l2",
                solver="lbfgs",
                C=5,
                class_weight={"pos": 5, "neg": 1},
                max_iter=1000,
                random_state=RANDOM_STATE
            ))
        ])
        strategy_D_pipeline.fit(X_train_strategy_D, y_train_strategy_D)
        y_pred = strategy_D_pipeline.predict(X_val_strategy_D)

        cm = confusion_matrix(y_real_val, y_pred, labels=["neg", "pos"])
        TN, FP, FN, TP = cm.ravel()
        precision = precision_score(y_real_val, y_pred, pos_label="pos", zero_division=0)
        recall = recall_score(y_real_val, y_pred, pos_label="pos", zero_division=0)
        f1 = f1_score(y_real_val, y_pred, pos_label="pos", zero_division=0)

        strategy_D_results.append({
            "seed": seed,
            "n_real_training": n_real,
            "validation_size": len(y_real_val),
            "TN": TN, "FP": FP, "FN": FN, "TP": TP,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "training_real_IDs": list(X_real_train.index),
            "validation_IDs": list(X_real_val.index)
        })

strategy_D_df = pd.DataFrame(strategy_D_results)
strategy_D_df.to_csv(os.path.join(RESULT_DIR_D, "Strategy_D_all_results.csv"), index=False)
strategy_D_summary = strategy_D_df.groupby("n_real_training")[["precision", "recall", "f1", "TN", "FP", "FN", "TP"]].agg(["mean", "std"])
strategy_D_summary.to_csv(os.path.join(RESULT_DIR_D, "Strategy_D_summary.csv"))
print("\nStrategy D summary:")
print(strategy_D_summary)


# ====================== 8. TEST SET EVALUATION (FULL FEATURES) ======================
threshold = 0.5

# Using original training data (X_train_orig) for full-feature evaluation
# Because above findings suggest the original split to be optimal 

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

selected_features = X_train_orig.columns
print(f"Actual number of selected features: {len(selected_features)}")

full_classifier = full_pipeline.named_steps["classifier"]
coefficients = full_classifier.coef_[0]

feature_coef_df = pd.DataFrame({
    "Feature": selected_features,
    "Coefficient": coefficients
})
feature_coef_df["Abs_Coefficient"] = feature_coef_df["Coefficient"].abs()
feature_coef_df = feature_coef_df.sort_values("Abs_Coefficient", ascending=False)
feature_coef_df.to_csv(os.path.join(RES_DIR, "s8_full_features_coefficients.csv"), index=False)

# Test prediction
y_test_prob = full_pipeline.predict_proba(X_test)[:, 1]
y_test_pred = np.where(y_test_prob >= threshold, "pos", "neg")

prediction_results = pd.DataFrame({
    "sample_ID": X_test.index,
    "true_label": y_test.values,
    "probability_pos": y_test_prob,
    "prediction": y_test_pred
})
prediction_results.to_csv(os.path.join(RES_DIR, "s8_test_sample_predictions.csv"), index=False)

print("\nThreshold used: ", threshold)
print("\nFull features classification report:\n", classification_report(y_test, y_test_pred))

report_dict_full = classification_report(y_test, y_test_pred, output_dict=True)
report_df_full = pd.DataFrame(report_dict_full).transpose()
report_df_full["Model"] = "Logistic_Regression_lbfgs"
cols = report_df_full.columns.tolist()
cols = cols[-1:] + cols[:-1]
report_df_full = report_df_full[cols]
report_df_full.loc['threshold'] = threshold
report_df_full.to_csv(os.path.join(RES_DIR, "s8_lr_classification_report_test.csv"), index=True)

cnf_matrix_full = confusion_matrix(y_test, y_test_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cnf_matrix_full)
disp.plot(cmap="Blues")
plt.title("Confusion Matrix (Full features)")
plt.savefig(os.path.join(RES_DIR, "s8_lr_full_feature_confMat_test.png"), dpi=300, bbox_inches="tight")
plt.show()
plt.close()


# ====================== 9. FEATURE SELECTION ======================
# Validation data
X_val_feature_sel = X_real_val.copy()
y_val_feature_sel = y_real_val.copy()

# Define metrics
scoring = {
    'accuracy': make_scorer(accuracy_score),
    'precision': make_scorer(precision_score, average='macro'),
    'recall': make_scorer(recall_score, average='macro'),
    'f1': make_scorer(f1_score, average='macro')
}

results = []
summary_results = []
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

for n in FEATURE_COUNTS:
    print(f"\nEvaluating top {n} features...")
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("selector", SelectFromModel(
            LogisticRegression(
                penalty="l2",
                solver="lbfgs",
                class_weight={'pos': 5, 'neg': 1},
                C=5.0,
                max_iter=1000,
                random_state=RANDOM_STATE
            ),
            max_features=n
        )),
        ("classifier", LogisticRegression(
            penalty="l2",
            solver="lbfgs",
            class_weight={'pos': 5, 'neg': 1},
            C=5.0,
            max_iter=1000,
            random_state=RANDOM_STATE
        ))
    ])

    cv_results = cross_validate(pipeline, X_train_orig, y_train_orig, cv=skf, scoring=scoring)
    cv_stats = {}

    for metric in scoring:
        scores = cv_results[f"test_{metric}"]
        mean = np.mean(scores)
        std = np.std(scores, ddof=1)
        ci95 = t.ppf(0.975, len(scores)-1) * std / np.sqrt(len(scores))
        cv_stats[metric] = mean
        summary_results.append({
            "n_features": n,
            "Metric": metric,
            "Mean": mean,
            "Std": std,
            "CI95": ci95
        })

    pipeline.fit(X_train_orig, y_train_orig)
    y_val_pred = pipeline.predict(X_val_feature_sel)
    val_accuracy = accuracy_score(y_val_feature_sel, y_val_pred)
    val_precision = precision_score(y_val_feature_sel, y_val_pred, average="macro")
    val_recall = recall_score(y_val_feature_sel, y_val_pred, average="macro")
    val_f1 = f1_score(y_val_feature_sel, y_val_pred, average="macro")

    results.append({
        "n_features": n,
        "mean_cv_train_accuracy": cv_stats["accuracy"],
        "mean_cv_train_precision": cv_stats["precision"],
        "mean_cv_train_recall": cv_stats["recall"],
        "mean_cv_train_f1": cv_stats["f1"],
        "validation_accuracy": val_accuracy,
        "validation_precision": val_precision,
        "validation_recall": val_recall,
        "validation_f1": val_f1
    })
    print(f"{n:4d} features | CV Recall = {cv_stats['recall']:.3f} | Validation Recall = {val_recall:.3f}")

# Save results as CSV
results_df = pd.DataFrame(results)
results_df.to_csv(os.path.join(RES_DIR, "s9_lr_cv_feature_selection_val_macro.csv"), index=False)
summary_df = pd.DataFrame(summary_results)
summary_df.to_csv(os.path.join(RES_DIR, "s9_lr_cv_summary_feature_selection_val_macro.csv"), index=False)


# Plotting
plt.figure(figsize=(16, 12))

# Plot CV training metrics
plt.subplot(2, 1, 1)
plt.errorbar(
    [r['n_features'] for r in results],
    [r['mean_cv_train_accuracy'] for r in results],
    label='CV Train Accuracy', marker='o', capsize=5
    )
plt.errorbar(
    [r['n_features'] for r in results],
    [r['mean_cv_train_precision'] for r in results],
    label='CV Train Precision', marker='o', capsize=5
    )
plt.errorbar(
    [r['n_features'] for r in results],
    [r['mean_cv_train_recall'] for r in results],
    label='CV Train Recall', marker='o', capsize=5
    )
plt.errorbar(
    [r['n_features'] for r in results],
    [r['mean_cv_train_f1'] for r in results],
    label='CV Train F1', marker='o', capsize=5
    )
plt.xlabel('Number of Features', fontsize=12)
plt.ylabel('Score', fontsize=12)
plt.title('Cross-Validated Training Metrics vs. Number of Features', fontsize=14)
plt.legend(fontsize=10)
plt.grid(True)

# Plot validation metrics
plt.subplot(2, 1, 2)
plt.plot(
    [r['n_features'] for r in results],
    [r['validation_accuracy'] for r in results],
    label='Validation Accuracy', marker='o'
    )
plt.plot(
    [r['n_features'] for r in results],
    [r['validation_precision'] for r in results],
    label='Validation Precision', marker='o'
    )
plt.plot(
    [r['n_features'] for r in results],
    [r['validation_recall'] for r in results],
    label='Validation Recall', marker='o'
    )
plt.plot(
    [r['n_features'] for r in results],
    [r['validation_f1'] for r in results],
    label='Validation F1', marker='o'
    )
plt.xlabel('Number of Features', fontsize=12)
plt.ylabel('Score', fontsize=12)
plt.title('Validation Metrics vs. Number of Features', fontsize=14)
plt.legend(fontsize=10)
plt.grid(True)

# Show the plot
plt.tight_layout()
plt.savefig(os.path.join(RES_DIR, "s9_lr_cv_feature_selection_all_metrics.png"), dpi=300)
plt.show()
plt.close()


# Select based on validation recall
best_result = max(results, key=lambda x: x['validation_recall'])
best_n = best_result['n_features']
print(f"Best number of features (by recall): {best_n} (Validation recall: {best_result['validation_recall']:.3f})")


# ------------ FEATURE STABILITY ANALYSIS ------------
stability_summary = {}
for n_features in CANDIDATE_FEATURES:
    print(f"\nEvaluating stability for {n_features} features")
    feature_counter = Counter()
    for b in range(N_BOOTSTRAP):
        # Use .copy() for bootstrapped samples to avoid modifying X_train_orig
        idx = np.random.choice(len(X_train_orig), len(X_train_orig), replace=True)
        X_boot = X_train_orig.iloc[idx].copy()  # <-- Add .copy() here
        y_boot = y_train_orig.iloc[idx].copy()  # <-- Add .copy() here
        selector_pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("selector", SelectFromModel(
                LogisticRegression(
                    penalty="l2",
                    solver="lbfgs",
                    C=5.0,
                    class_weight={"pos": 5, "neg": 1},
                    max_iter=1000,
                    random_state=b
                ),
                max_features=n_features
            ))
        ])
        selector_pipeline.fit(X_boot, y_boot)
        selector = selector_pipeline.named_steps["selector"]
        selected = X_boot.columns[selector.get_support()]  # Use X_boot.columns
        for f in selected:
            feature_counter[f] += 1

    freq = pd.DataFrame({
        "Feature": feature_counter.keys(),
        "Frequency": feature_counter.values()
    })
    freq["SelectionFrequency"] = freq["Frequency"] / N_BOOTSTRAP
    freq = freq.sort_values("SelectionFrequency", ascending=False)
    stability_summary[n_features] = freq
    freq.to_csv(os.path.join(RES_DIR, f"FeatureStability_{n_features}.csv"), index=False)

# Compare stability
for n in CANDIDATE_FEATURES:
    df = stability_summary[n]
    print(n)
    print("Median stability:", df.SelectionFrequency.median())
    print("Selected >80%:", np.sum(df.SelectionFrequency >= 0.80))
    print("Selected >90%:", np.sum(df.SelectionFrequency >= 0.90))


# ====================== 10. TEST SET EVALUATION (BEST FEATURES) ======================
# Select best number
best_n=4000
threshold = 0.5
print(f"Number of top features used in final model: {best_n}")

final_pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("selector", SelectFromModel(
        LogisticRegression(
            penalty="l2",
            solver="lbfgs",
            class_weight={'pos': 5, 'neg': 1},
            C=5.0,
            max_iter=1000,
            random_state=RANDOM_STATE
        ),
        max_features=best_n
    )),
    ("classifier", LogisticRegression(
        penalty="l2",
        solver="lbfgs",
        class_weight={'pos': 5, 'neg': 1},
        C=5.0,
        max_iter=1000,
        random_state=RANDOM_STATE
    ))
])
final_pipeline.fit(X_train_orig, y_train_orig)

y_test_prob = final_pipeline.predict_proba(X_test)[:, 1]
y_test_pred = np.where(y_test_prob >= threshold, "pos", "neg")

print("\nThreshold used: ", threshold)
print("\nBest features classification report:\n", classification_report(y_test, y_test_pred))

report_dict_final = classification_report(y_test, y_test_pred, output_dict=True)
report_df_final = pd.DataFrame(report_dict_final).transpose()
report_df_final["Model"] = "Logistic_Regression_lbfgs"
cols = report_df_final.columns.tolist()
cols = cols[-1:] + cols[:-1]
report_df_final = report_df_final[cols]
report_df_final.loc['threshold'] = threshold
report_df_final.loc['best_n'] = best_n
report_df_final.to_csv(os.path.join(RES_DIR, f"s10_lr_feature_selection_{best_n}_classification_report_test.csv"), index=True)

cnf_matrix_final = confusion_matrix(y_test, y_test_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cnf_matrix_final)
disp.plot(cmap="Blues")
plt.title(f"Confusion Matrix ({best_n} features)")
plt.savefig(os.path.join(RES_DIR, f"s10_lr_feature_selection_{best_n}_confMat_test.png"), dpi=300, bbox_inches="tight")
plt.show()
plt.close()


# Extract selected features
selector = final_pipeline.named_steps["selector"]
selected_mask = selector.get_support()
selected_features = X_train_orig.columns[selected_mask]
print(f"Actual number of selected features: {len(selected_features)}")

final_classifier = final_pipeline.named_steps["classifier"]
coefficients = final_classifier.coef_[0]
feature_coef_df = pd.DataFrame({
    "Feature": selected_features,
    "Coefficient": coefficients
})
feature_coef_df["Abs_Coefficient"] = feature_coef_df["Coefficient"].abs()
feature_coef_df = feature_coef_df.sort_values("Abs_Coefficient", ascending=False)
feature_coef_df.to_csv(os.path.join(RES_DIR, f"s10_selected_features_coefficients_{best_n}.csv"), index=False)

# Extract predictions
prediction_results = pd.DataFrame({
    "sample_ID": X_test.index,
    "true_label": y_test.values,
    "probability_pos": y_test_prob,
    "prediction": y_test_pred
})
prediction_results.to_csv(os.path.join(RES_DIR, f"s10_test_sample_predictions_{best_n}.csv"), index=False)


# ====================== 11. FEATURE ANALYSIS (HEATMAP) ======================
# Extract selected features and model coefficients
selector = final_pipeline.named_steps["selector"]
final_classifier = final_pipeline.named_steps["classifier"]
selected_indices = selector.get_support(indices=True)
selected_features = X_train_orig.columns[selected_indices]
coefficients = final_classifier.coef_[0]

feature_coef_df = pd.DataFrame({
    "Feature": selected_features,
    "Coefficient": coefficients
})
feature_coef_df = feature_coef_df.sort_values(by="Coefficient", ascending=False)

# Select top positive and negative features
top_25_positive = feature_coef_df.head(25)
bottom_25_negative = feature_coef_df.tail(25)
selected_features_df = pd.concat([top_25_positive, bottom_25_negative])
selected_feature_names = selected_features_df["Feature"].tolist()

# Reuse the scaler from final_pipeline (Section 10)
scaler_from_pipeline = final_pipeline.named_steps["scaler"]
X_train_viz_scaled = scaler_from_pipeline.transform(X_train_orig)
X_test_viz_scaled = scaler_from_pipeline.transform(X_test)

X_test_viz_scaled_df = pd.DataFrame(
    X_test_viz_scaled,
    columns=X_test.columns,
    index=X_test.index
)
# Use .copy() here because columns are added to X_test_selected
X_test_selected = X_test_viz_scaled_df[selected_feature_names].copy()  # <-- Add .copy() here
X_test_selected["actual"] = y_test.map({"pos": 1, "neg": 0}).astype(int)
X_test_selected["predicted"] = y_test_pred.copy()
X_test_selected["predicted"] = X_test_selected["predicted"].map({"pos": 1, "neg": 0})
X_test_selected = X_test_selected.sort_values(by="predicted", ascending=False)

# Prepare heatmap matrix
heatmap_data = X_test_selected.transpose()
rows_order = selected_features_df["Feature"].tolist() + ["actual", "predicted"]
heatmap_data = heatmap_data.reindex(rows_order)

# Plot heatmap
plt.figure(figsize=(15, 12))
ax = sns.heatmap(
    heatmap_data.iloc[:-2],
    cmap="RdBu_r",
    xticklabels=False,
    yticklabels=True,
    center=0,
    cbar_kws={"label": "Standardized Feature Value"}
)
plt.xlabel("Samples ordered by prediction: positive → negative", fontsize=15)
plt.ylabel("Features ordered by logistic regression coefficient", fontsize=15)
plt.title("Heatmap of Top 25 Positive and Bottom 25 Negative Features", fontsize=20)

n_pos = len(top_25_positive)
ax.axhline(y=n_pos, color="black", linewidth=2)
pred_pos = sum(y_test_pred == "pos")
ax.axvline(x=pred_pos, color="black", linewidth=2, linestyle="--")
ax.plot([], [], color="black", linestyle="--", label="Prediction threshold")
ax.legend(loc="upper right")

plt.tight_layout()
plt.savefig(os.path.join(RES_DIR, "s11_lr_feature_heatmap_top25_bottom25.png"), dpi=300, bbox_inches="tight")
plt.show()
plt.close()

