#!/usr/bin/env python3

# ====================== IMPORTS ======================
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import (confusion_matrix, classification_report, ConfusionMatrixDisplay,
                             make_scorer, accuracy_score, precision_score, 
                             recall_score, f1_score, precision_recall_curve)
from sklearn.base import clone
from sklearn.preprocessing import (
    StandardScaler,
    QuantileTransformer,
    RobustScaler,
    PowerTransformer,
    FunctionTransformer
)
from sklearn.decomposition import PCA
from scipy.stats import t
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.signal import find_peaks


# ==========================================================
# PREPROCESSING PIPELINES
# ==========================================================
epsilon = 1e-10

def clr_transform(X):
    X = np.asarray(X)
    log_X = np.log(X + epsilon)
    geometric_mean = np.exp(np.mean(log_X, axis=1, keepdims=True))
    return log_X - np.log(geometric_mean)

def signed_log_transform(X):
    X = np.asarray(X)
    return np.sign(X) * np.log1p(np.abs(X))

def get_preprocessing_pipeline(method):
    if method == "zscore":
        return Pipeline([
            ("scaler", StandardScaler())
        ])
    elif method == "quantile":
        return Pipeline([
            ("transform", QuantileTransformer(
                output_distribution="normal",
                random_state=42))
        ])
    elif method == "log_robust":
        return Pipeline([
            ("log", FunctionTransformer(np.log1p, validate=False)),
            ("scaler", RobustScaler())
        ])
    elif method == "yeojohnson":
        return Pipeline([
            ("transform", PowerTransformer(
                method="yeo-johnson",
                standardize=True))
        ])
    elif method == "clr":
        return Pipeline([
            ("clr", FunctionTransformer(
                clr_transform,
                validate=False)),
            ("scaler", StandardScaler())
        ])
    elif method == "signedlog":
        return Pipeline([
            ("signedlog", FunctionTransformer(
                signed_log_transform,
                validate=False)),
            ("scaler", StandardScaler())
        ])
    else:
        raise ValueError(f"Unknown preprocessing: {method}")

# ----------------------------------------------------------
# CHOOSE THE PREPROCESSING METHOD HERE
# ----------------------------------------------------------
STANDARDISATION = "zscore"  # "zscore", "quantile", "log_robust", "yeojohnson", "clr", "signedlog"
PREPROCESSOR = get_preprocessing_pipeline(STANDARDISATION)
print(f"Using preprocessing: {STANDARDISATION}")


# ====================== DIRECTORIES ======================

RES_DIR = "../results"
os.makedirs(RES_DIR, exist_ok=True)

DATA_REAL = "../data_real.csv.xz"
DATA_SIM = "../data_sim.csv.xz"
DATA_SEL = "../data_sel.csv.xz"

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

# Pivot to wide format
df_real_wide = pd.pivot_table(dfR, index=["ID", "type"], columns=["kmer_seq"], values="count", fill_value=0).reset_index()
print(f"Shape of the real data after pivot: {df_real_wide.shape}")

# Test data
X_test = df_real_wide.drop(['ID', 'type'], axis=1)
y_test = df_real_wide['type']

# --- Load Simulated (Training) Data ---
dfS = pd.read_csv(DATA_SIM, compression="xz")
print("Shape of the simulated data:", dfS.shape)
print("Number of samples in the simulated data:", len(dfS["ID"].unique()))

assert dfS.isnull().sum().sum() == 0, "Missing values found in simulated data!"
assert dfS.duplicated().sum() == 0, "Duplicates found in simulated data!"

# Pivot to wide format
df_sim_wide = pd.pivot_table(dfS, index=["ID", "type"], columns=["kmer_seq"], values="count", fill_value=0).reset_index()
print(f"Shape of the simulated data after pivot: {df_sim_wide.shape}")

df_sim_wide = df_sim_wide[df_sim_wide['ID'] != 'NIST']
print(f"Shape of the pivot simulated data after removing 'NIST': {df_sim_wide.shape}")

# Split into features and labels
X = df_sim_wide.drop(['ID', 'type'], axis=1)
y = df_sim_wide['type']

# Split into train/val (stratified)
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)


# ====================== 1.5. FEATURE ALIGNMENT ======================
print(f"Number of features in training dataset before alignment: {len(X_train.columns)}")
print(f"Number of features in test dataset before alignment: {len(X_test.columns)}")

# Align features
common_kmer_columns = X_train.columns.intersection(X_test.columns)
print(f"Number of common k-mers: {len(common_kmer_columns)}")

unique_to_X_train = X_train.columns.difference(X_test.columns)
print(f"Number of features in X_train, not in X_test: {len(unique_to_X_train)}")

selected_features = pd.read_csv(DATA_SEL, header=None).squeeze().tolist()

X_train = X_train[selected_features]
#X_val = X_val[selected_features]
X_val = X_val.reindex(columns=selected_features, fill_value=0)
X_test = X_test.reindex(columns=selected_features, fill_value=0)

assert list(X_train.columns) == list(X_test.columns), "Train and test feature order/columns do not match!"

print("Raw Train data shape after alignment:", X_train.shape)
print("Raw Validation data shape after alignment:", X_val.shape)
print("Raw Real data shape after alignment:", X_test.shape)


# --------- DATA INSPECTION --------- #

# Sample a few features for visualization
sample_features = X_train.columns[:10]

# 1) Plot distributions
for feat in sample_features:
    sns.histplot(X_train[feat], kde=True, label='Train')
    sns.histplot(X_test[feat], kde=True, label='Test')    
    plt.title(f"{feat}_(before)")
    plt.legend()
    plt.savefig(os.path.join(RES_DIR, f"s1_{feat}_Dist_before.png"), dpi=300) 
    plt.show()  
    plt.close()
    
# 2) Boxplots (for outliers and scale)
for feat in sample_features:
    sns.boxplot(data=pd.concat([
        X_train[feat].rename('Train'),
        X_test[feat].rename('Test')
    ], axis=1))
    plt.title(feat)
    plt.show()
    
# 3) Overlaid KDE Plots (for distribution shape)
for feat in sample_features:
    sns.kdeplot(X_train[feat], label='Train')
    sns.kdeplot(X_test[feat], label='Test')
    plt.title(f"{feat}_(before)")
    plt.legend()
    plt.savefig(os.path.join(RES_DIR, f"s1_{feat}_before.png"), dpi=300)
    plt.show()  
    plt.close()
    
        
# --------- BIMODALITY DUE TO ZERO FILL --------- #    
    
# Calculate proportion of zeros for each feature
zero_proportions = (X_train[selected_features] == 0).mean()

# Plot distribution of zero proportions
plt.figure(figsize=(10, 6))
sns.histplot(zero_proportions, bins=30, kde=True)
plt.title("Distribution of Zero Proportions Across Features")
plt.xlabel("Proportion of Zeros")
plt.ylabel("Number of Features")
plt.savefig(os.path.join(RES_DIR, "s1_zero_proportions.png"), dpi=300)
plt.show()
plt.close()    
    

# Automatically Detect and Count Bimodal Features
zero_count_threshold = 100  # Adjust based on data (e.g., 100, 500, etc.)
bimodal_features = []

for feat in selected_features:
    # Count zeros
    zero_count = (X_train[feat] == 0).sum()

    # Only proceed if enough zeros
    if zero_count > zero_count_threshold:
        # Get KDE values
        kde = sns.kdeplot(X_train[feat], bw_adjust=0.5)
        x, y = kde.get_lines()[0].get_data()

        # Find peaks
        peaks, _ = find_peaks(y, height=0.01*y.max())

        # If more than one peak, likely bimodal
        if len(peaks) > 1:
            bimodal_features.append(feat)
            plt.close()  

print(f"Number of bimodal features (zero count > {zero_count_threshold}): {len(bimodal_features)}") 
    

# --------- FILTER OUT FEATURES RESPONSIBLE --------- #    

# Get non-bimodal features
non_bimodal_features = [f for f in selected_features if f not in bimodal_features]

# Filter all DataFrames
X_train_filtered = X_train[non_bimodal_features]
X_val_filtered = X_val[non_bimodal_features]
X_test_filtered = X_test[non_bimodal_features]

print(f"Original number of features: {len(selected_features)}") 
print(f"Number of features after filtering: {len(non_bimodal_features)}") 
    
pd.Series(non_bimodal_features).to_csv(os.path.join(RES_DIR, 's1_non_bimodal_features.csv'), index=False, header=False)


# Sample a few features for visualization
sample_features = X_train_filtered.columns[:10]

# 1) Plot distributions
for feat in sample_features:
    sns.histplot(X_train_filtered[feat], kde=True, label='Train')
    sns.histplot(X_test_filtered[feat], kde=True, label='Test')
    plt.title(f"{feat}_(after)")
    plt.legend()
    plt.savefig(os.path.join(RES_DIR, f"s1_nonBimodal_{feat}_Dist_after.png"), dpi=300)
    plt.show()
    plt.close()

# 3) Overlaid KDE Plots (for distribution shape)
for feat in sample_features:
    sns.kdeplot(X_train_filtered[feat], label='Train')
    sns.kdeplot(X_test_filtered[feat], label='Test')
    plt.title(f"{feat}_(after)")
    plt.legend()
    plt.savefig(os.path.join(RES_DIR, f"s1_nonBimodal_{feat}_after.png"), dpi=300)
    plt.show()
    plt.close()


print("Raw Train data shape after alignment and filteration:", X_train_filtered.shape)
print("Raw Validation data shape after alignment and filteration:", X_val_filtered.shape)
print("Raw Real data shape after alignment and filteration:", X_test_filtered.shape)

print(f"Number of 'positive' and 'negative' samples in the raw training data: {y_train.value_counts()}")
print(f"Number of 'positive' and 'negative' samples in the raw validation data: {y_val.value_counts()}")
print(f"Number of 'positive' and 'negative' samples in the raw testing data: {y_test.value_counts()}")


# ====================== 2. BASELINE MODEL EVALUATION (FILTERED FEATURES) ======================
# Define models
modelscv = {
    "Logistic Regression": Pipeline([
        ("preprocessing", clone(PREPROCESSOR)),
        ("classifier", LogisticRegression(solver="liblinear", class_weight="balanced", max_iter=1000, random_state=42))
    ]),
    "Decision Tree": Pipeline([
        ("preprocessing", clone(PREPROCESSOR)),
        ("classifier", DecisionTreeClassifier(random_state=42))
    ]),
    "Random Forest": Pipeline([
        ("preprocessing", clone(PREPROCESSOR)),
        ("classifier", RandomForestClassifier(random_state=42))
    ]),
    "LinearSVC": Pipeline([
        ("preprocessing", clone(PREPROCESSOR)),
        ("classifier", LinearSVC(max_iter=10000, random_state=42))
    ])
}

# Define scoring metrics
scoring = {
    'accuracy': make_scorer(accuracy_score),
    'precision': make_scorer(precision_score, average='macro'),
    'recall': make_scorer(recall_score, average='macro'),
    'f1': make_scorer(f1_score, average='macro')
}

# Define CV strategy
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

results = []
summary_results = []

for model_name, model in modelscv.items():
    print(f"Evaluating {model_name}...")
    cv_results = cross_validate(model, X_train_filtered, y_train, cv=skf, scoring=scoring)

    for metric in scoring:
        scores = cv_results[f"test_{metric}"]
        for score in scores:
            results.append({
                "Classifier": model_name,
                "Metric": metric.capitalize(),
                "Score": score,
                "Set": "CV"
            })

        mean = np.mean(scores)
        std = np.std(scores, ddof=1)
        n = len(scores)
        ci95 = t.ppf(0.975, df=n-1) * std / np.sqrt(n)

        summary_results.append({
            "Classifier": model_name,
            "Metric": metric.capitalize(),
            "Mean": mean,
            "Std": std,
            "CI95": ci95
        })

    # Validation evaluation
    model.fit(X_train_filtered, y_train)
    y_val_pred = model.predict(X_val_filtered)
    val_accuracy = accuracy_score(y_val, y_val_pred)
    val_precision = precision_score(y_val, y_val_pred, average="macro")
    val_recall = recall_score(y_val, y_val_pred, average="macro")
    val_f1 = f1_score(y_val, y_val_pred, average="macro")

    results.extend([
        {"Classifier": model_name, "Metric": "Accuracy", "Score": val_accuracy, "Set": "Validation"},
        {"Classifier": model_name, "Metric": "Precision", "Score": val_precision, "Set": "Validation"},
        {"Classifier": model_name, "Metric": "Recall", "Score": val_recall, "Set": "Validation"},
        {"Classifier": model_name, "Metric": "F1", "Score": val_f1, "Set": "Validation"}
    ])

# Convert to DataFrames and save as CSV
results_df = pd.DataFrame(results)
summary_df = pd.DataFrame(summary_results)

results_df.to_csv(os.path.join(RES_DIR, "s2_cv_validation_results.csv"), index=False)
summary_df.to_csv(os.path.join(RES_DIR, "s2_cv_summary_statistics.csv"), index=False)

# --- PLOTS ---
# 1. CV Results: Barplot (mean) + Stripplot (individual folds)
cv_results_df = results_df[results_df['Set'] == 'CV']
fig, ax = plt.subplots(figsize=(15, 8))
sns.set(style="whitegrid")

sns.barplot(data=cv_results_df, x="Classifier", y="Score", hue="Metric", ci=None, alpha=0.7, ax=ax)
sns.stripplot(data=cv_results_df, x="Classifier", y="Score", hue="Metric", dodge=True, jitter=True, ax=ax, linewidth=1, marker="o", edgecolor="gray", alpha=0.4)

handles, labels = ax.get_legend_handles_labels()
n_metrics = cv_results_df["Metric"].nunique()
ax.legend_.remove()
ax.legend(handles[:n_metrics], labels[:n_metrics], loc="best", bbox_to_anchor=(1.02, 0.8), title="Metric")

for bars in ax.containers:
    ax.bar_label(bars, fmt="%.2f", label_type="edge", fontsize=10, padding=3)

ax.set_title("Cross-Validation (k=5) Model Scores", fontsize=24)
ax.set_xlabel("Classifier", fontsize=18)
ax.set_ylabel("Score", fontsize=18)
ax.tick_params(labelsize=14, rotation=45)
sns.despine()

plt.tight_layout()
fig.savefig(os.path.join(RES_DIR, "s2_cv_classifier_barplot.png"), dpi=300)
plt.show()
plt.close(fig)

# 2. Validation Results: Barplot
val_results_df = results_df[results_df['Set'] == 'Validation']
fig, ax = plt.subplots(figsize=(10, 6))
sns.barplot(data=val_results_df, x="Classifier", y="Score", hue="Metric", ax=ax)
ax.set_title("Validation Set Scores", fontsize=20)
ax.set_xlabel("Classifier", fontsize=16)
ax.set_ylabel("Score", fontsize=16)
ax.tick_params(labelsize=12)
sns.despine()
plt.tight_layout()
fig.savefig(os.path.join(RES_DIR, "s2_validation_barplot.png"), dpi=300)
plt.show()
plt.close(fig)

# 3. Confusion Matrices
fig, axes = plt.subplots(2, 2, figsize=(12, 12))
axes = axes.ravel()

for i, (model_name, model) in enumerate(modelscv.items()):
    model.fit(X_train_filtered, y_train)
    y_val_pred = model.predict(X_val_filtered)
    ConfusionMatrixDisplay.from_predictions(y_val, y_val_pred, ax=axes[i], cmap='Blues')
    axes[i].set_title(f"{model_name} Confusion Matrix (Validation)", fontsize=12)

plt.tight_layout()
fig.savefig(os.path.join(RES_DIR, "s2_all_models_validation_confusion_matrix.png"), dpi=300)
plt.show()
plt.close(fig)


# ====================== 3. TEST SET EVALUATION (FILTERED FEATURES) ======================
# ---------------------- (i) L2 PIPELINE 1 ----------------------
l2_pipeline1 = Pipeline([
    ("preprocessing", clone(PREPROCESSOR)),
    ("classifier", LogisticRegression(solver="liblinear", class_weight="balanced", max_iter=1000, random_state=42))
])
l2_pipeline1.fit(X_train_filtered, y_train)
y_test_pred_l2p1 = l2_pipeline1.predict(X_test_filtered)
print("\nl2p1 classification report:\n", classification_report(y_test, y_test_pred_l2p1))
report_dict = classification_report(y_test, y_test_pred_l2p1, output_dict=True)
report_df = pd.DataFrame(report_dict).transpose()
report_df.to_csv(os.path.join(RES_DIR, "s3_l2p1_testSet_classification_report.csv"))

cm = confusion_matrix(y_test, y_test_pred_l2p1)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap="Blues")
plt.title("Confusion Matrix (Full Features)")
plt.savefig(os.path.join(RES_DIR, "s3_l2p1_testSet_confusion_matrix.png"), dpi=300, bbox_inches='tight')
plt.show()
plt.close()

# ---------------------- (ii) L2 PIPELINE 2 ----------------------
l2_pipeline2 = Pipeline([
    ("preprocessing", clone(PREPROCESSOR)),
    ("classifier", LogisticRegression(solver="liblinear", class_weight={"pos": 3, "neg": 1}, max_iter=1000, random_state=42))
])
l2_pipeline2.fit(X_train_filtered, y_train)

y_test_pred_l2p2 = l2_pipeline2.predict(X_test_filtered)
print("\nl2p2 classification report:\n", classification_report(y_test, y_test_pred_l2p2))
report_dict = classification_report(y_test, y_test_pred_l2p2, output_dict=True)
report_df = pd.DataFrame(report_dict).transpose()
report_df.to_csv(os.path.join(RES_DIR, "s3_l2p2_testSet_classification_report.csv"))

cm = confusion_matrix(y_test, y_test_pred_l2p2)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap="Blues")
plt.title("Confusion Matrix (Full Features)")
plt.savefig(os.path.join(RES_DIR, "s3_l2p2_testSet_confusion_matrix.png"), dpi=300, bbox_inches='tight')
plt.show()
plt.close()

# ---------------------- (iii) L2 PIPELINE 3 ----------------------
l2_pipeline3 = Pipeline([
    ("preprocessing", clone(PREPROCESSOR)),
    ("classifier", LogisticRegression(penalty="l2", solver="lbfgs", class_weight='balanced', C=0.8, max_iter=1000, random_state=42))
])
l2_pipeline3.fit(X_train_filtered, y_train)

y_val_scores = l2_pipeline3.predict_proba(X_val_filtered)[:, 1]
precision, recall, thresholds = precision_recall_curve(y_val, y_val_scores, pos_label='pos')
f1_scores = 2 * (precision * recall) / (precision + recall + 1e-9)
# f1_scores = (2 * precision[:-1] * recall[:-1] / (precision[:-1] + recall[:-1] + 1e-9))
best_threshold_l2p3 = thresholds[np.argmax(f1_scores)]
y_test_scores = l2_pipeline3.predict_proba(X_test_filtered)[:, 1]

y_pred_adjusted = (y_test_scores >= best_threshold_l2p3).astype(int)
y_pred_adjusted = ['neg' if pred == 0 else 'pos' for pred in y_pred_adjusted]

print("\nl2p3 best threshold: ", best_threshold_l2p3)
print("\nl2p3 classification report:\n", classification_report(y_test, y_pred_adjusted, target_names=['neg', 'pos']))
report_dict = classification_report(y_test, y_pred_adjusted, target_names=['neg', 'pos'], output_dict=True)
report_df = pd.DataFrame(report_dict).transpose()
report_df.loc['best_threshold'] = best_threshold_l2p3
report_df.to_csv(os.path.join(RES_DIR, "s3_l2p3_testSet_classification_report.csv"))

cm = confusion_matrix(y_test, y_pred_adjusted)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix")
plt.savefig(os.path.join(RES_DIR, "s3_l2p3_testSet_confusion_matrix.png"), dpi=300, bbox_inches='tight')
plt.show()
plt.close()

# ---------------------- (iv) L2 PIPELINE 4 ----------------------
l2_pipeline4 = Pipeline([
    ("preprocessing", clone(PREPROCESSOR)),
    ("classifier", LogisticRegression(penalty="l2", solver="lbfgs", class_weight={'pos': 3, 'neg': 1}, C=0.8, max_iter=1000, random_state=42))
])
l2_pipeline4.fit(X_train_filtered, y_train)

y_val_scores = l2_pipeline4.predict_proba(X_val_filtered)[:, 1]
precision, recall, thresholds = precision_recall_curve(y_val, y_val_scores, pos_label='pos')
f1_scores = 2 * (precision * recall) / (precision + recall + 1e-9)
best_threshold_l2p4 = thresholds[np.argmax(f1_scores)]
y_test_scores = l2_pipeline4.predict_proba(X_test_filtered)[:, 1]

y_pred_adjusted = (y_test_scores >= best_threshold_l2p4).astype(int)
y_pred_adjusted = ['neg' if pred == 0 else 'pos' for pred in y_pred_adjusted]

print("\nl2p4 best threshold: ", best_threshold_l2p4)
print("\nl2p4 classification report:\n", classification_report(y_test, y_pred_adjusted, target_names=['neg', 'pos']))
report_dict = classification_report(y_test, y_pred_adjusted, target_names=['neg', 'pos'], output_dict=True)
report_df = pd.DataFrame(report_dict).transpose()
report_df.loc['best_threshold'] = best_threshold_l2p4
report_df.to_csv(os.path.join(RES_DIR, "s3_l2p4_testSet_classification_report.csv"))

cm = confusion_matrix(y_test, y_pred_adjusted)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix")
plt.savefig(os.path.join(RES_DIR, "s3_l2p4_testSet_confusion_matrix.png"), dpi=300, bbox_inches='tight')
plt.show()
plt.close()

# ---------------------- (v) L2 PIPELINE 5 ----------------------
l2_pipeline5 = Pipeline([
    ("preprocessing", clone(PREPROCESSOR)),
    ("classifier", LogisticRegression(penalty="l2", solver="lbfgs", class_weight={'pos': 5, 'neg': 1}, C=0.8, max_iter=1000, random_state=42))
])
l2_pipeline5.fit(X_train_filtered, y_train)

y_val_scores = l2_pipeline5.predict_proba(X_val_filtered)[:, 1]
precision, recall, thresholds = precision_recall_curve(y_val, y_val_scores, pos_label='pos')
f1_scores = 2 * (precision * recall) / (precision + recall + 1e-9)
best_threshold_l2p5 = thresholds[np.argmax(f1_scores)]
y_test_scores = l2_pipeline5.predict_proba(X_test_filtered)[:, 1]

y_pred_adjusted = (y_test_scores >= best_threshold_l2p5).astype(int)
y_pred_adjusted = ['neg' if pred == 0 else 'pos' for pred in y_pred_adjusted]

print("\nl2p5 best threshold: ", best_threshold_l2p5)
print("\nl2p5 classification report:\n", classification_report(y_test, y_pred_adjusted, target_names=['neg', 'pos']))
report_dict = classification_report(y_test, y_pred_adjusted, target_names=['neg', 'pos'], output_dict=True)
report_df = pd.DataFrame(report_dict).transpose()
report_df.loc['best_threshold'] = best_threshold_l2p5
report_df.to_csv(os.path.join(RES_DIR, "s3_l2p5_testSet_classification_report.csv"))

cm = confusion_matrix(y_test, y_pred_adjusted)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix")
plt.savefig(os.path.join(RES_DIR, "s3_l2p5_testSet_confusion_matrix.png"), dpi=300, bbox_inches='tight')
plt.show()
plt.close()

# ---------------------- (vi) L1 PIPELINE 1 ----------------------
l1_pipeline1 = Pipeline([
    ("preprocessing", clone(PREPROCESSOR)),
    ("classifier", LogisticRegression(penalty='l1', solver='liblinear', class_weight='balanced', C=0.2, max_iter=1000, random_state=42))
])
l1_pipeline1.fit(X_train_filtered, y_train)

y_val_scores = l1_pipeline1.predict_proba(X_val_filtered)[:, 1]
precision, recall, thresholds = precision_recall_curve(y_val, y_val_scores, pos_label='pos')
f1_scores = 2 * (precision * recall) / (precision + recall + 1e-9)
best_threshold_l1p1 = thresholds[np.argmax(f1_scores)]
y_test_scores = l1_pipeline1.predict_proba(X_test_filtered)[:, 1]

y_pred_adjusted = (y_test_scores >= best_threshold_l1p1).astype(int)
y_pred_adjusted = ['neg' if pred == 0 else 'pos' for pred in y_pred_adjusted]

print("\nl1p1 best threshold: ", best_threshold_l1p1)
print("\nl1p1 classification report:\n", classification_report(y_test, y_pred_adjusted, target_names=['neg', 'pos']))
report_dict = classification_report(y_test, y_pred_adjusted, target_names=['neg', 'pos'], output_dict=True)
report_df = pd.DataFrame(report_dict).transpose()
report_df.loc['best_threshold'] = best_threshold_l1p1
report_df.to_csv(os.path.join(RES_DIR, "s3_l1p1_testSet_classification_report.csv"))

cm = confusion_matrix(y_test, y_pred_adjusted)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix")
plt.savefig(os.path.join(RES_DIR, "s3_l1p1_testSet_confusion_matrix.png"), dpi=300, bbox_inches='tight')
plt.show()
plt.close()

# ---------------------- (vii) L1 PIPELINE 2 ----------------------
l1_pipeline2 = Pipeline([
    ("preprocessing", clone(PREPROCESSOR)),
    ("classifier", LogisticRegression(penalty='l1', solver='liblinear', class_weight={'pos': 3, 'neg': 1}, C=0.2, max_iter=1000, random_state=42))
])
l1_pipeline2.fit(X_train_filtered, y_train)

y_val_scores = l1_pipeline2.predict_proba(X_val_filtered)[:, 1]
precision, recall, thresholds = precision_recall_curve(y_val, y_val_scores, pos_label='pos')
f1_scores = 2 * (precision * recall) / (precision + recall + 1e-9)
best_threshold_l1p2 = thresholds[np.argmax(f1_scores)]
y_test_scores = l1_pipeline2.predict_proba(X_test_filtered)[:, 1]

y_pred_adjusted = (y_test_scores >= best_threshold_l1p2).astype(int)
y_pred_adjusted = ['neg' if pred == 0 else 'pos' for pred in y_pred_adjusted]

print("\nl1p2 best threshold: ", best_threshold_l1p2)
print("\nl1p2 classification report:\n", classification_report(y_test, y_pred_adjusted, target_names=['neg', 'pos']))
report_dict = classification_report(y_test, y_pred_adjusted, target_names=['neg', 'pos'], output_dict=True)
report_df = pd.DataFrame(report_dict).transpose()
report_df.loc['best_threshold'] = best_threshold_l1p2
report_df.to_csv(os.path.join(RES_DIR, "s3_l1p2_testSet_classification_report.csv"))

cm = confusion_matrix(y_test, y_pred_adjusted)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix")
plt.savefig(os.path.join(RES_DIR, "s3_l1p2_testSet_confusion_matrix.png"), dpi=300, bbox_inches='tight')
plt.show()
plt.close()

# ---------------------- (viii) L1 PIPELINE 3 ----------------------
l1_pipeline3 = Pipeline([
    ("preprocessing", clone(PREPROCESSOR)),
    ("classifier", LogisticRegression(penalty='l1', solver='liblinear', class_weight={'pos': 5, 'neg': 1}, C=0.2, max_iter=1000, random_state=42))
])
l1_pipeline3.fit(X_train_filtered, y_train)

y_val_scores = l1_pipeline3.predict_proba(X_val_filtered)[:, 1]
precision, recall, thresholds = precision_recall_curve(y_val, y_val_scores, pos_label='pos')
f1_scores = 2 * (precision * recall) / (precision + recall + 1e-9)
best_threshold_l1p3 = thresholds[np.argmax(f1_scores)]
y_test_scores = l1_pipeline3.predict_proba(X_test_filtered)[:, 1]

y_pred_adjusted = (y_test_scores >= best_threshold_l1p3).astype(int)
y_pred_adjusted = ['neg' if pred == 0 else 'pos' for pred in y_pred_adjusted]

print("\nl1p3 best threshold: ", best_threshold_l1p3)
print("\nl1p3 classification report:\n", classification_report(y_test, y_pred_adjusted, target_names=['neg', 'pos']))
report_dict = classification_report(y_test, y_pred_adjusted, target_names=['neg', 'pos'], output_dict=True)
report_df = pd.DataFrame(report_dict).transpose()
report_df.loc['best_threshold'] = best_threshold_l1p3
report_df.to_csv(os.path.join(RES_DIR, "s3_l1p3_testSet_classification_report.csv"))

cm = confusion_matrix(y_test, y_pred_adjusted)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix")
plt.savefig(os.path.join(RES_DIR, "s3_l1p3_testSet_confusion_matrix.png"), dpi=300, bbox_inches='tight')
plt.show()
plt.close()

# ---------------------- (ix) L1 PIPELINE 4 ----------------------
l1_pipeline4 = Pipeline([
    ("preprocessing", clone(PREPROCESSOR)),
    ("classifier", LogisticRegression(penalty='l1', solver='liblinear', class_weight={'pos': 3, 'neg': 1}, C=1.0, max_iter=1000, random_state=42))
])
l1_pipeline4.fit(X_train_filtered, y_train)

y_val_scores = l1_pipeline4.predict_proba(X_val_filtered)[:, 1]
precision, recall, thresholds = precision_recall_curve(y_val, y_val_scores, pos_label='pos')
f1_scores = 2 * (precision * recall) / (precision + recall + 1e-9)
best_threshold_l1p4 = thresholds[np.argmax(f1_scores)]
y_test_scores = l1_pipeline4.predict_proba(X_test_filtered)[:, 1]

y_pred_adjusted = (y_test_scores >= best_threshold_l1p4).astype(int)
y_pred_adjusted = ['neg' if pred == 0 else 'pos' for pred in y_pred_adjusted]

print("\nl1p4 best threshold: ", best_threshold_l1p4)
print("\nl1p4 classification report:\n", classification_report(y_test, y_pred_adjusted, target_names=['neg', 'pos']))
report_dict = classification_report(y_test, y_pred_adjusted, target_names=['neg', 'pos'], output_dict=True)
report_df = pd.DataFrame(report_dict).transpose()
report_df.loc['best_threshold'] = best_threshold_l1p4
report_df.to_csv(os.path.join(RES_DIR, "s3_l1p4_testSet_classification_report.csv"))

cm = confusion_matrix(y_test, y_pred_adjusted)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix")
plt.savefig(os.path.join(RES_DIR, "s3_l1p4_testSet_confusion_matrix.png"), dpi=300, bbox_inches='tight')
plt.show()
plt.close()


# ====================== 4. Dimensionality Reduction (PCA) ======================
pca_pipeline = Pipeline([
    ("preprocessing", clone(PREPROCESSOR)),
    ("pca", PCA(n_components=2, random_state=42))
])
pca_pipeline.fit(X_train_filtered)
X_sim_train_pca = pca_pipeline.transform(X_train_filtered)
X_real_test_pca = pca_pipeline.transform(X_test_filtered)

plt.figure(figsize=(10, 6))
plt.scatter(X_sim_train_pca[y_train == "pos", 0], X_sim_train_pca[y_train == "pos", 1], c='blue', label='Simulated Train (Positive)', alpha=0.5)
plt.scatter(X_sim_train_pca[y_train == "neg", 0], X_sim_train_pca[y_train == "neg", 1], c='cyan', label='Simulated Train (Negative)', alpha=0.5)
plt.scatter(X_real_test_pca[y_test == "pos", 0], X_real_test_pca[y_test == "pos", 1], c='red', label='Real Test (Positive)', alpha=0.5)
plt.scatter(X_real_test_pca[y_test == "neg", 0], X_real_test_pca[y_test == "neg", 1], c='orange', label='Real Test (Negative)', alpha=0.5)

explained_var = pca_pipeline.named_steps["pca"].explained_variance_ratio_

plt.xlabel(f'Principal Component 1 ({explained_var[0]*100:.1f}%)')
plt.ylabel(f'Principal Component 2 ({explained_var[1]*100:.1f}%)')
plt.title('PCA: Simulated vs Real Data')
plt.legend()
#plt.grid()
plt.tight_layout()
plt.savefig(os.path.join(RES_DIR, "s4_simReal_projection_colored_var_2.png"), dpi=300)
plt.show()
plt.close()
