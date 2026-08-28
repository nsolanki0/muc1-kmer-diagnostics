
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
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import (confusion_matrix, classification_report, ConfusionMatrixDisplay,
                             make_scorer, accuracy_score, precision_score, 
                             recall_score, f1_score)
from sklearn.base import clone
from sklearn.preprocessing import (
    StandardScaler,
    QuantileTransformer,
    RobustScaler,
    PowerTransformer,
    FunctionTransformer
)
from scipy.stats import t
import matplotlib.pyplot as plt
import seaborn as sns


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

RES_DIR = "/scratch/solankin/MUC1/results/reRun2/14_3sim2Dipc100Hapc200Unmerged_27posBin_RLKMC_RF_ZScoreC_31"
os.makedirs(RES_DIR, exist_ok=True)
DATA_REAL = "/scratch/solankin/MUC1/data/KMC/real_260421/realCombinedUnmerged31.csv.xz"
DATA_SIM = "/scratch/solankin/MUC1/data/KMC/sim2/sim2c200UnmergedDip/sim2c100Hapc200UnmergedDip_27PosBin_31.csv.xz"

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

selected_features = X_train.columns

X_train = X_train[selected_features]
X_val = X_val[selected_features]
X_test = X_test.reindex(columns=selected_features, fill_value=0)

assert list(X_train.columns) == list(X_test.columns), "Train and test feature order/columns do not match!"

print("Raw Train data shape after alignment:", X_train.shape)
print("Raw Validation data shape after alignment:", X_val.shape)
print("Raw Real data shape after alignment:", X_test.shape)

print(f"Number of 'positive' and 'negative' samples in the raw training data: {y_train.value_counts()}")
print(f"Number of 'positive' and 'negative' samples in the raw validation data: {y_val.value_counts()}")
print(f"Number of 'positive' and 'negative' samples in the raw testing data: {y_test.value_counts()}")


# ====================== 2. BASELINE MODEL EVALUATION (FULL FEATURES) ======================
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
    cv_results = cross_validate(model, X_train, y_train, cv=skf, scoring=scoring)

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
    model.fit(X_train, y_train)
    y_val_pred = model.predict(X_val)
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
    model.fit(X_train, y_train)
    y_val_pred = model.predict(X_val)
    ConfusionMatrixDisplay.from_predictions(y_val, y_val_pred, ax=axes[i], cmap='Blues')
    axes[i].set_title(f"{model_name} Confusion Matrix (Validation)", fontsize=12)

plt.tight_layout()
fig.savefig(os.path.join(RES_DIR, "s2_all_models_validation_confusion_matrix.png"), dpi=300)
plt.show()
plt.close(fig)


# ====================== 3. TEST SET EVALUATION (FULL FEATURES) ======================
# ---------------------- (i) RF PIPELINE 1 ----------------------
rf_pipeline = Pipeline([
    ("preprocessing", clone(PREPROCESSOR)),
    ("classifier", RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42))
])
rf_pipeline.fit(X_train, y_train)
y_test_pred_rf = rf_pipeline.predict(X_test)
print("\nrf classification report:\n", classification_report(y_test, y_test_pred_rf))
report_dict = classification_report(y_test, y_test_pred_rf, output_dict=True)
report_df = pd.DataFrame(report_dict).transpose()
report_df.to_csv(os.path.join(RES_DIR, "s3_rf_testSet_classification_report.csv"))

cm = confusion_matrix(y_test, y_test_pred_rf)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap="Blues")
plt.title("Confusion Matrix (Full Features)")
plt.savefig(os.path.join(RES_DIR, "s3_rf_testSet_confusion_matrix.png"), dpi=300, bbox_inches='tight')
plt.show()
plt.close()

