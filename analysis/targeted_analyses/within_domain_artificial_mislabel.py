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
import matplotlib.pyplot as plt
import seaborn as sns
import joblib


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
DATA = "../data.csv.xz"

print("Dataset: ", DATA)
print("Result directory: ", RES_DIR)


# ====================== 1. DATA PREP ======================
df = pd.read_csv(DATA, compression="xz")
print("Shape of the dataset:", df.shape)
print("Number of samples in the dataset:", len(df["ID"].unique()))

assert df.isnull().sum().sum() == 0, "Missing values found in the dataset!"
assert df.duplicated().sum() == 0, "Duplicates found in the dataset!"

# Pivot to wide format
df_wide = pd.pivot_table(df, index=["ID", "type"], columns=["kmer_seq"], values="count", fill_value=0).reset_index()
print(f"Shape of the dataset after pivot: {df_wide.shape}")

df_wide = df_wide[df_wide['ID'] != 'NIST']
print(f"Shape of the pivot dataset after removing 'NIST': {df_wide.shape}")

# Split into features and labels
X = df_wide.drop(['ID', 'type'], axis=1)
y = df_wide['type']

# Split into train+val and test (stratified)
X_trainval, X_test, y_trainval, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# Split train+val into train and validation (stratified)
X_train, X_val, y_train, y_val = train_test_split(
    X_trainval, y_trainval, test_size=0.25, stratify=y_trainval, random_state=42
)

print("Raw Training data shape:", X_train.shape)
print("Raw Validation data shape:", X_val.shape)
print("Raw Testing data shape:", X_test.shape)

print(f"Number of 'positive' and 'negative' samples in the raw training data before mislabeling: {y_train.value_counts()}")
print(f"Number of 'positive' and 'negative' samples in the raw validation data before mislabeling: {y_val.value_counts()}")
print(f"Number of 'positive' and 'negative' samples in the raw testing data before mislabeling: {y_test.value_counts()}")


#------------------MISLABELING-------------------------------------------

"""
Mislabel 50% of the positive samples to negative so that the final 
proportion of positive and negative samples are 25% and 75% respectively.

However, the mislabeling is to be applied only to the Training and the 
Validation set, leaving Test set unchanged. 

"""

# Function to relabel a fraction of "pos" to "neg"
def relabel_pos_to_neg(y_series, fraction=0.5):
    pos_mask = (y_series == "pos")
    pos_indices = np.where(pos_mask)[0]
    np.random.shuffle(pos_indices)
    n_to_relabel = int(len(pos_indices) * fraction)
    relabel_indices = pos_indices[:n_to_relabel]
    y_series.iloc[relabel_indices] = "neg"
    return y_series

# Relabel half of "pos" in train and validation sets
y_train = relabel_pos_to_neg(y_train, fraction=0.5)
y_val = relabel_pos_to_neg(y_val, fraction=0.5)

print(f"Number of 'positive' and 'negative' samples in the raw training data after mislabeling: {y_train.value_counts()}")
print(f"Number of 'positive' and 'negative' samples in the raw validation data after mislabeling: {y_val.value_counts()}")
print(f"Number of 'positive' and 'negative' samples in the raw testing data after mislabeling: {y_test.value_counts()}")


# ====================== 2. BASELINE MODEL EVALUATION (FULL FEATURES) ======================
# Define models as pipelines
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

fitted_models = {}

# Collect results
results = []

for model_name, model in modelscv.items():
    print(f"Evaluating {model_name}...")

    # Cross-validation on training data only
    cv_results = cross_validate(model, X_train, y_train, cv=skf, scoring=scoring)

    # Store CV results
    for metric in scoring.keys():
        for score in cv_results[f'test_{metric}']:
            results.append({
                'Classifier': model_name,
                'Metric': metric.capitalize(),
                'Score': score,
                'Set': 'CV'
            })

    # Fit on training data
    model.fit(X_train, y_train)

    # Store fitted model for later reuse
    fitted_models[model_name] = model
    print(f"{model_name} fitted and stored.")

    # Evaluate on validation set
    y_val_pred = model.predict(X_val)
    val_accuracy = accuracy_score(y_val, y_val_pred)
    val_precision = precision_score(y_val, y_val_pred, average='macro')
    val_recall = recall_score(y_val, y_val_pred, average='macro')
    val_f1 = f1_score(y_val, y_val_pred, average='macro')

    results.extend([
        {'Classifier': model_name, 'Metric': 'Accuracy', 'Score': val_accuracy, 'Set': 'Validation'},
        {'Classifier': model_name, 'Metric': 'Precision', 'Score': val_precision, 'Set': 'Validation'},
        {'Classifier': model_name, 'Metric': 'Recall', 'Score': val_recall, 'Set': 'Validation'},
        {'Classifier': model_name, 'Metric': 'F1', 'Score': val_f1, 'Set': 'Validation'}
    ])

    # Evaluate on test set
    y_test_pred = model.predict(X_test)
    test_accuracy = accuracy_score(y_test, y_test_pred)
    test_precision = precision_score(y_test, y_test_pred, average='macro')
    test_recall = recall_score(y_test, y_test_pred, average='macro')
    test_f1 = f1_score(y_test, y_test_pred, average='macro')

    results.extend([
        {'Classifier': model_name, 'Metric': 'Accuracy', 'Score': test_accuracy, 'Set': 'Test'},
        {'Classifier': model_name, 'Metric': 'Precision', 'Score': test_precision, 'Set': 'Test'},
        {'Classifier': model_name, 'Metric': 'Recall', 'Score': test_recall, 'Set': 'Test'},
        {'Classifier': model_name, 'Metric': 'F1', 'Score': test_f1, 'Set': 'Test'}
    ])

# Convert to DataFrame
results_df = pd.DataFrame(results)

# Save to CSV
output_path = os.path.join(RES_DIR, "s2_cv_validation_test_results.csv")
results_df.to_csv(output_path, index=False)
print(f"Results saved to: {output_path}")

models_path = os.path.join(RES_DIR, "s2_fitted_models.joblib")
joblib.dump(fitted_models, models_path)
print(f"Fitted models saved to: {models_path}")

# --- PLOTS ---

# 1. CV Results: Barplot (mean) + Stripplot (individual folds)
cv_results_df = results_df[results_df['Set'] == 'CV']
fig, ax = plt.subplots(figsize=(15, 8))
sns.set(style="whitegrid")

sns.barplot(
    data=cv_results_df,
    x="Classifier", y="Score", hue="Metric",
    ci=None, alpha=0.7, ax=ax
)
sns.stripplot(
    data=cv_results_df,
    x="Classifier", y="Score", hue="Metric",
    dodge=True, jitter=True, ax=ax,
    linewidth=1, marker="o", edgecolor="gray", alpha=0.4
)

handles, labels = ax.get_legend_handles_labels()
n_metrics = cv_results_df["Metric"].nunique()
ax.legend_.remove()
ax.legend(handles[:n_metrics], labels[:n_metrics],
          loc="best", bbox_to_anchor=(1.02, 0.8), title="Metric")

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

# 2. Validation & Test Results: Barplot (two rows, one column)
val_test_results_df = results_df[results_df['Set'].isin(['Validation', 'Test'])]

fig, axes = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

# Validation
sns.barplot(
    data=val_test_results_df[val_test_results_df['Set'] == 'Validation'],
    x="Classifier", y="Score", hue="Metric", ax=axes[0]
)
axes[0].set_title("Validation Set Scores", fontsize=16)
axes[0].set_ylabel("Score", fontsize=14)
axes[0].tick_params(labelsize=12)
for bars in axes[0].containers:
    axes[0].bar_label(bars, fmt="%.2f", label_type="edge", fontsize=10, padding=3)

# Test
sns.barplot(
    data=val_test_results_df[val_test_results_df['Set'] == 'Test'],
    x="Classifier", y="Score", hue="Metric", ax=axes[1]
)
axes[1].set_title("Test Set Scores", fontsize=16)
axes[1].set_xlabel("Classifier", fontsize=14)
axes[1].set_ylabel("Score", fontsize=14)
axes[1].tick_params(labelsize=12, rotation=45)
for bars in axes[1].containers:
    axes[1].bar_label(bars, fmt="%.2f", label_type="edge", fontsize=10, padding=3)

handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="upper right", bbox_to_anchor=(1.15, 0.9), title="Metric")
sns.despine()
plt.tight_layout()
fig.savefig(os.path.join(RES_DIR, "s2_validation_test_barplot.png"), dpi=300)
plt.show()
plt.close(fig)

# 3. Confusion Matrices for Validation and Test
fig, axes = plt.subplots(4, 2, figsize=(16, 16))  # 4 models, 2 columns (Validation, Test)
axes = axes.ravel()

# for i, (model_name, model) in enumerate(modelscv.items()):
#     model.fit(X_train, y_train)
for i, (model_name, model) in enumerate(fitted_models.items()):

    # Validation
    y_val_pred = model.predict(X_val)
    ConfusionMatrixDisplay.from_predictions(y_val, y_val_pred, ax=axes[2*i], cmap='Blues')
    axes[2*i].set_title(f"{model_name} (Validation)", fontsize=12)

    # Test
    y_test_pred = model.predict(X_test)
    ConfusionMatrixDisplay.from_predictions(y_test, y_test_pred, ax=axes[2*i+1], cmap='Blues')
    axes[2*i+1].set_title(f"{model_name} (Test)", fontsize=12)

plt.tight_layout()
fig.savefig(os.path.join(RES_DIR, "s2_all_models_validation_test_confusion_matrix.png"), dpi=300)
plt.show()
plt.close(fig)

