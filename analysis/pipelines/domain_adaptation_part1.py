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
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import (confusion_matrix, classification_report, ConfusionMatrixDisplay,
                             make_scorer, accuracy_score, precision_score, 
                             recall_score, f1_score, precision_recall_curve)
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy.stats import t
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

RES_DIR = "../results"
os.makedirs(RES_DIR, exist_ok=True)

DATA_REAL = "../data_real.csv.xz"
DATA_SIM = "../data_sim.csv.xz"

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
common_kmer_columns = X_train_orig.columns.intersection(X_test.columns)
print(f"Number of common k-mers: {len(common_kmer_columns)}")

unique_to_X_train = X_train_orig.columns.difference(X_test.columns)
print(f"Number of features in X_train, not in X_test: {len(unique_to_X_train)}")

USE_VARIANCE_FILTER = False
if USE_VARIANCE_FILTER:
    selected_features = X_train_orig.columns[X_train_orig.var(axis=0) > 0.001]
else:
    selected_features = X_train_orig.columns

X_train = X_train_orig[selected_features]
X_val = X_val_orig[selected_features]
X_test = X_test.reindex(columns=selected_features, fill_value=0)

assert list(X_train.columns) == list(X_test.columns), "Train and test feature order/columns do not match!"

print("Raw Train data shape after alignment and without scaling:", X_train.shape)
print("Raw Validation data shape after alignment and without scaling:", X_val.shape)
print("Raw Real data shape after alignment and without scaling:", X_test.shape)

y_train = y_train_orig.copy()
y_val = y_val_orig.copy()

print(f"Number of 'positive' and 'negative' samples in the raw training data: {y_train.value_counts()}")
print(f"Number of 'positive' and 'negative' samples in the raw validation data: {y_val.value_counts()}")
print(f"Number of 'positive' and 'negative' samples in the raw testing data: {y_test.value_counts()}")


# ====================== 2. BASELINE MODEL EVALUATION (FULL FEATURES) ======================
# Define models
modelscv = {
    "Logistic Regression": Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(solver="liblinear", class_weight="balanced", max_iter=1000, random_state=42))
    ]),
    "Decision Tree": Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", DecisionTreeClassifier(random_state=42))
    ]),
    "Random Forest": Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", RandomForestClassifier(random_state=42))
    ]),
    "LinearSVC": Pipeline([
        ("scaler", StandardScaler()),
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
# ---------------------- (i) L2 PIPELINE 1 ----------------------
l2_pipeline1 = Pipeline([
    ("scaler", StandardScaler()),
    ("classifier", LogisticRegression(solver="liblinear", class_weight="balanced", max_iter=1000, random_state=42))
])
l2_pipeline1.fit(X_train, y_train)
y_test_pred_l2p1 = l2_pipeline1.predict(X_test)
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
    ("scaler", StandardScaler()),
    ("classifier", LogisticRegression(solver="liblinear", class_weight={"pos": 3, "neg": 1}, max_iter=1000, random_state=42))
])
l2_pipeline2.fit(X_train, y_train)

y_test_pred_l2p2 = l2_pipeline2.predict(X_test)
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
    ("scaler", StandardScaler()),
    ("classifier", LogisticRegression(penalty="l2", solver="lbfgs", class_weight='balanced', C=0.8, max_iter=1000, random_state=42))
])
l2_pipeline3.fit(X_train, y_train)

y_val_scores = l2_pipeline3.predict_proba(X_val)[:, 1]
precision, recall, thresholds = precision_recall_curve(y_val, y_val_scores, pos_label='pos')
f1_scores = 2 * (precision * recall) / (precision + recall + 1e-9)
best_threshold_l2p3 = thresholds[np.argmax(f1_scores)]
y_test_scores = l2_pipeline3.predict_proba(X_test)[:, 1]

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
    ("scaler", StandardScaler()),
    ("classifier", LogisticRegression(penalty="l2", solver="lbfgs", class_weight={'pos': 3, 'neg': 1}, C=0.8, max_iter=1000, random_state=42))
])
l2_pipeline4.fit(X_train, y_train)

y_val_scores = l2_pipeline4.predict_proba(X_val)[:, 1]
precision, recall, thresholds = precision_recall_curve(y_val, y_val_scores, pos_label='pos')
f1_scores = 2 * (precision * recall) / (precision + recall + 1e-9)
best_threshold_l2p4 = thresholds[np.argmax(f1_scores)]
y_test_scores = l2_pipeline4.predict_proba(X_test)[:, 1]

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
    ("scaler", StandardScaler()),
    ("classifier", LogisticRegression(penalty="l2", solver="lbfgs", class_weight={'pos': 5, 'neg': 1}, C=0.8, max_iter=1000, random_state=42))
])
l2_pipeline5.fit(X_train, y_train)

y_val_scores = l2_pipeline5.predict_proba(X_val)[:, 1]
precision, recall, thresholds = precision_recall_curve(y_val, y_val_scores, pos_label='pos')
f1_scores = 2 * (precision * recall) / (precision + recall + 1e-9)
best_threshold_l2p5 = thresholds[np.argmax(f1_scores)]
y_test_scores = l2_pipeline5.predict_proba(X_test)[:, 1]

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
    ("scaler", StandardScaler()),
    ("classifier", LogisticRegression(penalty='l1', solver='liblinear', class_weight='balanced', C=0.2, max_iter=1000, random_state=42))
])
l1_pipeline1.fit(X_train, y_train)

y_val_scores = l1_pipeline1.predict_proba(X_val)[:, 1]
precision, recall, thresholds = precision_recall_curve(y_val, y_val_scores, pos_label='pos')
f1_scores = 2 * (precision * recall) / (precision + recall + 1e-9)
best_threshold_l1p1 = thresholds[np.argmax(f1_scores)]
y_test_scores = l1_pipeline1.predict_proba(X_test)[:, 1]

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
    ("scaler", StandardScaler()),
    ("classifier", LogisticRegression(penalty='l1', solver='liblinear', class_weight={'pos': 3, 'neg': 1}, C=0.2, max_iter=1000, random_state=42))
])
l1_pipeline2.fit(X_train, y_train)

y_val_scores = l1_pipeline2.predict_proba(X_val)[:, 1]
precision, recall, thresholds = precision_recall_curve(y_val, y_val_scores, pos_label='pos')
f1_scores = 2 * (precision * recall) / (precision + recall + 1e-9)
best_threshold_l1p2 = thresholds[np.argmax(f1_scores)]
y_test_scores = l1_pipeline2.predict_proba(X_test)[:, 1]

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
    ("scaler", StandardScaler()),
    ("classifier", LogisticRegression(penalty='l1', solver='liblinear', class_weight={'pos': 5, 'neg': 1}, C=0.2, max_iter=1000, random_state=42))
])
l1_pipeline3.fit(X_train, y_train)

y_val_scores = l1_pipeline3.predict_proba(X_val)[:, 1]
precision, recall, thresholds = precision_recall_curve(y_val, y_val_scores, pos_label='pos')
f1_scores = 2 * (precision * recall) / (precision + recall + 1e-9)
best_threshold_l1p3 = thresholds[np.argmax(f1_scores)]
y_test_scores = l1_pipeline3.predict_proba(X_test)[:, 1]

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
    ("scaler", StandardScaler()),
    ("classifier", LogisticRegression(penalty='l1', solver='liblinear', class_weight={'pos': 3, 'neg': 1}, C=1.0, max_iter=1000, random_state=42))
])
l1_pipeline4.fit(X_train, y_train)

y_val_scores = l1_pipeline4.predict_proba(X_val)[:, 1]
precision, recall, thresholds = precision_recall_curve(y_val, y_val_scores, pos_label='pos')
f1_scores = 2 * (precision * recall) / (precision + recall + 1e-9)
best_threshold_l1p4 = thresholds[np.argmax(f1_scores)]
y_test_scores = l1_pipeline4.predict_proba(X_test)[:, 1]

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


# ====================== 4. FEATURE SELECTION (IF BASELINE IS GOOD) ======================
feature_counts = sorted(list(set([50, 100, 200, 500, 750, 1000, 2000, 4000, 5000])))

scoring = {
    'accuracy': make_scorer(accuracy_score),
    'precision': make_scorer(precision_score, average='macro'),
    'recall': make_scorer(recall_score, average='macro'),
    'f1': make_scorer(f1_score, average='macro')
}

results = []
summary_results = []

for n in feature_counts:
    print(f"\nEvaluating top {n} features...")
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("selector", SelectFromModel(
            LogisticRegression(penalty="l2", solver="lbfgs", class_weight={'pos': 5, 'neg': 1}, C=0.8, max_iter=1000, random_state=42),
            max_features=n
        )),
        ("classifier", LogisticRegression(penalty="l2", solver="lbfgs", class_weight={'pos': 5, 'neg': 1}, C=0.8, max_iter=1000, random_state=42))
    ])

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_results = cross_validate(pipeline, X_train, y_train, cv=skf, scoring=scoring)
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

    pipeline.fit(X_train, y_train)
    y_val_pred = pipeline.predict(X_val)
    val_accuracy = accuracy_score(y_val, y_val_pred)
    val_precision = precision_score(y_val, y_val_pred, average="macro")
    val_recall = recall_score(y_val, y_val_pred, average="macro")
    val_f1 = f1_score(y_val, y_val_pred, average="macro")

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

results_df = pd.DataFrame(results)
results_df.to_csv(os.path.join(RES_DIR, "s4_lr_cv_feature_selection_val_macro.csv"), index=False)
summary_df = pd.DataFrame(summary_results)
summary_df.to_csv(os.path.join(RES_DIR, "s4_lr_cv_summary_feature_selection_val_macro.csv"), index=False)

# Plotting
plt.figure(figsize=(16, 12))
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

plt.tight_layout()
plt.savefig(os.path.join(RES_DIR, "s4_lr_cv_feature_selection_all_metrics.png"), dpi=300)
plt.show()
plt.close()

best_result = max(results, key=lambda x: x['validation_recall'])
best_n = best_result['n_features']
print(f"Best number of features (by recall): {best_n} (Validation recall: {best_result['validation_recall']:.3f})")


# ====================== 5. TEST SET EVALUATION (BEST FEATURES) ======================
threshold = 0.5
best_n = 4000
print(f"Number of top features used in final model: {best_n}")

final_pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("selector", SelectFromModel(
        LogisticRegression(penalty="l2", solver="lbfgs", class_weight={'pos': 5, 'neg': 1}, C=0.8, max_iter=1000, random_state=42),
        max_features=best_n
    )),
    ("classifier", LogisticRegression(penalty="l2", solver="lbfgs", class_weight={'pos': 5, 'neg': 1}, C=0.8, max_iter=1000, random_state=42))
])
final_pipeline.fit(X_train, y_train)

y_test_prob = final_pipeline.predict_proba(X_test)[:, 1]
y_test_pred = np.where(y_test_prob >= threshold, "pos", "neg")

print("\nThreshold used: ", threshold)
print("\nBest features classification report:\n", classification_report(y_test, y_test_pred))

report_dict_final = classification_report(y_test, y_test_pred, output_dict=True)
report_df_final = pd.DataFrame(report_dict_final).transpose()
report_df_final['Model'] = 'Logistic_Regression_lbfgs'
cols = report_df_final.columns.tolist()
cols = cols[-1:] + cols[:-1]
report_df_final = report_df_final[cols]
report_df_final.loc['threshold'] = threshold
report_df_final.loc['best_n'] = best_n
report_df_final.to_csv(os.path.join(RES_DIR, f"s5_lr_feature_selection_{best_n}_classification_report_test.csv"), index=True)

cnf_matrix_final = confusion_matrix(y_test, y_test_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cnf_matrix_final)
disp.plot(cmap="Blues")
plt.title(f"Confusion Matrix ({best_n} features)")
plt.savefig(os.path.join(RES_DIR, f"s5_lr_feature_selection_{best_n}_confMat_test.png"), dpi=300, bbox_inches="tight")
plt.show()
plt.close()


# Extract selected features
selector = final_pipeline.named_steps["selector"]
selected_mask = selector.get_support()
selected_features = X_train.columns[selected_mask]
print(f"Actual number of selected features: {len(selected_features)}")

final_classifier = final_pipeline.named_steps["classifier"]
coefficients = final_classifier.coef_[0]
feature_coef_df = pd.DataFrame({
    "Feature": selected_features,
    "Coefficient": coefficients
})
feature_coef_df["Abs_Coefficient"] = feature_coef_df["Coefficient"].abs()
feature_coef_df = feature_coef_df.sort_values("Abs_Coefficient", ascending=False)
feature_coef_df.to_csv(os.path.join(RES_DIR, f"s5_selected_features_coefficients_{best_n}.csv"), index=False)

# Extract predictions
prediction_results = pd.DataFrame({
    "sample_ID": X_test.index,
    "true_label": y_test.values,
    "probability_pos": y_test_prob,
    "prediction": y_test_pred
})
prediction_results.to_csv(os.path.join(RES_DIR, f"s5_test_sample_predictions_{best_n}.csv"), index=False)


# ====================== 6. FEATURE ANALYSIS (HEATMAP) ======================
# 1. Extract selected features and model coefficients
selector = final_pipeline.named_steps["selector"]
final_classifier = final_pipeline.named_steps["classifier"]
selected_indices = selector.get_support(indices=True)
selected_features = X_train.columns[selected_indices]
coefficients = final_classifier.coef_[0]

feature_coef_df = pd.DataFrame({
    "Feature": selected_features, 
    "Coefficient": coefficients
    })
feature_coef_df = feature_coef_df.sort_values(by="Coefficient", ascending=False)

top_25_positive = feature_coef_df.head(25)
bottom_25_negative = feature_coef_df.tail(25)
selected_features_df = pd.concat([top_25_positive, bottom_25_negative])
selected_feature_names = selected_features_df["Feature"].tolist()

# 2. Prepare scaled test data for visualization
pipeline_scaler = final_pipeline.named_steps["scaler"]  # Extract the scaler from the pipeline
X_train_viz_scaled = pipeline_scaler.transform(X_train)
X_test_viz_scaled = pipeline_scaler.transform(X_test)

X_test_viz_scaled_df = pd.DataFrame(
    X_test_viz_scaled, 
    columns=X_test.columns, 
    index=X_test.index
    )
X_test_selected = X_test_viz_scaled_df[selected_feature_names].copy()
X_test_selected["actual"] = y_test.map({"pos":1, "neg":0}).astype(int)
X_test_selected["predicted"] = y_test_pred.copy()
X_test_selected["predicted"] = X_test_selected["predicted"].map({"pos":1, "neg":0})
X_test_selected = X_test_selected.sort_values(by="predicted", ascending=False)

# 3. Prepare heatmap matrix
heatmap_data = X_test_selected.transpose()
rows_order = selected_features_df["Feature"].tolist() + ["actual", "predicted"]
heatmap_data = heatmap_data.reindex(rows_order)

# 4. Plot heatmap
plt.figure(figsize=(15,12))
ax = sns.heatmap(
    heatmap_data.iloc[:-2],
    cmap="RdBu_r",
    xticklabels=False,
    yticklabels=True,
    center=0,
    cbar_kws={"label":"Standardized Feature Value"}
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
plt.savefig(os.path.join(RES_DIR, "s6_lr_feature_heatmap_top25_bottom25.png"), dpi=300, bbox_inches="tight")
plt.show()
plt.close()


# ====================== 7. Dimensionality Reduction (PCA) ======================
pca_pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("pca", PCA(n_components=2, random_state=42))
])
pca_pipeline.fit(X_train)
X_sim_train_pca = pca_pipeline.transform(X_train)
X_real_test_pca = pca_pipeline.transform(X_test)

plt.figure(figsize=(10, 6))
plt.scatter(X_sim_train_pca[y_train == "pos", 0], X_sim_train_pca[y_train == "pos", 1], c='blue', label='Simulated Train (Positive)', alpha=0.5)
plt.scatter(X_sim_train_pca[y_train == "neg", 0], X_sim_train_pca[y_train == "neg", 1], c='cyan', label='Simulated Train (Negative)', alpha=0.5)
plt.scatter(X_real_test_pca[y_test == "pos", 0], X_real_test_pca[y_test == "pos", 1], c='red', label='Real Test (Positive)', alpha=0.5)
plt.scatter(X_real_test_pca[y_test == "neg", 0], X_real_test_pca[y_test == "neg", 1], c='orange', label='Real Test (Negative)', alpha=0.5)

explained_var = pca_pipeline.named_steps["pca"].explained_variance_ratio_

plt.xlabel(f'Principal Component 1 ({explained_var[0]*100:.1f}%)')
plt.ylabel(f'Principal Component 2 ({explained_var[1]*100:.1f}%)')
plt.title('PCA: Simulated+Real vs Real Data')
plt.legend()
#plt.grid()
plt.tight_layout()
plt.savefig(os.path.join(RES_DIR, "s7_simReal_projection_colored_var_2.png"), dpi=300)
plt.show()
plt.close()

