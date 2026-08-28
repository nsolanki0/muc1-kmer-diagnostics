
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
from sklearn.feature_selection import SelectFromModel
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
import argparse


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



def main():

    parser = argparse.ArgumentParser(description="Run ML pipeline on k-mer data.")

    parser.add_argument("--data", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--preprocessing",
    default="zscore",
    choices=[
        "zscore",
        "quantile",
        "log_robust",
        "yeojohnson",
        "clr",
        "signedlog"
    ])

    args = parser.parse_args()

    DATA = args.data
    RES_DIR = args.out

    print("Dataset: ", DATA)
    print("Result directory: ", RES_DIR)

    STANDARDISATION = args.preprocessing
    PREPROCESSOR = get_preprocessing_pipeline(STANDARDISATION)
    print(f"Using preprocessing: {STANDARDISATION}")

    os.makedirs(RES_DIR, exist_ok=True)
    
    
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
    # ---------------------- (i) L2 PIPELINE 1 ----------------------
    l2_pipeline1 = Pipeline([
        ("preprocessing", clone(PREPROCESSOR)),
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
    
    
    # ====================== 4. FEATURE SELECTION (IF BASELINE IS GOOD) ======================
    feature_counts = sorted(list(set([50, 100, 200, 500, 750, 1000, 2000, 4000, 5000])))
    #feature_counts = sorted(list(set([50, 100, 200, 500, 750, 1000, 2000, 4000, 6000, 7000])))
    
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
            ("preprocessing", clone(PREPROCESSOR)),
            ("selector", SelectFromModel(
                LogisticRegression(penalty="l2", solver="lbfgs", class_weight="balanced", C=0.8, max_iter=1000, random_state=42),
                max_features=n
            )),
            ("classifier", LogisticRegression(solver="liblinear", class_weight="balanced", max_iter=1000, random_state=42))
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
    best_n = 750
    print(f"Number of top features used in final model: {best_n}")
    
    final_pipeline = Pipeline([
        ("preprocessing", clone(PREPROCESSOR)),
        ("selector", SelectFromModel(
            LogisticRegression(penalty="l2", solver="lbfgs", class_weight="balanced", C=0.8, max_iter=1000, random_state=42),
            max_features=best_n
        )),
        ("classifier", LogisticRegression(solver="liblinear", class_weight="balanced", max_iter=1000, random_state=42))
    ])
    final_pipeline.fit(X_train, y_train)
    
    y_test_prob = final_pipeline.predict_proba(X_test)[:, 1]
    y_test_pred = np.where(y_test_prob >= threshold, "pos", "neg")
    
    print("\nThreshold used: ", threshold)
    print("\nBest features classification report:\n", classification_report(y_test, y_test_pred))
    
    report_dict_final = classification_report(y_test, y_test_pred, output_dict=True)
    report_df_final = pd.DataFrame(report_dict_final).transpose()
    report_df_final['Model'] = 'Logistic_Regression_liblinear'
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
    viz_preprocessor = final_pipeline.named_steps["preprocessing"]
    X_train_viz_scaled = viz_preprocessor.transform(X_train)  # NOT fit_transform, because it's already fitted!
    X_test_viz_scaled = viz_preprocessor.transform(X_test)
    
    # viz_preprocessor = clone(PREPROCESSOR)
    # X_train_viz_scaled = viz_preprocessor.fit_transform(X_train)
    # X_test_viz_scaled = viz_preprocessor.transform(X_test)
    
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
    #plt.title("Heatmap of Top 25 Positive and Bottom 25 Negative Features", fontsize=20)
    
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
    

if __name__ == "__main__":
    main()

    