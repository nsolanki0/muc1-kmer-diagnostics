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
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import (confusion_matrix, classification_report, ConfusionMatrixDisplay, silhouette_score)
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
from Bio import SeqIO
import gzip


# ====================== DIRECTORIES ======================

RES_DIR = "../results"
os.makedirs(RES_DIR, exist_ok=True)

DATA_REAL = "../data_real.csv.xz"
DATA_SIM = "../data_sim.csv.xz"
fasta_file = "../GCA_018469705.1_chr1_muc1_gene_region_27pos_27dupC_gene_region_101534451-101534482.fa.gz"

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


# ====================== 5. TEST SET EVALUATION (BEST FEATURES) ======================
threshold = 0.5
best_n = 750
print(f"Number of top features used in final model: {best_n}")

final_pipeline = Pipeline([
    ("scaler", StandardScaler()),
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

# Extract the scaler from the pipeline
pipeline_scaler = final_pipeline.named_steps["scaler"]
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
plt.title('PCA: Simulated vs Real Data')
plt.legend()
#plt.grid()
plt.tight_layout()
plt.savefig(os.path.join(RES_DIR, "s7_simReal_projection_colored_var_2.png"), dpi=300)
plt.show()
plt.close()


# ====================== 8. PCA LOADING ANALYSIS ======================

"""
The PCA is fitted ONLY on the simulated training data.
The real/test data are projected onto the same PCA axes.

This section:
1. Extracts the top features contributing to PC1 and PC2.
2. Separates positive and negative loadings.
3. Compares PCA features with:
   - train-only features
   - common train/test features
   - logistic-regression-selected features
4. Maps the top k-mers to the MUC1 sequence.
"""

# ----------------------------------------------------------------------
# 8.1 Fit PCA with 10 components
# ----------------------------------------------------------------------
n_components = 10

pca_scaler = StandardScaler()

X_train_scaled = pca_scaler.fit_transform(X_train)
X_val_scaled = pca_scaler.transform(X_val)
X_test_scaled = pca_scaler.transform(X_test)

# Fit PCA ONLY on simulated training data
pca_10 = PCA(n_components=n_components, random_state=42)
pca_10.fit(X_train_scaled)

# Project train and test
X_train_pca = pca_10.transform(X_train_scaled)
X_test_pca = pca_10.transform(X_test_scaled)

explained_var = pca_10.explained_variance_ratio_

print("\nExplained variance:")
for i, var in enumerate(explained_var, start=1):
    print(f"PC{i}: {var * 100:.2f}%")

print(f"Total variance explained by first {n_components} PCs: {explained_var.sum() * 100:.2f}%")

# ----------------------------------------------------------------------
# 8.2 Create loading DataFrame
# ----------------------------------------------------------------------
feature_names = X_train.columns.tolist()

# Extract loadings
loadings = pca_10.components_.T

# Create a DataFrame for loadings
pc_columns = [f"PC{i+1}" for i in range(n_components)]
loadings_df = pd.DataFrame(loadings, columns=pc_columns, index=feature_names)
loadings_df.to_csv(os.path.join(RES_DIR, "s8.2_pca_loadings_first10PCs.csv"))

# ----------------------------------------------------------------------
# 8.3 Top features for PC1 and PC2
# ----------------------------------------------------------------------
"""
Check the Loadings of PC1 and PC2 to see which original features contribute 
the most to each PC.

Extract Top Features for PC1 and PC2: Select the top N features 
(e.g., top 100) with the highest absolute loading values for each PC
to focus on the most influential features.

"""
top_n = 100

abs_loadings_pc1 = loadings_df["PC1"].abs()
abs_loadings_pc2 = loadings_df["PC2"].abs()

top_pc1_idx = abs_loadings_pc1.nlargest(top_n).index
top_pc2_idx = abs_loadings_pc2.nlargest(top_n).index

top_pc1 = loadings_df.loc[top_pc1_idx, "PC1"].sort_values(ascending=False)
top_pc2 = loadings_df.loc[top_pc2_idx, "PC2"].sort_values(ascending=False)

print("\nTop features for PC1:", top_pc1)
print("\nTop features for PC2:", top_pc2)

# ----------------------------------------------------------------------
# 8.4 Positive and negative contributors
# ----------------------------------------------------------------------
pos_pc1 = top_pc1[top_pc1 > 0]
neg_pc1 = top_pc1[top_pc1 < 0]

pos_pc2 = top_pc2[top_pc2 > 0]
neg_pc2 = top_pc2[top_pc2 < 0]

print("\nPositive contributors to PC1:", pos_pc1)
print("\nNegative contributors to PC1:", neg_pc1)
print("\nPositive contributors to PC2:", pos_pc2)
print("\nNegative contributors to PC2:", neg_pc2)

# ----------------------------------------------------------------------
# 8.5 Save summary
# ----------------------------------------------------------------------
summary = pd.DataFrame({
    "PC1_Top_Positive": pd.Series(pos_pc1.index.tolist()),
    "PC1_Top_Negative": pd.Series(neg_pc1.index.tolist()),
    "PC2_Top_Positive": pd.Series(pos_pc2.index.tolist()),
    "PC2_Top_Negative": pd.Series(neg_pc2.index.tolist())
})
summary.to_csv(os.path.join(RES_DIR, "s8.5_PC12_top_features.csv"), index=False)

# ----------------------------------------------------------------------
# 8.6 Common features between PC1 and PC2
# ----------------------------------------------------------------------
top_pc1_features = set(top_pc1.index)
top_pc2_features = set(top_pc2.index)

common_PC12 = top_pc1_features.intersection(top_pc2_features)
print(f"\nNumber of common features between PC1 and PC2: {len(common_PC12)}")

pd.DataFrame({
    "Feature": sorted(common_PC12)
}).to_csv(
    os.path.join(RES_DIR, "s8.6_PC1_PC2_common_features.csv"), index=False)

# ----------------------------------------------------------------------
# 8.7 Compare PCA features with train-only features
# ----------------------------------------------------------------------
unique_train_set = set(unique_to_X_train)

zero_filled_pc1 = unique_train_set.intersection(top_pc1_features)
zero_filled_pc2 = unique_train_set.intersection(top_pc2_features)

print(f"\nTrain-only features among top PC1 features: {len(zero_filled_pc1)}")
print(f"Train-only features among top PC2 features: {len(zero_filled_pc2)}")
print("PC1 train-only features:", zero_filled_pc1)
print("\nPC2 train-only features:", zero_filled_pc2)

# PC1 train-only features
pd.DataFrame({
    "Feature": sorted(zero_filled_pc1)
}).to_csv(
    os.path.join(RES_DIR, "s8.7_PC1_train_only_features.csv" ), index=False)

# PC2 train-only features
pd.DataFrame({
    "Feature": sorted(zero_filled_pc2)
}).to_csv(
    os.path.join(RES_DIR, "s8.7_PC2_train_only_features.csv"), index=False)

# ----------------------------------------------------------------------
# 8.8 Compare PCA features with common train/test features
# ----------------------------------------------------------------------
common_train_set = set(common_kmer_columns)

common_train_pc1 = common_train_set.intersection(top_pc1_features)
common_train_pc2 = common_train_set.intersection(top_pc2_features)

print(f"\nCommon train/test features among top PC1: {len(common_train_pc1)}")
print(f"Common train/test features among top PC2: {len(common_train_pc2)}")

# PC1 common train/test features
pd.DataFrame({
    "Feature": sorted(common_train_pc1)
}).to_csv(
    os.path.join(RES_DIR, "s8.8_PC1_common_train_test_features.csv"), index=False)

# PC2 common train/test features
pd.DataFrame({
    "Feature": sorted(common_train_pc2)
}).to_csv(
    os.path.join(RES_DIR, "s8.8_PC2_common_train_test_features.csv"), index=False)

# ----------------------------------------------------------------------
# 8.9 Compare PCA features with LR-selected features
# ----------------------------------------------------------------------
# Are the top features for PC1 and PC2 in 'Logistic regression selected features'?
#
# IMPORTANT:
# Use a separate variable name do avoid overwriting
# X_train.columns or the feature-alignment variable.

lr_selected_features = X_train.columns[final_pipeline.named_steps["selector"].get_support()]

lr_selected_set = set(lr_selected_features)

lr_pc1 = lr_selected_set.intersection(top_pc1_features)
lr_pc2 = lr_selected_set.intersection(top_pc2_features)

print(f"\nLR-selected features among top PC1: {len(lr_pc1)}")
print(f"LR-selected features among top PC2: {len(lr_pc2)}")
print("\nLR-selected + PC1:", lr_pc1)
print("\nLR-selected + PC2:", lr_pc2)

# PC1 + LR-selected
pd.DataFrame({
    "Feature": sorted(lr_pc1)
}).to_csv(
    os.path.join(RES_DIR, "s8.9_PC1_LR_selected_features.csv"), index=False)

# PC2 + LR-selected
pd.DataFrame({
    "Feature": sorted(lr_pc2)
}).to_csv(
    os.path.join(RES_DIR, "s8.9_PC2_LR_selected_features.csv"), index=False)

# Save overlap summary
overlap_summary = pd.DataFrame({
    "Category": [
        "Top PC1",
        "Top PC2",
        "PC1 & PC2",
        "PC1 & train-only",
        "PC2 & train-only",
        "PC1 & common train/test",
        "PC2 & common train/test",
        "PC1 & LR-selected",
        "PC2 & LR-selected"
    ],
    "Number": [
        len(top_pc1_features),
        len(top_pc2_features),
        len(common_PC12),
        len(zero_filled_pc1),
        len(zero_filled_pc2),
        len(common_train_pc1),
        len(common_train_pc2),
        len(lr_pc1),
        len(lr_pc2)
    ]
})

overlap_summary.to_csv(os.path.join(RES_DIR, "s8.9_pca_feature_overlap_summary.csv"), index=False)

# ----------------------------------------------------------------------
# 8.10 SAVE PCA LOADING SUMMARY
# ----------------------------------------------------------------------
pca_loading_summary = []

for pc_name, top_features in [("PC1", top_pc1), ("PC2", top_pc2)]:
    for rank, (feature, loading) in enumerate(top_features.items(), start=1):

        pca_loading_summary.append({
            "PC": pc_name,
            "Rank": rank,
            "Feature": feature,
            "Loading": loading,
            "Absolute_Loading": abs(loading),
            "Direction": (
                "Positive" if loading > 0 else "Negative"
            ),
            "Train_only": feature in unique_train_set,
            "Common_train_test": feature in common_train_set,
            "LR_selected": feature in lr_selected_set
        })

pca_loading_summary_df = pd.DataFrame(pca_loading_summary)
pca_loading_summary_df.to_csv(os.path.join(RES_DIR, "s8.10_PCA_top100_PC1_PC2_loading_analysis.csv"), index=False)


# ====================== 8.11 MUC1 k-mer POSITION MAPPING ======================

MUC1 = ("GCA_018469705.1_chr1_muc1_gene_region_27pos_27dupC_gene_region_101534451-101534482")

# Read MUC1 sequence
if fasta_file.endswith(".gz"):
    with gzip.open(fasta_file, "rt") as f:
        record = next(SeqIO.parse(f, "fasta"))
else:
    with open(fasta_file, "r") as f:
        record = next(SeqIO.parse(f, "fasta"))

genomic_sequence = str(record.seq)

print(f"\nLength of MUC1 sequence used: {len(genomic_sequence)}")

# Find k-mer positions
k = 31

def find_kmer_positions(sequence, kmer):
    positions = []

    for i in range(len(sequence) - len(kmer) + 1):
        if sequence[i:i + len(kmer)] == kmer:
            positions.append(i)

    return positions

# PC1
pc1_positions = {kmer: find_kmer_positions(genomic_sequence, kmer) for kmer in top_pc1_features}
# PC2
pc2_positions = {kmer: find_kmer_positions(genomic_sequence, kmer)for kmer in top_pc2_features}


pc1_position_rows = []

for kmer, positions in pc1_positions.items():
    # Save one row per position
    if len(positions) == 0:
        pc1_position_rows.append({
            "PC": "PC1",
            "Feature": kmer,
            "Position": np.nan,
            "Found": False
        })
    else:
        for position in positions:
            pc1_position_rows.append({
                "PC": "PC1",
                "Feature": kmer,
                "Position": position,
                "Found": True
            })

pc2_position_rows = []

for kmer, positions in pc2_positions.items():
    if len(positions) == 0:
        pc2_position_rows.append({
            "PC": "PC2",
            "Feature": kmer,
            "Position": np.nan,
            "Found": False
        })
    else:
        for position in positions:
            pc2_position_rows.append({
                "PC": "PC2",
                "Feature": kmer,
                "Position": position,
                "Found": True
            })

muc1_position_df = pd.DataFrame(pc1_position_rows + pc2_position_rows)

muc1_position_df["MUC1"] = MUC1
muc1_position_df["kmer_length"] = (muc1_position_df["Feature"].str.len())

muc1_position_df.to_csv(os.path.join(RES_DIR, "s8.11_PC1_PC2_MUC1_kmer_positions.csv"), index=False)

print("\nMUC1 k-mer position results saved to:", os.path.join(RES_DIR, "s8.11_PC1_PC2_MUC1_kmer_positions.csv"))


# ====================== 9. PCA: FIRST 10 PCs ======================

# 1. Fit PCA with 10 components
# n_components = 10

# pca_scaler = StandardScaler()

# X_train_scaled = pca_scaler.fit_transform(X_train)
# X_val_scaled = pca_scaler.transform(X_val)
# X_test_scaled = pca_scaler.transform(X_test)

# # Fit PCA ONLY on simulated training data
# pca_10 = PCA(n_components=n_components, random_state=42)
# pca_10.fit(X_train_scaled)

# # Project train and test
# X_train_pca = pca_10.transform(X_train_scaled)
# X_test_pca = pca_10.transform(X_test_scaled)

# explained_var = pca_10.explained_variance_ratio_

# 2. Prepare DataFrames
pc_columns = [f"PC{i+1}" for i in range(n_components)]

df_train_pos = pd.DataFrame(X_train_pca[y_train.values == "pos"], columns=pc_columns)
df_train_pos["Class"] = "Train (Positive)"

df_train_neg = pd.DataFrame(X_train_pca[y_train.values == "neg"], columns=pc_columns)
df_train_neg["Class"] = "Train (Negative)"

df_test_pos = pd.DataFrame(X_test_pca[y_test.values == "pos"], columns=pc_columns)
df_test_pos["Class"] = "Test (Positive)"

df_test_neg = pd.DataFrame(X_test_pca[y_test.values == "neg"], columns=pc_columns)
df_test_neg["Class"] = "Test (Negative)"

# Colors
colors = {
    "Train (Positive)": "blue",
    "Train (Negative)": "cyan",
    "Test (Positive)": "red",
    "Test (Negative)": "orange"
}

plot_data = [
    (df_train_pos, "Train (Positive)"),
    (df_train_neg, "Train (Negative)"),
    (df_test_pos, "Test (Positive)"),
    (df_test_neg, "Test (Negative)")
]

# 3. Create LOWER-TRIANGLE plot
fig, axes = plt.subplots(n_components, n_components, figsize=(24, 24))
fig.suptitle("Pairwise Scatter Plot Matrix — First 10 Principal Components", y=1.02, fontsize=18)

for row in range(n_components):
    for col in range(n_components):
        ax = axes[row, col]
        # Upper triangle: remove completely
        if col > row:
            ax.axis("off")
            continue
        # Diagonal: show PC name + explained variance
        if row == col:
            ax.text(0.5, 0.5, f"PC{row+1}\n" f"({explained_var[row] * 100:.1f}%)",
                ha="center", va="center", fontsize=11, transform=ax.transAxes)
            ax.set_xticks([])
            ax.set_yticks([])
            continue
        # -------------------------------------------------------------
        # LOWER TRIANGLE
        #
        # col = x-axis
        # row = y-axis
        #
        # Example:
        # row=8, col=0
        # → PC1 on x
        # → PC9 on y
        # -------------------------------------------------------------
        for df, label in plot_data:
            ax.scatter(df.iloc[:, col], df.iloc[:, row],
                c=colors[label], label=label, alpha=0.5, s=12)

        # Remove ticks from inner plots
        # if row < n_components - 1:
        #     ax.set_xticks([])
        # if col > 0:
        #     ax.set_yticks([])

        # X-axis label only on bottom row 
        if row == n_components - 1:
            ax.set_xlabel(f"PC{col+1} ({explained_var[col] * 100:.1f}%)")
        # Y-axis label only on left column
        if col == 0:
            ax.set_ylabel(f"PC{row+1} ({explained_var[row] * 100:.1f}%)")
        # Title showing the exact pair for each subplot
        ax.set_title(f"PC{col+1} (x) vs PC{row+1} (y)", fontsize=8)
        ax.grid(alpha=0.2)

# Single legend
handles, labels = axes[1, 0].get_legend_handles_labels()
fig.legend(handles, labels, loc="upper right", bbox_to_anchor=(0.98, 0.98))
plt.tight_layout(rect=[0, 0, 0.96, 0.98])
plt.savefig(os.path.join(RES_DIR, "s9_pca_pairplot_10_components_lower_triangle.png"), dpi=300, bbox_inches="tight")
plt.show()
plt.close()


# ====================== 9.1 SELECTED PCA PAIRS ======================
pairs_to_plot = [(0, 8)]   # PC1 vs PC9
for i, j in pairs_to_plot:
    plt.figure(figsize=(8, 6))

    # Train Positive
    mask = y_train.values == "pos"
    plt.scatter(
        X_train_pca[mask, i],
        X_train_pca[mask, j],
        c="blue", label="Train (Positive)", alpha=0.5, s=30)
    # Train Negative
    mask = y_train.values == "neg"
    plt.scatter(
        X_train_pca[mask, i],
        X_train_pca[mask, j],
        c="cyan", label="Train (Negative)", alpha=0.5, s=30)
    # ---------------------------------------------------------
    # Test Positive
    mask = y_test.values == "pos"
    plt.scatter(
        X_test_pca[mask, i],
        X_test_pca[mask, j],
        c="red", label="Test (Positive)", alpha=0.5, s=30)
    # Test Negative
    mask = y_test.values == "neg"
    plt.scatter(
        X_test_pca[mask, i],
        X_test_pca[mask, j],
        c="orange", label="Test (Negative)", alpha=0.5, s=30)

    # Labels
    plt.xlabel(f"PC{i+1} ({explained_var[i] * 100:.1f}% variance)")
    plt.ylabel(f"PC{j+1} ({explained_var[j] * 100:.1f}% variance)")
    plt.title(f"PCA: PC{i+1} vs PC{j+1}\n" "Simulated Train vs Real Test")
    plt.legend()
    #plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(os.path.join(RES_DIR, f"s9.1_selected_pca_pair_PC{i+1}_vs_PC{j+1}.png"), dpi=300, bbox_inches="tight")
    plt.show()
    plt.close()


# ====================== 9.2 SILHOUETTE SCORE ======================
# (higher = better separation)
# ---------------------- (i) POSITIVE vs NEGATIVE ----------------------
# Measures how well the PCA representation separates positive vs negative, not how well it separates simulated vs real

# Combine simulated train and real test
X_pca_all = np.vstack([X_train_pca, X_test_pca])
y_all = np.concatenate([y_train.values, y_test.values])

# Numeric class labels
y_num = np.where(y_all == "pos", 0, 1)

# Silhouette using PC1 and PC2
silhouette_pc12 = silhouette_score(X_pca_all[:, :2], y_num)

print(f"Silhouette Score (PC1 vs PC2): {silhouette_pc12:.4f}")

# Save
pd.DataFrame({
    "Analysis": ["PC1_vs_PC2"],
    "Silhouette_Score": [silhouette_pc12]
}).to_csv(
    os.path.join(RES_DIR, "s9.2_PC1_vs_PC2_silhouette_positive_vs_negative.csv"), index=False)

# Silhouette using all PCs
silhouette_pc10 = silhouette_score(X_pca_all, y_num)

print(f"Silhouette Score (PC1-PC10): {silhouette_pc10:.4f}")

# Save
pd.DataFrame({
    "Analysis": ["PC1-PC10"],
    "Silhouette_Score": [silhouette_pc10]
}).to_csv(
    os.path.join(RES_DIR, "s9.2_PC1-PC10_silhouette_positive_vs_negative.csv"), index=False)


# ---------------------- (ii) SIMULATED vs REAL ----------------------
# How distinguishable are the simulated and real distributions: close to 1 – Simulated and real are strongly separated;
# around 0 – Strong overlap / poorly separated

# Combine simulated and real samples
X_pca_all = np.vstack([X_train_pca, X_test_pca])

# Dataset origin
# 0 = Simulated
# 1 = Real
origin_labels = np.concatenate([
    np.zeros(len(X_train_pca), dtype=int),
    np.ones(len(X_test_pca), dtype=int)
])

# Calculate silhouette scores
silhouette_pc12_origin = silhouette_score(X_pca_all[:, :2], origin_labels)
silhouette_pc10_origin = silhouette_score(X_pca_all, origin_labels)

print(f"\nSilhouette Score — Simulated vs Real (PC1-PC2): {silhouette_pc12_origin:.4f}")
print(f"Silhouette Score — Simulated vs Real (PC1-PC10): {silhouette_pc10_origin:.4f}")

# Save
silhouette_origin_df = pd.DataFrame({
    "Comparison": ["Simulated_vs_Real", "Simulated_vs_Real"],
    "Feature_Space": ["PC1-PC2", "PC1-PC10"],
    "Silhouette_Score": [silhouette_pc12_origin, silhouette_pc10_origin]
})

silhouette_origin_df.to_csv(os.path.join(RES_DIR,"s9.2_silhouette_simulated_vs_real.csv"), index=False)
print("\nSilhouette results saved to:", os.path.join(RES_DIR, "s9.2_silhouette_simulated_vs_real.csv"))


# ====================== 9.3 t-SNE ======================

# Combine scaled train and test
X_combined_scaled = np.vstack([X_train_scaled, X_test_scaled])
y_combined = np.concatenate([y_train.values, y_test.values])

# Fit t-SNE
tsne = TSNE(n_components=2, random_state=42, perplexity=30, max_iter=1000)
X_combined_tsne = tsne.fit_transform(X_combined_scaled)

# Split back into train and test
X_train_tsne = X_combined_tsne[:len(X_train_scaled)]
X_test_tsne = X_combined_tsne[len(X_train_scaled):]

# Plot
plt.figure(figsize=(8, 6))

# Simulated train positive
mask = y_train.values == "pos"
plt.scatter(
    X_train_tsne[mask, 0],
    X_train_tsne[mask, 1],
    c="blue", label="Simulated Train (Positive)", alpha=0.5)
# Simulated train negative
mask = y_train.values == "neg"
plt.scatter(
    X_train_tsne[mask, 0],
    X_train_tsne[mask, 1],
    c="cyan", label="Simulated Train (Negative)", alpha=0.5)

# Real test positive
mask = y_test.values == "pos"
plt.scatter(
    X_test_tsne[mask, 0],
    X_test_tsne[mask, 1],
    c="red", label="Real Test (Positive)", alpha=0.5)
# Real test negative
mask = y_test.values == "neg"
plt.scatter(
    X_test_tsne[mask, 0],
    X_test_tsne[mask, 1],
    c="orange", label="Real Test (Negative)", alpha=0.5)

plt.xlabel("t-SNE 1")
plt.ylabel("t-SNE 2")
plt.title("t-SNE: Simulated vs Real Data")
plt.legend()
#plt.grid()
plt.tight_layout()
plt.savefig(os.path.join(RES_DIR, "s9.3_tsne_simulated_vs_real.png"), dpi=300, bbox_inches="tight")
plt.show()
plt.close()


# ====================== 10. COMMON-FEATURE ANALYSIS ======================

"""
Separate analysis using ONLY k-mers present in both simulated
training data and real test data.

This does NOT replace the main analysis.

It creates a second branch so as to compare:

A. Full training feature space
B. Common train/test feature space
"""

# 10.1 Re-create raw feature matrices
X_test_common = df_real_wide.drop(["ID", "type"], axis=1)
y_test_common = df_real_wide["type"].copy()

X_sim_common = df_sim_wide.drop(["ID", "type"], axis=1)
y_sim_common = df_sim_wide["type"].copy()

# 10.2 Train/validation split
X_train_common, X_val_common, y_train_common, y_val_common = \
    train_test_split(X_sim_common,y_sim_common,
        test_size=0.2, stratify=y_sim_common, random_state=42)

# 10.3 Determine common features
common_features = X_train_common.columns.intersection(X_test_common.columns)
print(f"\nNumber of common train/test k-mers: {len(common_features)}")

# 10.4 Restrict ALL datasets to common features
X_train_common = X_train_common[common_features]

X_val_common = X_val_common.reindex(columns=common_features, fill_value=0)
X_test_common = X_test_common.reindex(columns=common_features, fill_value=0)

print("Common-feature train shape after alignment:", X_train_common.shape)
print("Common-feature validation shape after alignment:", X_val_common.shape)
print("Common-feature test shape after alignment:", X_test_common.shape)

assert list(X_train_common.columns) == list(X_test_common.columns)

print(f"Number of 'positive' and 'negative' samples in the common feature training data: {y_train_common.value_counts()}")
print(f"Number of 'positive' and 'negative' samples in the common feature validation data: {y_val_common.value_counts()}")
print(f"Number of 'positive' and 'negative' samples in the common feature testing data: {y_test_common.value_counts()}")


# ====================== 11. PCA — COMMON FEATURES ONLY ======================

"""
PCA using only features that occur in BOTH the simulated training
data and the real test data.

PCA is fitted ONLY on the simulated training data.
"""

# 11.1 Scale common features
common_scaler = StandardScaler()

X_train_common_scaled = common_scaler.fit_transform(X_train_common)
X_val_common_scaled = common_scaler.transform(X_val_common)
X_test_common_scaled = common_scaler.transform(X_test_common)

# 11.2 Fit PCA on simulated training data
pca_common = PCA(n_components=2, random_state=42)
pca_common.fit(X_train_common_scaled)

# 11.3 Transform train and real test
X_train_common_pca = pca_common.transform(X_train_common_scaled)
X_test_common_pca = pca_common.transform(X_test_common_scaled)
explained_var_common = (pca_common.explained_variance_ratio_)

print("\nCommon-feature PCA explained variance:")
print(f"PC1: {explained_var_common[0] * 100:.2f}%")
print(f"PC2: {explained_var_common[1] * 100:.2f}%")
print(f"Total: {explained_var_common.sum() * 100:.2f}%")

# 11.4 Plot
plt.figure(figsize=(10, 6))

# Train positive
mask = y_train_common.values == "pos"
plt.scatter(
    X_train_common_pca[mask, 0],
    X_train_common_pca[mask, 1],
    c="blue", label="Simulated Train (Positive)", alpha=0.5)
# Train negative
mask = y_train_common.values == "neg"
plt.scatter(
    X_train_common_pca[mask, 0],
    X_train_common_pca[mask, 1],
    c="cyan", label="Simulated Train (Negative)", alpha=0.5)
# ---------------------------------------------------------
# Test positive
mask = y_test_common.values == "pos"
plt.scatter(
    X_test_common_pca[mask, 0],
    X_test_common_pca[mask, 1],
    c="red", label="Real Test (Positive)", alpha=0.5)
# Test negative
mask = y_test_common.values == "neg"
plt.scatter(
    X_test_common_pca[mask, 0],
    X_test_common_pca[mask, 1],
    c="orange", label="Real Test (Negative)", alpha=0.5)

plt.xlabel(f"Principal Component 1 ({explained_var_common[0] * 100:.1f}%)")
plt.ylabel(f"Principal Component 2 ({explained_var_common[1] * 100:.1f}%)")
plt.title("PCA: Simulated vs Real Data")
plt.legend()
#plt.grid(alpha=0.25)
plt.tight_layout()
plt.savefig(os.path.join(RES_DIR, "s11_common_features_simReal_PCA_PC1_PC2.png"), dpi=300, bbox_inches="tight")
plt.show()
plt.close()

