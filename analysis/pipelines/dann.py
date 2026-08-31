#!/usr/bin/env python3

# ====================== IMPORTS ======================
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    classification_report,
    f1_score
)
import torch
import torch.nn as nn
import torch.optim as optim


# ====================== CONFIGURATION ======================
RANDOM_STATE = 42
TEST_SIZE_REAL_SPLIT = 0.5          # For splitting real data into train/val and test
TEST_SIZE_VAL_SPLIT = 0.2           # For splitting simulated+real into train/val

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

# Training set
X_train_tun = X_train_orig.copy()
y_train_tun = y_train_orig.copy()

# Final validation set: 18 real-world samples
X_val_tun = X_real_val.copy()
y_val_tun = y_real_val.copy()

# Final test set: untouched
X_test_tun = X_test
y_test_tun = y_test

print("Tuning train data shape:", X_train_tun.shape)
print("Tuning validation data shape:", X_val_tun.shape)
print("Tuning test data shape:", X_test_tun.shape)

print(f"Number of 'positive' and 'negative' samples in the tuning training data: {y_train_tun.value_counts()}")
print(f"Number of 'positive' and 'negative' samples in the tuning validation data: {y_val_tun.value_counts()}")
print(f"Number of 'positive' and 'negative' samples in the tuning test data: {y_test_tun.value_counts()}")



# ====================== 4. DANN ARCHITECTURE ======================

# 1) Scale data
tun_pipeline = Pipeline([
    ("scaler", StandardScaler())
])

X_train_tun_scaled = tun_pipeline.fit_transform(X_train_tun)
X_val_tun_scaled = tun_pipeline.transform(X_val_tun)
X_test_tun_scaled = tun_pipeline.transform(X_test_tun)

# Encode labels
label_mapping = {"neg": 0, "pos": 1}

y_train_encoded = y_train_tun.map(label_mapping)
y_val_encoded = y_val_tun.map(label_mapping)
y_test_encoded = y_test_tun.map(label_mapping).values.copy()
                                           
# Convert to PyTorch tensors
X_train = torch.FloatTensor(X_train_tun_scaled)
y_train = torch.FloatTensor(y_train_encoded.values.copy())

X_val_tensor = torch.FloatTensor(X_val_tun_scaled)
y_val_tensor = torch.FloatTensor(y_val_encoded.values.copy())

X_test_tensor = torch.FloatTensor(X_test_tun_scaled)

# Domain labels (0 = simulated, 1 = real=
domain_labels = torch.FloatTensor(
    (origin_train == "real").astype(float).values
)

# 2) Define models
feature_dim = X_train.shape[1]
latent_dim = 10         # CUSTOM PARAMETER

feature_extractor = nn.Sequential(
    nn.Linear(feature_dim, latent_dim),
    nn.ReLU()
)

label_classifier = nn.Sequential(
    nn.Linear(latent_dim, 1)
)

domain_classifier = nn.Sequential(
    nn.Linear(latent_dim, 1)
)

# 3) Optimizer
optimizer = torch.optim.Adam(
    list(feature_extractor.parameters()) +
    list(label_classifier.parameters()) +
    list(domain_classifier.parameters()),
    lr=1e-3,        # CUSTOM PARAMETER
    weight_decay=1e-5
)

criterion = nn.BCEWithLogitsLoss()

# 4) Gradient Reversal Layer
class GradientReversal(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.alpha * grad_output, None

def grad_reverse(x, alpha):
    return GradientReversal.apply(x, alpha)

# ====================== 5. TRAINING ======================

# 5) Training loop
history = []

epochs = 200
patience = 5        # CUSTOM PARAMETER

best_val_f1 = 0
no_improvement = 0

lambda_domain = 0.01        # CUSTOM PARAMETER

for epoch in range(epochs):
    feature_extractor.train()
    label_classifier.train()
    domain_classifier.train()

    # Dynamic gradient reversal strength
    p = epoch / epochs
    alpha = 2 / (1 + np.exp(-10 * p)) - 1

    features = feature_extractor(X_train)
    label_logits = label_classifier(features).squeeze()
    reverse_features = grad_reverse(features, alpha)

    domain_logits = domain_classifier(reverse_features).squeeze()

    label_loss = criterion(label_logits, y_train)
    domain_loss = criterion(domain_logits, domain_labels)

    total_loss = label_loss + lambda_domain * domain_loss

    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()

    # Validation
    if epoch % 5 == 0:
        feature_extractor.eval()
        label_classifier.eval()
        domain_classifier.eval()

        # Validation label performance
        with torch.no_grad():
            val_features = feature_extractor(X_val_tensor)
            val_logits = label_classifier(val_features)
            val_probs = torch.sigmoid(val_logits)

            val_pred = (val_probs > 0.5).float()
            val_f1 = f1_score(y_val_tensor.numpy(), val_pred.numpy())

            # Training domain performance
            domain_probs = torch.sigmoid(domain_logits)
            domain_pred = (domain_probs > 0.5).float()

            train_domain_acc = (domain_pred.squeeze() == domain_labels).float().mean().item()

        history.append({
            "epoch": epoch,
            "label_loss": label_loss.item(),
            "domain_loss": domain_loss.item(),
            "validation_f1": val_f1,
            "train_domain_accuracy": train_domain_acc,
            "alpha": alpha
        })

        print(
            f"Epoch {epoch:3d} | "
            f"LabelLoss={label_loss:.4f} | "
            f"DomainLoss={domain_loss:.4f} | "
            f"ValF1={val_f1:.3f} | "
            f"TrainDomainAcc={train_domain_acc:.3f}"
        )

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_epoch = epoch

            no_improvement = 0
            torch.save(
                feature_extractor.state_dict(),
                os.path.join(RES_DIR, "best_feature_extractor.pth"))

            torch.save(
                label_classifier.state_dict(),
                os.path.join(RES_DIR,"best_label_classifier.pth"))
        else:
            no_improvement += 1
            if no_improvement >= patience:
                print("Early stopping.")
                break


# ====================== 6. TEST SET EVALUATION ======================
# 6) Test evaluation
feature_extractor.load_state_dict(
    torch.load(os.path.join(RES_DIR, "best_feature_extractor.pth")))

label_classifier.load_state_dict(
    torch.load(os.path.join(RES_DIR, "best_label_classifier.pth")))

feature_extractor.eval()
label_classifier.eval()

## ---------- (i) Latent features visualizaiton ---------- ##
# After training, extract latent features for training data
with torch.no_grad():
    latent_features = feature_extractor(X_train).numpy()

# Plot first two dimensions of latent space
plt.figure(figsize=(10, 6))
plt.scatter(latent_features[origin_train == "simulated", 0],
            latent_features[origin_train == "simulated", 1],
            label="Simulated", alpha=0.5)
plt.scatter(latent_features[origin_train == "real", 0],
            latent_features[origin_train == "real", 1],
            label="Real", alpha=0.5)
plt.xlabel("Latent Dimension 1")
plt.ylabel("Latent Dimension 2")
plt.title("Latent Space: Simulated vs. Real")
plt.legend()
plt.savefig(os.path.join(RES_DIR, "latent_space_plot.png"), dpi=300)
plt.show()
plt.close()

## ---------- (ii) Latent features PCA visualizaiton ---------- ##
# Reduce latent features to 2D using PCA
pca = PCA(n_components=2)
latent_pca = pca.fit_transform(latent_features)
explained_var = pca.explained_variance_ratio_

# Create a figure
plt.figure(figsize=(10, 8))

# Plot each group with the specified colors
# Simulated Pos (blue)
plt.scatter(
    latent_pca[(origin_train == "simulated") & (y_train_tun == "pos"), 0],
    latent_pca[(origin_train == "simulated") & (y_train_tun == "pos"), 1],
    color='blue',
    alpha=0.5,
    label='Simulated Pos'
)
# Simulated Neg (cyan)
plt.scatter(
    latent_pca[(origin_train == "simulated") & (y_train_tun == "neg"), 0],
    latent_pca[(origin_train == "simulated") & (y_train_tun == "neg"), 1],
    color='cyan',
    alpha=0.5,
    label='Simulated Neg'
)
# Real Pos (red)
plt.scatter(
    latent_pca[(origin_train == "real") & (y_train_tun == "pos"), 0],
    latent_pca[(origin_train == "real") & (y_train_tun == "pos"), 1],
    color='red',
    alpha=0.5,
    label='Real Pos'
)
# Real Neg (orange)
plt.scatter(
    latent_pca[(origin_train == "real") & (y_train_tun == "neg"), 0],
    latent_pca[(origin_train == "real") & (y_train_tun == "neg"), 1],
    color='orange',
    alpha=0.5,
    label='Real Neg'
)

# Add labels and title
plt.xlabel(f'Principal Component 1 ({explained_var[0]*100:.1f}%)')
plt.ylabel(f'Principal Component 2 ({explained_var[1]*100:.1f}%)')
plt.title('PCA of Learned Latent Features by Domain and Class')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)

# Save and show
plt.tight_layout()
plt.savefig(os.path.join(RES_DIR, "latent_space_pca.png"), dpi=300)
plt.show()
plt.close()

## ---------- (iii) Test set evaluation ---------- ##
with torch.no_grad():
    test_features = feature_extractor(X_test_tensor)
    test_logits = label_classifier(test_features)

    test_probs = torch.sigmoid(test_logits)
    test_pred = (test_probs > 0.5).float()

test_cm = confusion_matrix(y_test_encoded, test_pred.numpy())
test_f1 = f1_score(y_test_encoded, test_pred.numpy())

print(test_cm)
print(test_f1)

# ====================== 7. RESULTS ======================
# 7) Save results
disp = ConfusionMatrixDisplay(confusion_matrix=test_cm)
disp.plot(cmap="Blues")
plt.title(f"Confusion Matrix")
plt.savefig(os.path.join(RES_DIR, "adversarial_test_cm.png"), dpi=300, bbox_inches="tight")
plt.show()
plt.close()

# Classification report
report = classification_report(y_test_encoded, test_pred.numpy(), target_names=["neg", "pos"], output_dict=True)

report_df = pd.DataFrame(report).transpose()
report_df.to_csv(os.path.join(RES_DIR, "adversarial_test_classification_report.csv"))

# Predictions
prediction_results = pd.DataFrame({
    "ID": test_df_wide["ID"],
    "True_Label": y_test_tun.values,
    "Prediction": test_pred.numpy().astype(int).flatten(),
    "Probability": test_probs.numpy().flatten()
})

prediction_results["Correct"] = (
    prediction_results["Prediction"] == prediction_results["True_Label"])

prediction_results.to_csv(os.path.join(RES_DIR, "adversarial_test_predictions.csv"), index=False)

# Parameters
params = {
    "latent_dim": latent_dim,
    #"epochs": epochs,
    "learning_rate": 1e-3,
    "lambda_domain": lambda_domain,
    "patience": patience,
    "random_state": RANDOM_STATE,
    "best_validation_f1": best_val_f1,
    "best_epoch": best_epoch
}

with open(os.path.join(RES_DIR, "experiment_parameters.json"), "w") as f:
    json.dump(params, f, indent=4)


# ====================== 8. TRAINING HISTORY ======================
# Training histroy
history_df = pd.DataFrame(history)
history_df.to_csv(os.path.join(RES_DIR, "adversarial_training_history.csv"), index=False)


plt.figure(figsize=(12, 8))

plt.subplot(2, 2, 1)
plt.plot(history_df["epoch"], history_df["label_loss"], label="Label Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()

plt.subplot(2, 2, 2)
plt.plot(history_df["epoch"], history_df["domain_loss"], label="Domain Loss", color="orange")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()

plt.subplot(2, 2, 3)
plt.plot(history_df["epoch"], history_df["validation_f1"], label="Validation F1", color="green")
plt.xlabel("Epoch")
plt.ylabel("F1 Score")
plt.legend()

plt.subplot(2, 2, 4)
plt.plot(history_df["epoch"], history_df["train_domain_accuracy"], label="Train domain accuracy", color="red")
plt.xlabel("Epoch")
plt.ylabel("Alpha")
plt.legend()

plt.tight_layout()
plt.savefig(os.path.join(RES_DIR, "adversarial_training_history_plot.png"), dpi=300)
plt.show()
plt.close()
