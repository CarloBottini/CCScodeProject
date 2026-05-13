
"""
baseline_ecfp4.py

ECFP4 + Feed-Forward Neural Network baseline for CCS prediction.

Purpose: provide a NON-GNN baseline so we can fairly compare our Chemprop MPNN against a traditional fingerprint-based approach.

Key points:
- Same dataset, same scaffold-balanced split, same seed, same extra features
  (Adduct one-hot+Dimer flag+MW) as the GNN

This script is CPU friendly
"""

import os
import math
import time
import warnings

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

import lightning.pytorch as pl
from lightning.pytorch.callbacks import EarlyStopping, Callback

from rdkit import Chem
from rdkit.Chem import Descriptors
# Modern RDKit fingerprint API (replaces the deprecated GetMorganFingerprintAsBitVect)
from rdkit.Chem import rdFingerprintGenerator

# Reuse the same split function as the GNN to guarantee identical test set
from chemprop.data import make_split_indices

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

start_time = time.time()
SEED = 42
pl.seed_everything(SEED, workers=True)

# 1. LOAD AND CLEAN DATA (identical to the GNN pipeline)
print("1. DATA PREPARATION")

csv_path = "data/METLIN_IMS_dimers_rmTM.csv"
df = pd.read_csv(csv_path)
df = df.dropna(subset=['smiles', 'CCS_AVG', 'Adduct', 'Dimer.1'])
print(f"Rows remaining after cleaning missing values: {df.shape[0]}")

# 2. ONE-HOT ENCODING (same as GNN script)
print("\n2. ONE HOT ENCODING")
df_encoded = pd.get_dummies(df, columns=['Adduct', 'Dimer.1'], dtype=float)
extra_feature_columns = [
    col for col in df_encoded.columns
    if col.startswith('Adduct_') or col.startswith('Dimer.1_')
]
print(f"Extra categorical features: {extra_feature_columns}")

# 3. ECFP4 FINGERPRINT GENERATION
# ECFP4=Extended Connectivity FingerPrint with diameter 4 (radius=2)
# We hash each molecule into a fixed-length bit vector. 2048 bits is the standard; smaller (1024) is also possible
print("\n3. COMPUTING ECFP4 FINGERPRINTS (radius=2, 2048 bits)")

FP_RADIUS= 2          # radius 2 = ECFP4 diameter
FP_NBITS= 2048        # standard size; trades off collisions vs. dimensionality

morgan_gen= rdFingerprintGenerator.GetMorganGenerator(
    radius=FP_RADIUS, fpSize=FP_NBITS
)

fingerprints= []   # ECFP4 bit vectors (one per molecule)
extra_features= [] # Adduct one-hot + Dimer flag + MW (same as GNN's x_d)
targets= []        # CCS values
valid_indices= []  # row indices that survived (for the split later)
mols= []           # RDKit Mol objects, needed for scaffold split
invalid_mols= 0

for index, row in df_encoded.iterrows():
    smi= row['smiles']
    mol= Chem.MolFromSmiles(smi)
    if mol is None:
        invalid_mols+=1
        continue

    # Fingerprint as a numpy array of 0/1 floats
    fp= morgan_gen.GetFingerprintAsNumPy(mol).astype(np.float32)

    # Molecular weight, doubled for dimers (same rule as in the GNN code)
    mw= Descriptors.MolWt(mol)
    if row.get('Dimer.1_Dimer', 0.0) == 1.0:
        mw=mw * 2

    categorical_values= row[extra_feature_columns].values.astype(np.float32)
    extra= np.append(categorical_values, mw).astype(np.float32)

    fingerprints.append(fp)
    extra_features.append(extra)
    targets.append(row['CCS_AVG'])
    valid_indices.append(index)
    mols.append(mol)

fingerprints = np.stack(fingerprints)          # shape (N, 2048)
extra_features = np.stack(extra_features)      # shape (N, n_extra)
targets = np.array(targets, dtype=np.float32).reshape(-1, 1)

print(f"Valid molecules: {len(mols)}  (Discarded: {invalid_mols})")
print(f"Fingerprint matrix shape: {fingerprints.shape}")
print(f"Extra-features matrix shape: {extra_features.shape}")


# 4. SAME SCAFFOLD-BALANCED SPLIT AS THE GNN (critical for fair comparison)
print("\n4. SCAFFOLD-BALANCED SPLIT (80/10/10, seed=42)")

indices_train, indices_val, indices_test = make_split_indices(
    mols,
    split="scaffold_balanced",
    sizes=(0.8, 0.1, 0.1),
    seed=SEED,
)
idx_train = np.array(indices_train[0], dtype=int)
idx_val   = np.array(indices_val[0],   dtype=int)
idx_test  = np.array(indices_test[0],  dtype=int)

X_train_fp,    X_val_fp,    X_test_fp    = fingerprints[idx_train],    fingerprints[idx_val],    fingerprints[idx_test]
X_train_extra, X_val_extra, X_test_extra = extra_features[idx_train],  extra_features[idx_val],  extra_features[idx_test]
y_train,       y_val,       y_test       = targets[idx_train],         targets[idx_val],         targets[idx_test]

print(f"Train: {len(idx_train)} | Val: {len(idx_val)} | Test: {len(idx_test)}")

# 5. SCALING
# Fingerprints are already 0/1, no scaling needed.
# Extra features (MW especially) and targets benefit from standardization,exactly like in the GNN pipeline
extra_scaler = StandardScaler().fit(X_train_extra)
X_train_extra = extra_scaler.transform(X_train_extra).astype(np.float32)
X_val_extra   = extra_scaler.transform(X_val_extra).astype(np.float32)
X_test_extra  = extra_scaler.transform(X_test_extra).astype(np.float32)

target_scaler = StandardScaler().fit(y_train)
y_train_scaled = target_scaler.transform(y_train).astype(np.float32)
y_val_scaled   = target_scaler.transform(y_val).astype(np.float32)
# y_test is left UNSCALED -> we evaluate in original Å^2 units

# Concatenate ECFP4 + extra features
X_train = np.concatenate([X_train_fp, X_train_extra], axis=1).astype(np.float32)
X_val   = np.concatenate([X_val_fp,   X_val_extra],   axis=1).astype(np.float32)
X_test  = np.concatenate([X_test_fp,  X_test_extra],  axis=1).astype(np.float32)

INPUT_DIM = X_train.shape[1]
print(f"Final input dimension (FP + extras): {INPUT_DIM}")

# 6. DATALOADERS
BATCH_SIZE = 128

def make_loader(X, y, shuffle):
    ds = TensorDataset(torch.from_numpy(X), torch.from_numpy(y))
    return DataLoader(ds, batch_size=BATCH_SIZE, shuffle=shuffle, num_workers=0)

loader_train = make_loader(X_train, y_train_scaled, shuffle=True)
loader_val   = make_loader(X_val,   y_val_scaled,   shuffle=False)
loader_test  = make_loader(X_test,  y_test,         shuffle=False)  # unscaled y

# 7. FFN MODEL
# Mirrors the structure of Chemprop's RegressionFFN: a stack of Linear+ReLU+Dropout
# blocks ending in a linear regression head. The output is in the SCALED space during training; it is un-scale only at test time
class FFNBaseline(pl.LightningModule):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 512,
        n_layers: int = 3,
        dropout: float = 0.2,
        lr: float = 1e-3,
    ):
        super().__init__()
        self.save_hyperparameters()

        layers = []
        in_d = input_dim
        # n_layers hidden layers, each followed by ReLU+Dropout
        for _ in range(n_layers):
            layers += [nn.Linear(in_d, hidden_dim), nn.ReLU(), nn.Dropout(dropout)]
            in_d = hidden_dim
        layers += [nn.Linear(in_d, 1)]  # regression head
        self.net = nn.Sequential(*layers)
        self.loss_fn = nn.MSELoss()

    def forward(self, x):
        return self.net(x)

    def training_step(self, batch, _):
        x, y = batch
        pred = self(x)
        loss = self.loss_fn(pred, y)
        self.log("train_loss", loss, prog_bar=True, on_epoch=True, on_step=False)
        return loss

    def validation_step(self, batch, _):
        x, y = batch
        pred = self(x)
        loss = self.loss_fn(pred, y)
        self.log("val_loss", loss, prog_bar=True, on_epoch=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)


class BestWeightsTracker(Callback):
    #Save the best weights seen during training (by val_loss). Same idea as the GNN code
    def __init__(self, monitor="val_loss"):
        self.monitor = monitor
        self.best_score = None
        self.best_state_dict = None

    def on_validation_end(self, trainer, pl_module):
        if trainer.sanity_checking:
            return
        m = trainer.callback_metrics.get(self.monitor)
        if m is None:
            return
        score = float(m.detach().cpu().item())
        if self.best_score is None or score < self.best_score:
            self.best_score = score
            self.best_state_dict = {
                k: v.detach().cpu().clone() for k, v in pl_module.state_dict().items()
            }

# 8. TRAINING
# Sensible default hyperparameters for a baseline. 
# Can be tuned later with Optuna, in a separate script. The goal here is a strong, clean reference number, not another HP search
print("\n5. TRAINING THE FFN BASELINE")

model = FFNBaseline(
    input_dim=INPUT_DIM,
    hidden_dim=512,
    n_layers=3,
    dropout=0.2,
    lr=1e-3,
)

early_stop = EarlyStopping(monitor="val_loss", patience=10, mode="min", verbose=True)
best_weights = BestWeightsTracker(monitor="val_loss")

trainer = pl.Trainer(
    max_epochs=200,             # high cap; early stopping ends training naturally
    accelerator="cpu",          # explicit CPU; change to "auto" if there is a GPU
    devices=1,
    logger=False,
    enable_progress_bar=True,
    enable_checkpointing=False,
    enable_model_summary=True,
    callbacks=[early_stop, best_weights],
)
trainer.fit(model, loader_train, loader_val)

# Restore the best weights before evaluating on the test set
if best_weights.best_state_dict is not None:
    model.load_state_dict(best_weights.best_state_dict)

# 9. TEST EVALUATION (unscaled Å^2)
print("\n6. EVALUATING ON TEST SET")

model.eval()
all_preds_scaled = []
all_true = []
with torch.no_grad():
    for x, y in loader_test:
        pred = model(x).cpu().numpy()
        all_preds_scaled.append(pred)
        all_true.append(y.numpy())

all_preds_scaled = np.concatenate(all_preds_scaled, axis=0)
all_true = np.concatenate(all_true, axis=0)

# Convert predictions back to Å^2 to match the GNN's evaluation
all_preds = target_scaler.inverse_transform(all_preds_scaled)

mae = mean_absolute_error(all_true, all_preds)
mse = mean_squared_error(all_true, all_preds)
rmse = math.sqrt(mse)
r2 = r2_score(all_true, all_preds)

print("\n ECFP4 + FFN BASELINE RESULTS")
print(f"Test samples : {len(all_true)}")
print(f"MAE          : {mae:>6.2f} Å^2")
print(f"RMSE         : {rmse:>6.2f} Å^2")
print(f"R^2           : {r2:>6.4f}")

# Save predictions for later plotting / comparison with the GNN
os.makedirs("baseline_results", exist_ok=True)
pd.DataFrame({
    "y_true": all_true.flatten(),
    "y_pred": all_preds.flatten(),
}).to_csv("baseline_results/baseline_ecfp4_predictions.csv", index=False)

pd.DataFrame([{
    "model": "ECFP4 + FFN",
    "test_samples": len(all_true),
    "mae": mae,
    "rmse": rmse,
    "r2": r2,
    "fp_radius": FP_RADIUS,
    "fp_nbits": FP_NBITS,
    "hidden_dim": 512,
    "n_layers": 3,
    "dropout": 0.2,
    "seed": SEED,
}]).to_csv("baseline_results/baseline_ecfp4_metrics.csv", index=False)

print("\nPredictions and metrics saved to baseline_results/")
print(f"Total execution time: {(time.time() - start_time) / 60:.2f} minutes.")

'''
ECFP4 + FFN BASELINE RESULTS
Test samples : 6185
MAE          :   4.12 Å^2
RMSE         :   5.66 Å^2
R^2           : 0.9241

'''