

"""
baseline_ecfp4_optuna.py
Optuna hyperparameter search for the ECFP4 + FFN baseline.

Key points:
- We tune the fixed hyperparamer baseline the same way we tuned the GNN, so the comparison is fair.
- Both GNN and ecfp4 have Optuna HP.

- Same data, same scaffold-balanced split, same seed, same extra features.
- ECFP4 fingerprints are computed ONCE before the study (not per trial). That is way it will be much faster
- Search space is small and FFN-relevant: hidden_dim, n_layers, dropout, learning rate, batch size. 
  We also tune fingerprint size and radius lightly, since those are similar to graph hyperparameters for the baseline.
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
from rdkit.Chem import rdFingerprintGenerator

import optuna
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

# 3. PARSE MOLECULES AND BUILD EXTRA-FEATURE MATRIX ONCE
# We parse SMILES, compute MW, and store the RDKit Mol objects ONCE here.
# Fingerprints depend on (radius, nbits), so we recompute those inside a small
# cache keyed by those two hyperparameters, not on every trial.
print("\n3. PARSING MOLECULES (once)")

mols= []
extra_features= []
targets= []
invalid_mols= 0

for _, row in df_encoded.iterrows():
    smi= row['smiles']
    mol= Chem.MolFromSmiles(smi)
    if mol is None:
        invalid_mols+=1
        continue

    mw= Descriptors.MolWt(mol)
    if row.get('Dimer.1_Dimer', 0.0)== 1.0:
        mw= mw*2

    categorical_values= row[extra_feature_columns].values.astype(np.float32)
    extra= np.append(categorical_values, mw).astype(np.float32)

    mols.append(mol)
    extra_features.append(extra)
    targets.append(row['CCS_AVG'])

extra_features= np.stack(extra_features)
targets= np.array(targets, dtype=np.float32).reshape(-1, 1)
print(f"Valid molecules: {len(mols)}  (Discarded: {invalid_mols})")

# 4. SCAFFOLD-BALANCED SPLIT (computed ONCE, itis identical for every trial)
print("\n4. SCAFFOLD-BALANCED SPLIT (80/10/10, seed=42)")

indices_train, indices_val, indices_test = make_split_indices(
    mols, split="scaffold_balanced", sizes=(0.8, 0.1, 0.1), seed=SEED,
)
idx_train= np.array(indices_train[0], dtype=int)
idx_val = np.array(indices_val[0],   dtype=int)
idx_test= np.array(indices_test[0],  dtype=int)
print(f"Train: {len(idx_train)} | Val: {len(idx_val)} | Test: {len(idx_test)}")

# Presplit extras and targets (these don't depend on FP hyperparameters)
extra_train, extra_val, extra_test = extra_features[idx_train], extra_features[idx_val], extra_features[idx_test]
y_train, y_val, y_test = targets[idx_train], targets[idx_val], targets[idx_test]

# Fit scalers on the TRAIN split only, it prevents data leakage into val/test
extra_scaler= StandardScaler().fit(extra_train)
extra_train_s= extra_scaler.transform(extra_train).astype(np.float32)
extra_val_s = extra_scaler.transform(extra_val).astype(np.float32)
extra_test_s= extra_scaler.transform(extra_test).astype(np.float32)

target_scaler= StandardScaler().fit(y_train)
y_train_s= target_scaler.transform(y_train).astype(np.float32)
y_val_s= target_scaler.transform(y_val).astype(np.float32)
# y_test is left in original Å^2 units for the final report

# 4. FINGERPRINT CACHE
# If a trial picks the same (radius, nbits) as a previous one, reuse the (FP) FingerPrint
# matrix instead of recomputing. Big speedup when Optuna revisits combinations.
_fp_cache = {}

def get_fingerprints(radius: int, nbits: int):
    key = (radius, nbits)
    if key in _fp_cache:
        return _fp_cache[key]
    gen= rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=nbits)
    fps= np.stack([gen.GetFingerprintAsNumPy(m).astype(np.float32) for m in mols])
    _fp_cache[key] = fps
    return fps

# 5. FFN MODEL (same as baseline_ecfp4.py)
class FFNBaseline(pl.LightningModule):
    def __init__(self, input_dim, hidden_dim, n_layers, dropout, lr):
        super().__init__()
        self.save_hyperparameters()
        layers = []
        in_d = input_dim
        for _ in range(n_layers):
            layers += [nn.Linear(in_d, hidden_dim), nn.ReLU(), nn.Dropout(dropout)]
            in_d = hidden_dim
        layers += [nn.Linear(in_d, 1)]
        self.net = nn.Sequential(*layers)
        self.loss_fn = nn.MSELoss()

    def forward(self, x):
        return self.net(x)

    def training_step(self, batch, _):
        x, y = batch
        loss = self.loss_fn(self(x), y)
        self.log("train_loss", loss, prog_bar=True, on_epoch=True, on_step=False)
        return loss

    def validation_step(self, batch, _):
        x, y = batch
        loss = self.loss_fn(self(x), y)
        self.log("val_loss", loss, prog_bar=True, on_epoch=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)


class BestWeightsTracker(Callback):
    #Save weights at the best validation epoch
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


def evaluate_on_test(model, loader_test):
    #Return MAE / RMSE / R^2 in original Å^2 units.
    model.eval()
    preds_scaled, trues = [], []
    with torch.no_grad():
        for x, y in loader_test:
            preds_scaled.append(model(x).cpu().numpy())
            trues.append(y.numpy())
    preds_scaled = np.concatenate(preds_scaled, axis=0)
    trues = np.concatenate(trues, axis=0)
    preds = target_scaler.inverse_transform(preds_scaled)
    return {
        "mae": mean_absolute_error(trues, preds),
        "rmse": math.sqrt(mean_squared_error(trues, preds)),
        "r2": r2_score(trues, preds),
        "n_samples": len(trues),
    }

# 6. OPTUNA OBJECTIVE
def objective(trial):
    # Fingerprint hyperparameters (the "graph hyperparameters" of a baseline)
    # Radius 2 = ECFP4, radius 3 = ECFP6. Worth comparing both.
    fp_radius= trial.suggest_categorical("fp_radius", [2, 3])
    fp_nbits= trial.suggest_categorical("fp_nbits", [1024, 2048])

    # --- FFN hyperparameters ---
    hidden_dim = trial.suggest_categorical("hidden_dim", [256, 512, 1024])
    n_layers   = trial.suggest_int("n_layers", 1, 4)
    dropout    = trial.suggest_float("dropout", 0.0, 0.4)
    lr         = trial.suggest_float("lr", 1e-4, 5e-3, log=True)
    batch_size = trial.suggest_categorical("batch_size", [64, 128, 256])

    # Build the input matrices for this trial fingerprint choice 
    fps = get_fingerprints(fp_radius, fp_nbits)
    X_train = np.concatenate([fps[idx_train], extra_train_s], axis=1).astype(np.float32)
    X_val   = np.concatenate([fps[idx_val],   extra_val_s],   axis=1).astype(np.float32)
    X_test  = np.concatenate([fps[idx_test],  extra_test_s],  axis=1).astype(np.float32)
    input_dim = X_train.shape[1]

    def make_loader(X, y, shuffle):
        ds = TensorDataset(torch.from_numpy(X), torch.from_numpy(y))
        return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=0)

    loader_train = make_loader(X_train, y_train_s, shuffle=True)
    loader_val   = make_loader(X_val,   y_val_s,   shuffle=False)
    loader_test  = make_loader(X_test,  y_test,    shuffle=False)

    model = FFNBaseline(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        n_layers=n_layers,
        dropout=dropout,
        lr=lr,
    )

    early_stop= EarlyStopping(monitor="val_loss", patience=10, mode="min", verbose=False)
    best_weights= BestWeightsTracker(monitor="val_loss")

    trainer= pl.Trainer(
        max_epochs=150,             # high cap; early stopping ends naturally
        accelerator="cpu",
        devices=1,
        logger=False,
        enable_progress_bar=False,  # cleaner Optuna logs
        enable_checkpointing=False,
        enable_model_summary=False,
        callbacks=[early_stop, best_weights],
    )
    trainer.fit(model, loader_train, loader_val)

    if best_weights.best_state_dict is not None:
        model.load_state_dict(best_weights.best_state_dict)

    # Store test metrics on the trial for the final summary CSV
    test_metrics = evaluate_on_test(model, loader_test)
    trial.set_user_attr("test_mae", test_metrics["mae"])
    trial.set_user_attr("test_rmse", test_metrics["rmse"])
    trial.set_user_attr("test_r2", test_metrics["r2"])
    trial.set_user_attr("test_samples", test_metrics["n_samples"])

    # Free memory between trials
    del model, trainer

    return float("inf") if best_weights.best_score is None else best_weights.best_score

# 7. RUN THE STUDY
print("\n5. STARTING OPTUNA SEARCH (10 trials)")

study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=SEED))
study.optimize(objective, n_trials=10)

# 8. EXPORT RESULTS
print("\n6. EXPORTING RESULTS")
os.makedirs("baseline_results", exist_ok=True)

df_results = study.trials_dataframe()
df_results.to_csv("baseline_results/optuna_results_baseline_ecfp4.csv", index=False)

best_trial = study.best_trial
print(f"\nBEST TRIAL: #{best_trial.number}")
print(f"Best validation loss: {best_trial.value:.4f}")
print("Optimal hyperparameters:")
for k, v in best_trial.params.items():
    print(f"  - {k}: {v}")

print("\nTEST SET RESULTS FOR THE BEST TRIAL (Å^2 units):")
print(f"Samples : {best_trial.user_attrs.get('test_samples', 0)}")
print(f"MAE     : {best_trial.user_attrs.get('test_mae', float('nan')):>6.2f} Å^2")
print(f"RMSE    : {best_trial.user_attrs.get('test_rmse', float('nan')):>6.2f} Å^2")
print(f"R^2      : {best_trial.user_attrs.get('test_r2', float('nan')):>6.4f}")

# Append the best result as a single-row summary CSV, easy to plug into the report
pd.DataFrame([{
    "model": "ECFP4 + FFN (Optuna)",
    "best_trial": best_trial.number,
    "val_loss": best_trial.value,
    "test_samples": best_trial.user_attrs.get("test_samples", 0),
    "mae": best_trial.user_attrs.get("test_mae", float("nan")),
    "rmse": best_trial.user_attrs.get("test_rmse", float("nan")),
    "r2": best_trial.user_attrs.get("test_r2", float("nan")),
    **best_trial.params,
    "seed": SEED,
}]).to_csv("baseline_results/baseline_ecfp4_optuna_best.csv", index=False)

print(f"\nTotal execution time: {(time.time() - start_time) / 60:.2f} minutes.")


'''
BEST TRIAL: #4
Best validation loss: 0.0710
Optimal hyperparameters:
  - fp_radius: 2
  - fp_nbits: 2048
  - hidden_dim: 512
  - n_layers: 2
  - dropout: 0.15547091587579281
  - lr: 0.00028907721743726757
  - batch_size: 64

TEST SET RESULTS FOR THE BEST TRIAL (Å^2 units):
Samples : 6185
MAE     :   4.08 Å^2
RMSE    :   5.67 Å^2
R^2      : 0.9239
Total execution time: 15.55 minutes.
'''

