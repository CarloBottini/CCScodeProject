
# Focused Optuna search (10 trials) on the FULL METLIN dataset.
# Only 3 hyperparameters are searched: depth, message_hidden_dim, ffn_hidden_dim.
# These were the bimodal ones in the original 100-trial run, so we want to resolve which mode (low vs high) generalizes better.
# All other hyperparameters are FIXED to the robust-zone values identified from the analysis of the original 100-trial Optuna study.


import os
import math
import time
import warnings

import numpy as np
import pandas as pd
import torch
import lightning.pytorch as pl
from lightning.pytorch.callbacks import Callback, EarlyStopping

from rdkit import Chem
from rdkit.Chem import Descriptors

import optuna
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from chemprop.data import (
    MoleculeDatapoint,
    MoleculeDataset,
    build_dataloader,
    make_split_indices,
)
from chemprop.nn import (
    BondMessagePassing,
    MeanAggregation,
    RegressionFFN,
    ScaleTransform,
    UnscaleTransform,
)
from chemprop.models import MPNN


warnings.filterwarnings("ignore")
pl.seed_everything(42, workers=True)
start_time = time.time()



CSV_RESULTS_DIR = "CSVresults"
os.makedirs(CSV_RESULTS_DIR, exist_ok=True)
RESULTS_CSV_PATH = os.path.join(CSV_RESULTS_DIR, "optuna_results_focused.csv")


# CLASS 1 - Data preparation pipeline
# Wrapping data prep (scalers, datasets, extra-feature columns)
class CCSDataPipeline:
    """Loads METLIN-CCS, builds MoleculeDatapoints, splits, scales."""

    def __init__(self, csv_path, batch_size=128, num_workers=0, seed=42):
        self.csv_path= csv_path
        self.batch_size= batch_size
        self.num_workers= num_workers   # 0 on CPU-only machines
        self.seed= seed

        self.extra_feature_columns= None
        self.num_extra_features= None
        self.target_scaler= None
        self.x_d_scaler= None
        self.X_d_transform= None
        self.loader_train= None
        self.loader_val= None
        self.loader_test= None

    def run(self):
        # Load + drop rows missing any key field
        df= pd.read_csv(self.csv_path)
        df= df.dropna(subset=["smiles", "CCS_AVG", "Adduct", "Dimer.1"])
        print(f"Rows after cleaning: {df.shape[0]}")

        # One-hot encode Adduct and Dimer.1 (categorical -> numeric flags)
        df_encoded = pd.get_dummies(df, columns=["Adduct", "Dimer.1"], dtype=float)
        self.extra_feature_columns = [
            c for c in df_encoded.columns
            if c.startswith("Adduct_") or c.startswith("Dimer.1_")
        ]

        # Build datapoints (skip invalid SMILES, double MW for dimers)
        datapoints= []
        valid, invalid = 0, 0
        for _, row in df_encoded.iterrows():
            smi= row["smiles"]
            mol= Chem.MolFromSmiles(smi)
            if mol is None:
                invalid += 1
                continue
            mw= Descriptors.MolWt(mol)
            if row.get("Dimer.1_Dimer", 0.0) == 1.0:
                mw *= 2
            cat_vals= row[self.extra_feature_columns].values.astype(np.float32)
            x_d= np.append(cat_vals, mw).astype(np.float32)
            datapoints.append(
                MoleculeDatapoint.from_smi(
                    smi, y=np.array([row["CCS_AVG"]]), x_d=x_d
                )
            )
            valid += 1
        print(f"Valid datapoints: {valid} | Discarded: {invalid}")

        # Scaffold-balanced split (80/10/10). same seed as the original run
        mols = [d.mol for d in datapoints]
        idx_tr, idx_val, idx_te = make_split_indices(
            mols, split="scaffold_balanced",
            sizes=(0.8, 0.1, 0.1), seed=self.seed,
        )
        ds_tr= MoleculeDataset([datapoints[int(i)] for i in idx_tr[0]])
        ds_val= MoleculeDataset([datapoints[int(i)] for i in idx_val[0]])
        ds_te= MoleculeDataset([datapoints[int(i)] for i in idx_te[0]])

        # Fit scalers on TRAIN only, then apply to VAL. Test stays raw.
        self.target_scaler= ds_tr.normalize_targets()
        ds_val.normalize_targets(self.target_scaler)
        self.x_d_scaler= ds_tr.normalize_inputs("X_d")
        ds_val.normalize_inputs("X_d", self.x_d_scaler)
        self.X_d_transform= ScaleTransform.from_standard_scaler(self.x_d_scaler)

        # Dataloaders
        self.loader_train= build_dataloader(
            ds_tr, batch_size=self.batch_size, shuffle=True,
            num_workers=self.num_workers,
        )
        self.loader_val= build_dataloader(
            ds_val, batch_size=self.batch_size, shuffle=False,
            num_workers=self.num_workers,
        )
        self.loader_test= build_dataloader(
            ds_te, batch_size=self.batch_size, shuffle=False,
            num_workers=self.num_workers,
        )

        self.num_extra_features= len(self.extra_feature_columns) + 1  # +1 for MW
        return self


# CLASS 2 - Lightning callback that snapshots the best weights.
# EarlyStopping stops training, but Lightning leaves the model with the weights from the LAST epoch, not the best one. This callback fixes that.
class BestWeightsTracker(Callback):
    def __init__(self, monitor="val_loss", mode="min"):
        self.monitor= monitor
        self.mode= mode
        self.best_score= None
        self.best_state_dict= None

    def _is_better(self, score):
        if self.best_score is None:
            return True
        return score < self.best_score if self.mode == "min" else score > self.best_score

    def on_validation_end(self, trainer, pl_module):
        if trainer.sanity_checking:
            return
        metric= trainer.callback_metrics.get(self.monitor)
        if metric is None:
            return
        score= float(metric.detach().cpu().item())
        if self._is_better(score):
            self.best_score = score
            self.best_state_dict = {
                k: v.detach().cpu().clone() for k, v in pl_module.state_dict().items()
            }


# CLASS 3 - Focused Optuna study
# Only depth, message_hidden_dim, ffn_hidden_dim are searched. Everything else is fixed to the robust zone defaults.
class FocusedOptunaStudy:

    # Fixed hyperparameters from the analysis of the original 100 trial run
    FIXED= {
        "dropout": 0.25,
        "aggregation": "mean",
        "batch_norm": False,
        "warmup_epochs": 2,        
        "max_lr": 0.0025,
        "init_lr_ratio": 0.27,
        "final_lr_ratio": 0.25,
        "ffn_num_layers": 6,       # robust zone (5-6), prioritising accuracy, rather than speed 2
    }

    def __init__(self, pipeline: CCSDataPipeline, n_trials=10,
                 max_epochs=30, patience=5, results_csv=RESULTS_CSV_PATH):
        self.pipe = pipeline
        self.n_trials= n_trials
        self.max_epochs= max_epochs
        self.patience= patience
        self.results_csv= results_csv

    def _build_model(self, depth, message_hidden_dim, ffn_hidden_dim):
        f = self.FIXED
        mp = BondMessagePassing(d_h=message_hidden_dim, depth=depth)
        agg = MeanAggregation()  # fixed= mean
        total_input_dim= mp.output_dim + self.pipe.num_extra_features
        ffn = RegressionFFN(
            input_dim=total_input_dim,
            hidden_dim=ffn_hidden_dim,
            n_layers=f["ffn_num_layers"],
            dropout=f["dropout"],
            output_transform=UnscaleTransform.from_standard_scaler(self.pipe.target_scaler),
        )
        return MPNN(
            mp, agg, ffn,
            batch_norm=f["batch_norm"],
            warmup_epochs=f["warmup_epochs"],
            init_lr=f["max_lr"] * f["init_lr_ratio"],
            max_lr=f["max_lr"],
            final_lr=f["max_lr"] * f["final_lr_ratio"],
            X_d_transform=self.pipe.X_d_transform,
        )

    def _evaluate_test(self, model):
        model.eval()
        preds, reals = [], []
        with torch.no_grad():
            for batch in self.pipe.loader_test:
                p= model(batch.bmg, X_d=batch.X_d)
                preds.extend(p.flatten().tolist())
                reals.extend(batch.Y.flatten().tolist())
        return {
            "mae": mean_absolute_error(reals, preds),
            "rmse": math.sqrt(mean_squared_error(reals, preds)),
            "r2": r2_score(reals, preds),
            "n_samples": len(reals),
        }

    def _objective(self, trial):
        # Only THESE three are searched
        depth = trial.suggest_int("depth", 2, 6)
        message_hidden_dim = trial.suggest_int("message_hidden_dim", 256, 2048, log=True)
        ffn_hidden_dim = trial.suggest_int("ffn_hidden_dim", 256, 4096, log=True)

        model= self._build_model(depth, message_hidden_dim, ffn_hidden_dim)

        early_stop= EarlyStopping(monitor="val_loss", patience=self.patience,
                                   mode="min", verbose=True)
        best_w= BestWeightsTracker(monitor="val_loss", mode="min")

        trainer= pl.Trainer(
            max_epochs=self.max_epochs,
            accelerator="auto",
            devices=1,
            logger=False,
            enable_progress_bar=True,
            enable_checkpointing=False,
            enable_model_summary=False,
            callbacks=[early_stop, best_w],
        )
        trainer.fit(model, self.pipe.loader_train, self.pipe.loader_val)

        # Restore the best weights before testing (otherwise we test the LAST epoch)
        if best_w.best_state_dict is not None:
            model.load_state_dict(best_w.best_state_dict)

        # Persist test metrics to the trial so they appear in the CSV
        m = self._evaluate_test(model)
        trial.set_user_attr("test_mae", m["mae"])
        trial.set_user_attr("test_rmse", m["rmse"])
        trial.set_user_attr("test_r2", m["r2"])
        trial.set_user_attr("test_samples", m["n_samples"])

        del model, trainer  # free RAM between trials
        return float("inf") if best_w.best_score is None else best_w.best_score

    def run(self):
        study = optuna.create_study(direction="minimize")
        study.optimize(self._objective, n_trials=self.n_trials)

        study.trials_dataframe().to_csv(self.results_csv, index=False)
        print(f"\nResults saved to '{self.results_csv}'.")

        best = study.best_trial
        print(f"\nBEST TRIAL: #{best.number} | Val Loss: {best.value:.4f}")
        for k, v in best.params.items():
            print(f"  {k}: {v}")
        print("\nTEST SET RESULTS (best trial):")
        print(f"  MAE  : {best.user_attrs.get('test_mae', float('nan')):.2f} A^2")
        print(f"  RMSE : {best.user_attrs.get('test_rmse', float('nan')):.2f} A^2")
        print(f"  R^2  : {best.user_attrs.get('test_r2', float('nan')):.4f}")
        return study


# Main
if __name__ == "__main__":
    print("1. DATA PREPARATION (FULL DATASET)")
    pipeline = CCSDataPipeline(
        csv_path="data/METLIN_IMS_dimers_rmTM.csv",
        batch_size=128,
        num_workers=0,    # 0 on CPU-only machines (avoid hangs / overhead)
        seed=42,
    ).run()

    print("\n2. FOCUSED OPTUNA SEARCH (10 trials, 30 epochs)")
    study = FocusedOptunaStudy(
        pipeline=pipeline,
        n_trials=10,
        max_epochs=30,
        patience=5,
    ).run()

    print(f"\nTotal time: {(time.time() - start_time) / 60:.2f} min")



'''
Results saved to 'CSVresults\optuna_results_focused.csv'.

BEST TRIAL: #0 | Val Loss: 0.0691
  depth: 6
  message_hidden_dim: 682
  ffn_hidden_dim: 2218

TEST SET RESULTS (best trial):
  MAE  : 4.06 A^2
  RMSE : 5.53 A^2
  R^2  : 0.9276

Total time: 1321.18 min

'''


