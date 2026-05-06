
#The following code cannot be run in a simple basic computer
#This code is designed to run on a more powerful machine with GPU capabilities


import pandas as pd
import numpy as np
import torch
import lightning.pytorch as pl
from chemprop.data import MoleculeDatapoint, MoleculeDataset, build_dataloader, make_split_indices
from chemprop.nn import (
    BondMessagePassing,
    MeanAggregation,
    NormAggregation,
    SumAggregation,
    RegressionFFN,
    ScaleTransform,
    UnscaleTransform,
)
from chemprop.models import MPNN

from rdkit import Chem
from rdkit.Chem import Descriptors
import optuna
from lightning.pytorch.callbacks import Callback, EarlyStopping
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import math
import copy
import os
import warnings
import time

#Disable minor warnings to keep the Optuna console clean
warnings.filterwarnings("ignore")

start_time=time.time()
pl.seed_everything(42, workers=True)

print("1. DATA PREPARATION FULL DATASET") #It will only run once 

csv_path= "data/METLIN_IMS_dimers_rmTM.csv" 
df= pd.read_csv(csv_path)
df= df.dropna(subset=['smiles', 'CCS_AVG', 'Adduct', 'Dimer.1'])

print(f"Rows remaining after cleaning missing values: {df.shape[0]}")


print("\n2. ONE HOT ENCODING ")
#get_dummies will create new columns for each unique value in 'Adduct' and 'Dimer.1', with 1.0 indicating the presence of that category and 0.0 indicating its absence.
df_encoded= pd.get_dummies(df, columns=['Adduct', 'Dimer.1'], dtype=float)
extra_feature_columns= [col for col in df_encoded.columns if col.startswith('Adduct_') or col.startswith('Dimer.1_')]
print(f"Extra features created for x_d: {extra_feature_columns}")




print("\n 3. BUILDING DATAPOINTS")
datapoints= []
valid_mols= 0
invalid_mols= 0
for index, row in df_encoded.iterrows():
    smi= row['smiles']
    ccs_target= row['CCS_AVG']

    #Calculation of the Molecular Weight using RDKit
    mol= Chem.MolFromSmiles(smi)
    if mol is None:
        invalid_mols +=1
        continue #Skip the corrupted smiles strings

    mw= Descriptors.MolWt(mol)

    #Dimer logic: If the molecule is a dimer, we will double the Molecular Weight to reflect that it is two molecules together.
    if row.get('Dimer.1_Dimer', 0.0) == 1.0: #If the Dimer.1_Dimer column is 1.0, it means this molecule is a dimer.
        mw= mw * 2


    categorical_values =row[extra_feature_columns].values.astype(np.float32)
    #It is needed to combine the categorical features with MW in the same array x_d
    x_d_values= np.append(categorical_values, mw).astype(np.float32)
    

    datapoints.append(MoleculeDatapoint.from_smi(smi, y=np.array([ccs_target]), x_d=x_d_values))
    valid_mols +=1

print(f"Successfully packaged {valid_mols} MoleculeDatapoints. (Discarded: {invalid_mols})")





print("\n 4. DATA SPLIT AND SCALING (80% TRAIN / 10% VAL / 10% TEST)")
#Split for scafold
mols= [d.mol for d in datapoints]
indices_train, indices_val, indices_test= make_split_indices(
    mols,
    #split="random_with_repeated_smiles",
    #but it is better this next one
    split="scaffold_balanced", #This split method ensures that molecules with similar scaffolds are grouped together, which is crucial for evaluating the model's ability to generalize to unseen chemical structures
    sizes=(0.8, 0.1, 0.1),
    seed=42
)


dataset_train= MoleculeDataset([datapoints[int(i)] for i in indices_train[0]])
dataset_val= MoleculeDataset([datapoints[int(i)] for i in indices_val[0]])
dataset_test= MoleculeDataset([datapoints[int(i)] for i in indices_test[0]])


#Scaling the target values (CCS)
target_scaler= dataset_train.normalize_targets()
dataset_val.normalize_targets(target_scaler)

#Normalization of X_d (Adducts, Dimers and Molecular Weight)
x_d_scaler = dataset_train.normalize_inputs("X_d")
dataset_val.normalize_inputs("X_d", x_d_scaler)
X_d_transform = ScaleTransform.from_standard_scaler(x_d_scaler)

#Grouping 32 molecules per batch each time
#The batch size can be upgraded to 64 or 128
num_dataloader_workers = 8
loader_train= build_dataloader(dataset_train, batch_size=128, shuffle=True, num_workers=num_dataloader_workers) 
loader_val= build_dataloader(dataset_val, batch_size=128, shuffle=False, num_workers=num_dataloader_workers)
loader_test= build_dataloader(dataset_test, batch_size=128, shuffle=False, num_workers=num_dataloader_workers)


num_extra_features = len(extra_feature_columns) + 1
aggregation_map = {
    "sum": SumAggregation,
    "mean": MeanAggregation,
    "norm": NormAggregation,
}


print("\n2. STARTING OPTUNA HYPERPARAMETER SEARCH")

class BestWeightsTracker(Callback):
    def __init__(self, monitor="val_loss", mode="min"):
        self.monitor = monitor
        self.mode = mode
        self.best_score = None
        self.best_state_dict = None

    def _is_better(self, current_score):
        if self.best_score is None:
            return True
        if self.mode == "min":
            return current_score < self.best_score
        return current_score > self.best_score

    def on_validation_end(self, trainer, pl_module):
        if trainer.sanity_checking:
            return

        current_metric = trainer.callback_metrics.get(self.monitor)
        if current_metric is None:
            return

        current_score = float(current_metric.detach().cpu().item())
        if self._is_better(current_score):
            self.best_score = current_score
            self.best_state_dict = {
                key: value.detach().cpu().clone()
                for key, value in pl_module.state_dict().items()
            }

def evaluate_on_test(model):
    model.eval()
    all_predictions = []
    all_real_values = []

    with torch.no_grad():
        for test_batch in loader_test:
            pred = model(test_batch.bmg, X_d=test_batch.X_d)

            all_predictions.extend(pred.flatten().tolist())
            all_real_values.extend(test_batch.Y.flatten().tolist())

    mae = mean_absolute_error(all_real_values, all_predictions)
    mse = mean_squared_error(all_real_values, all_predictions)
    rmse = math.sqrt(mse)
    r2 = r2_score(all_real_values, all_predictions)

    return {
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
        "n_samples": len(all_real_values),
    }

def objective(trial):
    depth= trial.suggest_int("depth", 2, 6) 
    message_hidden_dim= trial.suggest_int("message_hidden_dim", 256, 2048, log=True) 
    ffn_hidden_dim= trial.suggest_int("ffn_hidden_dim", 256, 4096, log=True)
    ffn_num_layers= trial.suggest_int("ffn_num_layers", 1, 6)
    dropout= trial.suggest_float("dropout", 0.0, 0.4) #Helps prevent overfitting
    aggregation_name = trial.suggest_categorical("aggregation", ["sum", "mean", "norm"])
    batch_norm = trial.suggest_categorical("batch_norm", [False, True])
    warmup_epochs = trial.suggest_int("warmup_epochs", 1, 6)
    max_lr = trial.suggest_float("max_lr", 5e-4, 5e-3, log=True)
    init_lr_ratio = trial.suggest_float("init_lr_ratio", 0.05, 1.0)
    final_lr_ratio = trial.suggest_float("final_lr_ratio", 0.01, 1.0, log=True)

    init_lr = max_lr * init_lr_ratio
    final_lr = max_lr * final_lr_ratio

    #Builidng the MPNN model with the current set of hyperparameters
    mp= BondMessagePassing(d_h=message_hidden_dim, depth=depth)
    agg= aggregation_map[aggregation_name]()
    total_input_dim= mp.output_dim +num_extra_features

    ffn = RegressionFFN(
        input_dim=total_input_dim, 
        hidden_dim=ffn_hidden_dim,
        n_layers=ffn_num_layers,
        dropout=dropout,
        output_transform=UnscaleTransform.from_standard_scaler(target_scaler)
    )
    model= MPNN(
        mp,
        agg,
        ffn,
        batch_norm=batch_norm,
        warmup_epochs=warmup_epochs,
        init_lr=init_lr,
        max_lr=max_lr,
        final_lr=final_lr,
        X_d_transform=X_d_transform,
    )

    #Training the model and extracting the best validation loss for this trial
    #Optuna will use this value to compare against other trials and determine which hyperparameters are performing better.
    early_stop_callback= EarlyStopping(
        monitor="val_loss", 
        patience=7, 
        mode="min",
        verbose=True)
    best_weights_callback = BestWeightsTracker(monitor="val_loss", mode="min")


    trainer = pl.Trainer(
        max_epochs=80, #HIgh for full optimization
        accelerator="auto",
        devices=1,
        logger=False, #Disabled to avoid clutter
        enable_progress_bar=True,
        enable_checkpointing=False,
        enable_model_summary=False,
        callbacks=[early_stop_callback, best_weights_callback]
    )
    
    trainer.fit(model, loader_train, loader_val)
    
    if best_weights_callback.best_state_dict is not None:
        model.load_state_dict(best_weights_callback.best_state_dict)

    test_metrics = evaluate_on_test(model)
    trial.set_user_attr("test_mae", test_metrics["mae"])
    trial.set_user_attr("test_rmse", test_metrics["rmse"])
    trial.set_user_attr("test_r2", test_metrics["r2"])
    trial.set_user_attr("test_samples", test_metrics["n_samples"])
    
    #Clean up memory to prevent RAM overflow during 10 trials
    del model, trainer
    
    #Return the validation loss. If a trial fails (None), return infinity.
    return float("inf") if best_weights_callback.best_score is None else best_weights_callback.best_score

#Create the study aiming to 'minimize' the validation loss
study =optuna.create_study(direction="minimize")

#Execute 10 trials
study.optimize(objective, n_trials=2)

print("\n3. EXPORTING RESULTS AND BEST HYPERPARAMETERS")
#changing the file name so it do not overwrite the previous one with 1000 rows
df_results =study.trials_dataframe()
df_results.to_csv("optuna_results_fullDataset.csv", index=False)
print("Search complete. History saved to 'optuna_results_fullDataset.csv'.")

best_trial=study.best_trial
print(f"\nTHE BEST MODEL IS TRIAL #{best_trial.number}")
print(f"Best Validation Loss: {best_trial.value:.4f}")
print("Optimal Parameters Found:")
for key, value in best_trial.params.items():
    print(f"  - {key}: {value}")

print("\nTEST SET RESULTS FOR THE BEST TRIAL")
print(f"Tested on exactly {best_trial.user_attrs.get('test_samples', 0)} unseen molecules (10% Test Set):")
print(f"MAE       : {best_trial.user_attrs.get('test_mae', float('nan')):>6.2f} Å2")
print(f"RMSE      : {best_trial.user_attrs.get('test_rmse', float('nan')):>6.2f} Å2")
print(f"R-squared : {best_trial.user_attrs.get('test_r2', float('nan')):>6.4f}")

end_time=time.time()
print(f"\nTotal execution time: {(end_time - start_time) / 60:.2f} minutes.")


'''
Results from Optuna search with full dataset



'''

