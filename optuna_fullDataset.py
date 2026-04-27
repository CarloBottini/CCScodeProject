
#The following code cannot be run in a simple basic computer
#This code is designed to run on a more powerful machine with GPU capabilities


import pandas as pd
import numpy as np
import torch
import lightning.pytorch as pl
from chemprop.data import MoleculeDatapoint, MoleculeDataset, build_dataloader, make_split_indices
from chemprop.nn import BondMessagePassing, SumAggregation, RegressionFFN, UnscaleTransform
from chemprop.models import MPNN

from rdkit import Chem
from rdkit.Chem import Descriptors
import optuna
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
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





print("\n 4. DATA SPLIT AND SCALING (80% TRAIN / 20% VAL)")
#Split for scafold
mols= [d.mol for d in datapoints]
#Optuna only need train and val sets, but I will create the test set as well for future use.
indices_train, indices_val, indices_test= make_split_indices(
    mols,
    #split="random_with_repeated_smiles",
    #but it is better this next one
    split="scaffold_balanced", #This split method ensures that molecules with similar scaffolds are grouped together, which is crucial for evaluating the model's ability to generalize to unseen chemical structures
    sizes=(0.8, 0.2, 0.0), #Since we are not going to use the test set during Optuna
    seed=42
)


dataset_train= MoleculeDataset([datapoints[int(i)] for i in indices_train[0]])
dataset_val= MoleculeDataset([datapoints[int(i)] for i in indices_val[0]])

#dataset_test= MoleculeDataset([datapoints[int(i)] for i in indices_test[0]])


#Scaling the target values (CCS)
target_scaler= dataset_train.normalize_targets()
dataset_val.normalize_targets(target_scaler)
#dataset_test.normalize_targets(target_scaler) 

#Normalization of X_d (Adducts, Dimers and Molecular Weight)
x_d_scaler = dataset_train.normalize_inputs("X_d")
dataset_val.normalize_inputs("X_d", x_d_scaler)
#dataset_test.normalize_inputs("X_d", x_d_scaler)
#X_d_transform = ScaleTransform.from_standard_scaler(x_d_scaler) #not going to use this

#Grouping 32 molecules per batch each time
#The batch size can be upgraded to 64 or 128
num_dataloader_workers = 8
loader_train= build_dataloader(dataset_train, batch_size=128, shuffle=True, num_workers=num_dataloader_workers) 
loader_val= build_dataloader(dataset_val, batch_size=128, shuffle=False, num_workers=num_dataloader_workers)
#loader_test= build_dataloader(dataset_test, batch_size=128, shuffle=False, num_workers=num_dataloader_workers)


num_extra_features = len(extra_feature_columns) + 1


print("\n2. STARTING OPTUNA HYPERPARAMETER SEARCH")

def objective(trial):
    depth= trial.suggest_int("depth", 2, 6) 
    message_hidden_dim= trial.suggest_int("message_hidden_dim", 256, 2048, log=True) 
    ffn_hidden_dim= trial.suggest_int("ffn_hidden_dim", 256, 4096, log=True)
    ffn_num_layers= trial.suggest_int("ffn_num_layers", 1, 6)
    dropout= trial.suggest_float("dropout", 0.0, 0.4) #Helps prevent overfitting

    #Builidng the MPNN model with the current set of hyperparameters
    mp= BondMessagePassing(d_h=message_hidden_dim, depth=depth)
    agg= SumAggregation()
    total_input_dim= mp.output_dim +num_extra_features

    ffn = RegressionFFN(
        input_dim=total_input_dim, 
        hidden_dim=ffn_hidden_dim,
        n_layers=ffn_num_layers,
        dropout=dropout,
        output_transform=UnscaleTransform.from_standard_scaler(target_scaler)
    )
    
    model= MPNN(mp,agg,ffn)

    #Training the model and extracting the best validation loss for this trial
    #Optuna will use this value to compare against other trials and determine which hyperparameters are performing better.
    #Create a temporary directory for this specific trial's weights
    checkpoint_dir= os.path.join(os.getcwd(), "optuna_temp", f"trial_{trial.number}")
    checkpoint_callback= ModelCheckpoint(dirpath=checkpoint_dir, monitor="val_loss", mode="min", save_top_k=1)
    early_stop_callback= EarlyStopping(
        monitor="val_loss", 
        patience=7, 
        mode="min",
        verbose=True)


    trainer = pl.Trainer(
        max_epochs=80, #HIgh for full optimization
        accelerator="auto",
        devices=1,
        logger=False, #Disabled to avoid clutter
        enable_progress_bar=False, #Disabled to keep the console clean for Optuna logs
        enable_model_summary=False,
        callbacks=[checkpoint_callback, early_stop_callback]
    )
    
    trainer.fit(model, loader_train, loader_val)
    
    best_score= checkpoint_callback.best_model_score
    
    #Clean up memory to prevent RAM overflow during 10 trials
    del model,trainer
    
    #Return the validation loss. If a trial fails (None), return infinity.
    return float("inf") if best_score is None else best_score.item()

#Create the study aiming to 'minimize' the validation loss
study =optuna.create_study(direction="minimize")

#Execute 10 trials
study.optimize(objective, n_trials=100)

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

end_time=time.time()
print(f"\nTotal execution time: {(end_time - start_time) / 60:.2f} minutes.")


'''
Results from Optuna search with full dataset



'''

