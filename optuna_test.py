
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

print("1. DATA PREPARATION ") #It will only run once 

#Load only 1000 rows for this local testing phase
csv_path= "data/METLIN_IMS_dimers_rmTM.csv" 
df= pd.read_csv(csv_path, nrows=1000)
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
loader_train= build_dataloader(dataset_train, batch_size=32, shuffle=True, num_workers=0) 
loader_val= build_dataloader(dataset_val, batch_size=32, shuffle=False, num_workers=0)
#loader_test= build_dataloader(dataset_test, batch_size=32, shuffle=False, num_workers=0)


num_extra_features = len(extra_feature_columns) + 1


print("\n2. STARTING OPTUNA HYPERPARAMETER SEARCH")

def objective(trial):
    #The hyperparameters we want to optimize. 
    #We will test different values for the depth of the message passing, the hidden dimension of the message passing,
    #the hidden dimension and number of layers of the FFN, and the dropout rate.
    #Controls how many times messages are passed between atoms. Higher depth allows for capturing more complex interactions but can lead to overfitting and increased training time
    depth= trial.suggest_int("depth", 2, 6) 
    #The size of the hidden representations in the message passing layers. Larger dimensions can capture more complex patterns but also increase the risk of overfitting and computational cost
    message_hidden_dim= trial.suggest_int("message_hidden_dim", 256, 2048, log=True) 
    #The hidden dimension of the feedforwardnetwork FFN that processes the aggregated messages. Larger dimensions can capture more complex relationships but also increase the risk of overfitting and computational cost
    ffn_hidden_dim= trial.suggest_int("ffn_hidden_dim", 256, 4096, log=True)
    #Number of layers in the FFN. More layers can model more complex functions but also increase the risk of overfitting and training time
    ffn_num_layers= trial.suggest_int("ffn_num_layers", 1, 4)
    #The dropout rate applied to the FFN layers. It is a regularization technique that randomly sets a fraction of the input units to 0 during training, 
    #which helps prevent overfitting. A value of 0.0 means no dropout, while higher values increase the regularization effect but can also make training more difficult if set too high.
    dropout= trial.suggest_float("dropout", 0.0, 0.4) # Helps prevent overfitting

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
        max_epochs=30, #Kept low for local testing
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
study.optimize(objective, n_trials=10)

print("\n3. EXPORTING RESULTS AND BEST HYPERPARAMETERS")
#Convert Optuna's internal database into a clean Pandas DataFrame
df_results =study.trials_dataframe()
df_results.to_csv("optuna_results_test.csv", index=False)
print("Search complete. History saved to 'optuna_results_test.csv'.")

best_trial=study.best_trial
print(f"\nTHE BEST MODEL IS TRIAL #{best_trial.number}")
print(f"Best Validation Loss: {best_trial.value:.4f}")
print("Optimal Parameters Found:")
for key, value in best_trial.params.items():
    print(f"  - {key}: {value}")

end_time=time.time()
print(f"\nTotal execution time: {(end_time - start_time) / 60:.2f} minutes.")



'''
Seed set to 42
1. DATA PREPARATION 
Rows remaining after cleaning missing values: 1000

2. ONE HOT ENCODING 
Extra features created for x_d: ['Adduct_[M+H]', 'Dimer.1_Dimer', 'Dimer.1_Monomer']

 3. BUILDING DATAPOINTS
Successfully packaged 1000 MoleculeDatapoints. (Discarded: 0)

 4. DATA SPLIT AND SCALING (80% TRAIN / 20% VAL)
The return type of make_split_indices has changed in v2.1 - see help(make_split_indices)

2. STARTING OPTUNA HYPERPARAMETER SEARCH
[I 2026-04-27 00:45:52,545] A new study created in memory with name: no-name-526c864a-a07d-4a9a-abbb-d1fb8c7803a8
GPU available: False, used: False
TPU available: False, using: 0 TPU cores
💡 Tip: For seamless cloud logging and experiment tracking, try installing [litlogger](https://pypi.org/project/litlogger/) to enable LitLogger, which logs metrics and artifacts automatically to the Lightning Experiments platform.
Loading `train_dataloader` to estimate number of stepping batches.
The 'train_dataloader' does not have many workers which may be a bottleneck. Consider increasing the value of the `num_workers` argument` to `num_workers=7` in the `DataLoader` to improve performance.
The 'val_dataloader' does not have many workers which may be a bottleneck. Consider increasing the value of the `num_workers` argument` to `num_workers=7` in the `DataLoader` to improve performance.
Metric val_loss improved. New best score: 0.618
Metric val_loss improved by 0.462 >= min_delta = 0.0. New best score: 0.157
Metric val_loss improved by 0.107 >= min_delta = 0.0. New best score: 0.049
Metric val_loss improved by 0.004 >= min_delta = 0.0. New best score: 0.045
Metric val_loss improved by 0.011 >= min_delta = 0.0. New best score: 0.035
Metric val_loss improved by 0.002 >= min_delta = 0.0. New best score: 0.033
Metric val_loss improved by 0.002 >= min_delta = 0.0. New best score: 0.030
Metric val_loss improved by 0.000 >= min_delta = 0.0. New best score: 0.030
Metric val_loss improved by 0.001 >= min_delta = 0.0. New best score: 0.029
Metric val_loss improved by 0.000 >= min_delta = 0.0. New best score: 0.029
Metric val_loss improved by 0.002 >= min_delta = 0.0. New best score: 0.027
Monitored metric val_loss did not improve in the last 7 records. Best score: 0.027. Signaling Trainer to stop.
[I 2026-04-27 00:53:28,930] Trial 0 finished with value: 0.026820283383131027 and parameters: {'depth': 6, 'message_hidden_dim': 964, 'ffn_hidden_dim': 1914, 'ffn_num_layers': 2, 'dropout': 0.29502734031307476}. Best is trial 0 with value: 0.026820283383131027.
GPU available: False, used: False
TPU available: False, using: 0 TPU cores
💡 Tip: For seamless cloud logging and experiment tracking, try installing [litlogger](https://pypi.org/project/litlogger/) to enable LitLogger, which logs metrics and artifacts automatically to the Lightning Experiments platform.
Loading `train_dataloader` to estimate number of stepping batches.
Metric val_loss improved. New best score: 0.448
Metric val_loss improved by 0.321 >= min_delta = 0.0. New best score: 0.128
Metric val_loss improved by 0.038 >= min_delta = 0.0. New best score: 0.090
Metric val_loss improved by 0.023 >= min_delta = 0.0. New best score: 0.067
Metric val_loss improved by 0.009 >= min_delta = 0.0. New best score: 0.058
Metric val_loss improved by 0.017 >= min_delta = 0.0. New best score: 0.040
Metric val_loss improved by 0.007 >= min_delta = 0.0. New best score: 0.034
Metric val_loss improved by 0.003 >= min_delta = 0.0. New best score: 0.031
Metric val_loss improved by 0.005 >= min_delta = 0.0. New best score: 0.026
Metric val_loss improved by 0.001 >= min_delta = 0.0. New best score: 0.024
Metric val_loss improved by 0.001 >= min_delta = 0.0. New best score: 0.023
Metric val_loss improved by 0.001 >= min_delta = 0.0. New best score: 0.022
Monitored metric val_loss did not improve in the last 7 records. Best score: 0.022. Signaling Trainer to stop.
[I 2026-04-27 01:04:18,624] Trial 1 finished with value: 0.022328566759824753 and parameters: {'depth': 6, 'message_hidden_dim': 1135, 'ffn_hidden_dim': 3508, 'ffn_num_layers': 1, 'dropout': 0.0005514392392346857}. Best is trial 1 with value: 0.022328566759824753.
GPU available: False, used: False
TPU available: False, using: 0 TPU cores
💡 Tip: For seamless cloud logging and experiment tracking, try installing [litlogger](https://pypi.org/project/litlogger/) to enable LitLogger, which logs metrics and artifacts automatically to the Lightning Experiments platform.
Loading `train_dataloader` to estimate number of stepping batches.
Metric val_loss improved. New best score: 0.642
Metric val_loss improved by 0.059 >= min_delta = 0.0. New best score: 0.583
Metric val_loss improved by 0.399 >= min_delta = 0.0. New best score: 0.185
Metric val_loss improved by 0.007 >= min_delta = 0.0. New best score: 0.178
Metric val_loss improved by 0.120 >= min_delta = 0.0. New best score: 0.058
Metric val_loss improved by 0.019 >= min_delta = 0.0. New best score: 0.039
Metric val_loss improved by 0.001 >= min_delta = 0.0. New best score: 0.037
Metric val_loss improved by 0.001 >= min_delta = 0.0. New best score: 0.036
Metric val_loss improved by 0.001 >= min_delta = 0.0. New best score: 0.036
Metric val_loss improved by 0.001 >= min_delta = 0.0. New best score: 0.035
Metric val_loss improved by 0.000 >= min_delta = 0.0. New best score: 0.034
Metric val_loss improved by 0.002 >= min_delta = 0.0. New best score: 0.033
Metric val_loss improved by 0.001 >= min_delta = 0.0. New best score: 0.032
Metric val_loss improved by 0.000 >= min_delta = 0.0. New best score: 0.031
Metric val_loss improved by 0.000 >= min_delta = 0.0. New best score: 0.031
Metric val_loss improved by 0.000 >= min_delta = 0.0. New best score: 0.031
Metric val_loss improved by 0.000 >= min_delta = 0.0. New best score: 0.031
Metric val_loss improved by 0.000 >= min_delta = 0.0. New best score: 0.030
`Trainer.fit` stopped: `max_epochs=30` reached.
[I 2026-04-27 01:09:31,549] Trial 2 finished with value: 0.030423752963542938 and parameters: {'depth': 2, 'message_hidden_dim': 1138, 'ffn_hidden_dim': 2024, 'ffn_num_layers': 4, 'dropout': 0.03897698441977729}. Best is trial 1 with value: 0.022328566759824753.
GPU available: False, used: False
TPU available: False, using: 0 TPU cores
💡 Tip: For seamless cloud logging and experiment tracking, try installing [litlogger](https://pypi.org/project/litlogger/) to enable LitLogger, which logs metrics and artifacts automatically to the Lightning Experiments platform.
Loading `train_dataloader` to estimate number of stepping batches.
Metric val_loss improved. New best score: 0.664
Metric val_loss improved by 0.516 >= min_delta = 0.0. New best score: 0.148
Metric val_loss improved by 0.054 >= min_delta = 0.0. New best score: 0.094
Metric val_loss improved by 0.054 >= min_delta = 0.0. New best score: 0.040
Metric val_loss improved by 0.001 >= min_delta = 0.0. New best score: 0.039
Metric val_loss improved by 0.002 >= min_delta = 0.0. New best score: 0.037
Metric val_loss improved by 0.001 >= min_delta = 0.0. New best score: 0.037
Metric val_loss improved by 0.003 >= min_delta = 0.0. New best score: 0.033
Metric val_loss improved by 0.001 >= min_delta = 0.0. New best score: 0.032
Metric val_loss improved by 0.000 >= min_delta = 0.0. New best score: 0.032
Metric val_loss improved by 0.000 >= min_delta = 0.0. New best score: 0.032
Metric val_loss improved by 0.000 >= min_delta = 0.0. New best score: 0.032
Metric val_loss improved by 0.001 >= min_delta = 0.0. New best score: 0.030
`Trainer.fit` stopped: `max_epochs=30` reached.
[I 2026-04-27 01:11:46,637] Trial 3 finished with value: 0.030498437583446503 and parameters: {'depth': 3, 'message_hidden_dim': 402, 'ffn_hidden_dim': 1160, 'ffn_num_layers': 4, 'dropout': 0.3855485102595399}. Best is trial 1 with value: 0.022328566759824753.
GPU available: False, used: False
TPU available: False, using: 0 TPU cores
💡 Tip: For seamless cloud logging and experiment tracking, try installing [litlogger](https://pypi.org/project/litlogger/) to enable LitLogger, which logs metrics and artifacts automatically to the Lightning Experiments platform.
Loading `train_dataloader` to estimate number of stepping batches.
Metric val_loss improved. New best score: 0.656
Metric val_loss improved by 0.425 >= min_delta = 0.0. New best score: 0.232
Metric val_loss improved by 0.168 >= min_delta = 0.0. New best score: 0.063
Metric val_loss improved by 0.015 >= min_delta = 0.0. New best score: 0.048
Metric val_loss improved by 0.006 >= min_delta = 0.0. New best score: 0.042
Metric val_loss improved by 0.001 >= min_delta = 0.0. New best score: 0.041
Metric val_loss improved by 0.003 >= min_delta = 0.0. New best score: 0.038
Metric val_loss improved by 0.001 >= min_delta = 0.0. New best score: 0.038
Metric val_loss improved by 0.001 >= min_delta = 0.0. New best score: 0.037
Metric val_loss improved by 0.001 >= min_delta = 0.0. New best score: 0.037
Metric val_loss improved by 0.000 >= min_delta = 0.0. New best score: 0.037
`Trainer.fit` stopped: `max_epochs=30` reached.
[I 2026-04-27 01:17:28,986] Trial 4 finished with value: 0.03657121956348419 and parameters: {'depth': 3, 'message_hidden_dim': 969, 'ffn_hidden_dim': 1977, 'ffn_num_layers': 4, 'dropout': 0.17549638821895105}. Best is trial 1 with value: 0.022328566759824753.
GPU available: False, used: False
TPU available: False, using: 0 TPU cores
💡 Tip: For seamless cloud logging and experiment tracking, try installing [litlogger](https://pypi.org/project/litlogger/) to enable LitLogger, which logs metrics and artifacts automatically to the Lightning Experiments platform.
Loading `train_dataloader` to estimate number of stepping batches.
Metric val_loss improved. New best score: 0.494
Metric val_loss improved by 0.447 >= min_delta = 0.0. New best score: 0.048
Metric val_loss improved by 0.005 >= min_delta = 0.0. New best score: 0.042
Metric val_loss improved by 0.008 >= min_delta = 0.0. New best score: 0.035
Metric val_loss improved by 0.001 >= min_delta = 0.0. New best score: 0.034
Metric val_loss improved by 0.002 >= min_delta = 0.0. New best score: 0.032
Metric val_loss improved by 0.002 >= min_delta = 0.0. New best score: 0.030
Metric val_loss improved by 0.001 >= min_delta = 0.0. New best score: 0.029
Metric val_loss improved by 0.000 >= min_delta = 0.0. New best score: 0.028
Metric val_loss improved by 0.000 >= min_delta = 0.0. New best score: 0.028
Metric val_loss improved by 0.000 >= min_delta = 0.0. New best score: 0.028
Metric val_loss improved by 0.000 >= min_delta = 0.0. New best score: 0.028
`Trainer.fit` stopped: `max_epochs=30` reached.
[I 2026-04-27 01:19:57,804] Trial 5 finished with value: 0.02777571976184845 and parameters: {'depth': 6, 'message_hidden_dim': 261, 'ffn_hidden_dim': 1198, 'ffn_num_layers': 4, 'dropout': 0.0509477056981511}. Best is trial 1 with value: 0.022328566759824753.
GPU available: False, used: False
TPU available: False, using: 0 TPU cores
💡 Tip: For seamless cloud logging and experiment tracking, try installing [litlogger](https://pypi.org/project/litlogger/) to enable LitLogger, which logs metrics and artifacts automatically to the Lightning Experiments platform.
Loading `train_dataloader` to estimate number of stepping batches.
Metric val_loss improved. New best score: 0.724
Metric val_loss improved by 0.500 >= min_delta = 0.0. New best score: 0.224
Metric val_loss improved by 0.157 >= min_delta = 0.0. New best score: 0.067
Metric val_loss improved by 0.024 >= min_delta = 0.0. New best score: 0.043
Metric val_loss improved by 0.007 >= min_delta = 0.0. New best score: 0.036
Metric val_loss improved by 0.000 >= min_delta = 0.0. New best score: 0.035
Metric val_loss improved by 0.000 >= min_delta = 0.0. New best score: 0.035
Metric val_loss improved by 0.004 >= min_delta = 0.0. New best score: 0.032
Metric val_loss improved by 0.004 >= min_delta = 0.0. New best score: 0.028
Metric val_loss improved by 0.000 >= min_delta = 0.0. New best score: 0.028
Metric val_loss improved by 0.001 >= min_delta = 0.0. New best score: 0.027
Metric val_loss improved by 0.000 >= min_delta = 0.0. New best score: 0.027
`Trainer.fit` stopped: `max_epochs=30` reached.
[I 2026-04-27 01:21:13,488] Trial 6 finished with value: 0.02671232260763645 and parameters: {'depth': 2, 'message_hidden_dim': 374, 'ffn_hidden_dim': 465, 'ffn_num_layers': 2, 'dropout': 0.23628525744990558}. Best is trial 1 with value: 0.022328566759824753.
GPU available: False, used: False
TPU available: False, using: 0 TPU cores
💡 Tip: For seamless cloud logging and experiment tracking, try installing [litlogger](https://pypi.org/project/litlogger/) to enable LitLogger, which logs metrics and artifacts automatically to the Lightning Experiments platform.
Loading `train_dataloader` to estimate number of stepping batches.
Metric val_loss improved. New best score: 0.619
Metric val_loss improved by 0.177 >= min_delta = 0.0. New best score: 0.442
Metric val_loss improved by 0.157 >= min_delta = 0.0. New best score: 0.285
Metric val_loss improved by 0.146 >= min_delta = 0.0. New best score: 0.139
Metric val_loss improved by 0.045 >= min_delta = 0.0. New best score: 0.093
Metric val_loss improved by 0.023 >= min_delta = 0.0. New best score: 0.070
Metric val_loss improved by 0.013 >= min_delta = 0.0. New best score: 0.057
Metric val_loss improved by 0.006 >= min_delta = 0.0. New best score: 0.051
Metric val_loss improved by 0.002 >= min_delta = 0.0. New best score: 0.049
Metric val_loss improved by 0.007 >= min_delta = 0.0. New best score: 0.042
Metric val_loss improved by 0.006 >= min_delta = 0.0. New best score: 0.036
Metric val_loss improved by 0.001 >= min_delta = 0.0. New best score: 0.035
Metric val_loss improved by 0.001 >= min_delta = 0.0. New best score: 0.034
Metric val_loss improved by 0.002 >= min_delta = 0.0. New best score: 0.032
Metric val_loss improved by 0.000 >= min_delta = 0.0. New best score: 0.032
Metric val_loss improved by 0.001 >= min_delta = 0.0. New best score: 0.031
`Trainer.fit` stopped: `max_epochs=30` reached.
[I 2026-04-27 01:30:50,050] Trial 7 finished with value: 0.031370263546705246 and parameters: {'depth': 6, 'message_hidden_dim': 1011, 'ffn_hidden_dim': 411, 'ffn_num_layers': 1, 'dropout': 0.04015503210459999}. Best is trial 1 with value: 0.022328566759824753.
GPU available: False, used: False
TPU available: False, using: 0 TPU cores
💡 Tip: For seamless cloud logging and experiment tracking, try installing [litlogger](https://pypi.org/project/litlogger/) to enable LitLogger, which logs metrics and artifacts automatically to the Lightning Experiments platform.
Loading `train_dataloader` to estimate number of stepping batches.
Metric val_loss improved. New best score: 0.192
Metric val_loss improved by 0.117 >= min_delta = 0.0. New best score: 0.075
Metric val_loss improved by 0.031 >= min_delta = 0.0. New best score: 0.044
Metric val_loss improved by 0.002 >= min_delta = 0.0. New best score: 0.042
Metric val_loss improved by 0.001 >= min_delta = 0.0. New best score: 0.041
Metric val_loss improved by 0.002 >= min_delta = 0.0. New best score: 0.039
Metric val_loss improved by 0.001 >= min_delta = 0.0. New best score: 0.039
Metric val_loss improved by 0.003 >= min_delta = 0.0. New best score: 0.036
Metric val_loss improved by 0.002 >= min_delta = 0.0. New best score: 0.033
Metric val_loss improved by 0.000 >= min_delta = 0.0. New best score: 0.033
Metric val_loss improved by 0.000 >= min_delta = 0.0. New best score: 0.033
Metric val_loss improved by 0.001 >= min_delta = 0.0. New best score: 0.032
Metric val_loss improved by 0.001 >= min_delta = 0.0. New best score: 0.031
`Trainer.fit` stopped: `max_epochs=30` reached.
[I 2026-04-27 01:34:55,401] Trial 8 finished with value: 0.031057721003890038 and parameters: {'depth': 4, 'message_hidden_dim': 378, 'ffn_hidden_dim': 2627, 'ffn_num_layers': 4, 'dropout': 0.23166900243127092}. Best is trial 1 with value: 0.022328566759824753.
GPU available: False, used: False
TPU available: False, using: 0 TPU cores
💡 Tip: For seamless cloud logging and experiment tracking, try installing [litlogger](https://pypi.org/project/litlogger/) to enable LitLogger, which logs metrics and artifacts automatically to the Lightning Experiments platform.
Loading `train_dataloader` to estimate number of stepping batches.
Metric val_loss improved. New best score: 0.687
Metric val_loss improved by 0.002 >= min_delta = 0.0. New best score: 0.685
Metric val_loss improved by 0.345 >= min_delta = 0.0. New best score: 0.340
Metric val_loss improved by 0.231 >= min_delta = 0.0. New best score: 0.109
Metric val_loss improved by 0.035 >= min_delta = 0.0. New best score: 0.074
Metric val_loss improved by 0.029 >= min_delta = 0.0. New best score: 0.045
Metric val_loss improved by 0.006 >= min_delta = 0.0. New best score: 0.039
Metric val_loss improved by 0.003 >= min_delta = 0.0. New best score: 0.036
Metric val_loss improved by 0.004 >= min_delta = 0.0. New best score: 0.032
Metric val_loss improved by 0.000 >= min_delta = 0.0. New best score: 0.032
Metric val_loss improved by 0.001 >= min_delta = 0.0. New best score: 0.031
Metric val_loss improved by 0.001 >= min_delta = 0.0. New best score: 0.030
Metric val_loss improved by 0.003 >= min_delta = 0.0. New best score: 0.027
Metric val_loss improved by 0.001 >= min_delta = 0.0. New best score: 0.026
Monitored metric val_loss did not improve in the last 7 records. Best score: 0.026. Signaling Trainer to stop.
`Trainer.fit` stopped: `max_epochs=30` reached.
[I 2026-04-27 01:41:40,193] Trial 9 finished with value: 0.025839976966381073 and parameters: {'depth': 2, 'message_hidden_dim': 1512, 'ffn_hidden_dim': 256, 'ffn_num_layers': 4, 'dropout': 0.13922405237141394}. Best is trial 1 with value: 0.022328566759824753.

3. EXPORTING RESULTS AND BEST HYPERPARAMETERS
Search complete. History saved to 'optuna_results_test.csv'.

THE BEST MODEL IS TRIAL #1
Best Validation Loss: 0.0223
Optimal Parameters Found:
  - depth: 6
  - message_hidden_dim: 1135
  - ffn_hidden_dim: 3508
  - ffn_num_layers: 1
  - dropout: 0.0005514392392346857

Total execution time: 55.81 minutes.



'''

