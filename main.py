
import pandas as pd
import numpy as np
import torch
import lightning.pytorch as pl
from chemprop.data import MoleculeDatapoint, MoleculeDataset, build_dataloader, make_split_indices
from chemprop.nn import BondMessagePassing, SumAggregation, RegressionFFN, UnscaleTransform
from chemprop.models import MPNN

import math
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
#Importing RDKit to calculate Molecular Weight from smiles
from rdkit import Chem
from rdkit.Chem import Descriptors

#Imports for the early stop and the visualization
from lightning.pytorch.callbacks import EarlyStopping
from lightning.pytorch.loggers import CSVLogger
import matplotlib.pyplot as plt
import os

import time #for the cronometer

startCronometer= time.time()

print("1. LOADING AND CLEANING REAL DATA")
csv_path= "data/METLIN_IMS_dimers_rmTM.csv" 
#Loading 2000 rows for example from METLIN
#df= pd.read_csv(csv_path, nrows=2000) #first 2000 rows
df= pd.read_csv(csv_path) #All the dataset


#Neural networks cannot process NaN values
#If a molecule is missing its SMILES, Target, or Extra Features, we must drop the entire row.
#Because we need those parameters to predict the CCS
df = df.dropna(subset=['smiles', 'CCS_AVG', 'Adduct', 'Dimer.1'])

print(f"Rows remaining after cleaning missing values: {df.shape[0]}")

print("\n2. ONE HOT ENCODING ")
#Neural Networks only do numbers. They cannot read words "Monomer" or "[M+H]".
#get_dummies converts text categories into binary columns (0.0 or 1.0)
df_encoded = pd.get_dummies(df, columns=['Adduct', 'Dimer.1'], dtype=float)

#Extracting the names of the newly created binary columns for x_d
extra_feature_columns = [col for col in df_encoded.columns if col.startswith('Adduct_') or col.startswith('Dimer.1_')]
print(f"Extra features created for x_d: {extra_feature_columns}")

print("\n 3. BUILDING DATAPOINTS")
datos = []
valid_mols= 0
invalid_mols= 0
for index, row in df_encoded.iterrows():
    smi = row['smiles']
    ccs_target = row['CCS_AVG']

    #Calculation of the Molecular Weight using RDKit
    mol= Chem.MolFromSmiles(smi)
    if mol is None:
        invalid_mols +=1
        continue #Skip the corrupted smiles strings
    mw= Descriptors.MolWt(mol)

    categorical_values =row[extra_feature_columns].values.astype(np.float32)
    #It is needed to combine the categorical features with MW in the same array x_d
    x_d_values= np.append(categorical_values, mw).astype(np.float32)
    

    datos.append(MoleculeDatapoint.from_smi(smi, y=np.array([ccs_target]), x_d=x_d_values))
    valid_mols +=1

print(f"Successfully packaged {valid_mols} MoleculeDatapoints. (Discarded: {invalid_mols})")



print("\n 4. DATA SPLIT AND SCALING (80% TRAIN / 10%VAL / 10% TEST)")
dataset_completo= MoleculeDataset(datos)
# 80% of data to update weights, 20% to evaluate performance
indices_train, indices_val, indices_test = make_split_indices(dataset_completo, sizes=(0.8, 0.1, 0.1), seed=42)

dataset_train= MoleculeDataset([datos[int(i)] for i in indices_train[0]])
dataset_val= MoleculeDataset([datos[int(i)] for i in indices_val[0]])
dataset_test= MoleculeDataset([datos[int(i)] for i in indices_test[0]])

escalador= dataset_train.normalize_targets()
dataset_val.normalize_targets(escalador)
dataset_test.normalize_targets(escalador) 

#Grouping 32 molecules per batch each time
#The batch size can be upgraded to 64 or 128
loader_train= build_dataloader(dataset_train, batch_size=32, shuffle=True, num_workers=0) 
loader_val= build_dataloader(dataset_val, batch_size=32, shuffle=False, num_workers=0)
loader_test= build_dataloader(dataset_test, batch_size=32, shuffle=False, num_workers=0)


print("\n 5. BUILDING THE PROTOTYPE MPNN ")
mp = BondMessagePassing()
agg = SumAggregation()



#The extra dimension is now: len(extra_feature_columns) + 1 (for the Molecular Weight)
num_extra_features = len(extra_feature_columns) + 1
#The graph neural network (mp) outputs a fixed vector of 300 neurons.
#We must add the number of extra features (Adducts + Dimers) 
#so the Predictor knows exactly how many input neurons to expect.
total_input_dim = mp.output_dim + num_extra_features

ffn = RegressionFFN(
    input_dim=total_input_dim, 
    output_transform=UnscaleTransform.from_standard_scaler(escalador)
)

modelo_ccs = MPNN(mp, agg, ffn)
print(f"Model assembled. Predictor input dimension: {total_input_dim} (300 Graph + {num_extra_features} Macroscopic)")



print("\n 6. TRAINING (REAL PROTOTYPE) WITH EARLY STOPPING ")
EmergenceStop=EarlyStopping(
    monitor="val_loss",
    patience=5,
    mode="min",
    verbose=True
)

logger_csv= CSVLogger("logs", name="ccs_model")

entrenador = pl.Trainer(
    max_epochs=50, 
    enable_checkpointing=False, 
    logger=logger_csv,
    callbacks=[EmergenceStop],
    enable_progress_bar=True
)

entrenador.fit(modelo_ccs, loader_train, loader_val)


print("\n 7. TEST RESULTS")
modelo_ccs.eval()
#Empty lists to save the results that we will obtain
todas_predicciones = []
todos_reales = []

#Using loader_test instead of loader_val
with torch.no_grad():
    for paquete in loader_test: 
        pred = modelo_ccs(paquete.bmg, X_d=paquete.X_d) 

        real_desescalado = escalador.inverse_transform(paquete.Y)
        
        todas_predicciones.extend(pred.flatten().tolist())
        todos_reales.extend(real_desescalado.flatten().tolist())



print(f"\nFINAL PREDICTIONS (showing only the first 30)")
#Only 30, not all
limit = min(30, len(todos_reales))

for i in range(limit): 
    print(f"{i+1:>3}. Real CCS: {todos_reales[i]:>6.2f} | Predicted CCS: {todas_predicciones[i]:>6.2f}")
print("=============================================")


print("\n 8. EVALUATION METRICS")

mae= mean_absolute_error(todos_reales, todas_predicciones)
mse= mean_squared_error(todos_reales, todas_predicciones)
rmse= math.sqrt(mse)
r2= r2_score(todos_reales, todas_predicciones)

print(f"Tested on exactly {len(todos_reales)} unseen molecules (10% Test Set):")
print(f"Mean Absolute Error (MAE) : {mae:>6.2f} Å2  <-- (the Lower the better)")
print(f"Root Mean Squared (RMSE)  : {rmse:>6.2f} Å2  <-- (the Lower the better, this penalizes big errors)")
print(f"R-squared Score (R^2)      : {r2:>6.4f}     <-- (Closer to 1.0 the better)")



print("\n 9. LOSS PLOT")
try:
    version_dir= logger_csv.log_dir
    metrics_path= os.path.join(version_dir, "metrics.csv")

    metrics_df= pd.read_csv(metrics_path)
    
    #Separate Train and Val losses.
    train_loss= metrics_df[['epoch', 'train_loss_epoch']].dropna()
    val_loss= metrics_df[['epoch', 'val_loss']].dropna()

    #Plot
    plt.figure(figsize=(12, 8))
    plt.plot(train_loss['epoch'], train_loss['train_loss_epoch'], label='Train Loss', color='blue', marker='o')
    plt.plot(val_loss['epoch'], val_loss['val_loss'], label='Validation Loss', color='red', marker='x')
    
    plt.title('Model Training History (Learning Curves)')
    plt.xlabel('Epoch')
    plt.ylabel('Loss (Normalized MAE)')
    plt.legend()
    plt.grid(True)
    
    #Saving the plot
    plot_filename= "learning_curve.png"
    plt.savefig(plot_filename)
    print(f"Learning curve saved as '{plot_filename}' in the project folder.")
    
except Exception as e:
    print(f"Plot not generated. Error: {e}")

print("==================================================")



endCronometer=time.time()
timeInSeconds= endCronometer- startCronometer

hours= int(timeInSeconds //3600)
minutes= int((timeInSeconds %3600) //60)
seconds= timeInSeconds%60

print(f"\nTotal executiontime: {hours}h {minutes}m {seconds:.2f}s")






'''
2000 molecules, 50 epochs, 80%, 10%, 10%, 32 batch
EVALUATION METRICS
Tested on exactly 200 unseen molecules (10% Test Set):
Mean Absolute Error (MAE) :   5.52 Å2  <-- (the Lower the better)
Root Mean Squared (RMSE)  :   7.61 Å2  <-- (the Lower the better, this penalizes big errors)
R-squared Score (R^2)      : 0.9059     <-- (Closer to 1.0 the better)

Total executiontime: 0h 3m 38.25s
'''

'''
All dataset, 50 epochs, 80%, 10%, 10%, 32 batch
 EVALUATION METRICS
Tested on exactly 6187 unseen molecules (10% Test Set):
Mean Absolute Error (MAE) :   4.61 Å2  <-- (the Lower the better)
Root Mean Squared (RMSE)  :   6.12 Å2  <-- (the Lower the better, this penalizes big errors)
R-squared Score (R^2)      : 0.9138     <-- (Closer to 1.0 the better)

Total executiontime: 1h 45m 54.24s
I am not sure if this worked correctly, because it stopped at epoch 19/49

'''


