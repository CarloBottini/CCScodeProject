'''
30 EPOCHS, 2000 MOLECULES, 400 RESULTS VISUALIZED (20% VALIDATION)

'''

import pandas as pd
import numpy as np
import torch
import lightning.pytorch as pl
from chemprop.data import MoleculeDatapoint, MoleculeDataset, build_dataloader, make_split_indices
from chemprop.nn import BondMessagePassing, SumAggregation, RegressionFFN, UnscaleTransform
from chemprop.models import MPNN

import math
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


#EXPLANATION OF PREVIOUS ERROS ABOUT THE "DIMER HEADER"
csv_path = "data/METLIN_IMS_dimers_rmTM.csv" 
df_debug = pd.read_csv(csv_path, nrows=5)

#Investigating the One Hot Encoding bug
#The model created over 750 extra columns, crashing the system
#To understand why: Lets compare the "Dimer" and "Dimer.1" columns for the first 3 rows

#Displaying only the problematic columns to prove the error
print(df_debug[['smiles', 'Dimer', 'Dimer.1']].head(3).to_string())
'''
The 'Dimer' column contains intermediate numerical calculations (245.44)
The 'Dimer.1' column contains the actual text labels ('Monomer' or 'Dimer')
Because Pandas 'get_dummies' creates a new binary column for every unique value
using the numerical 'Dimer' column created a massive, overfitted network
Then, the RESOLUTION: We must specifically target 'Dimer.1' for our categorical extraction
'''





print("1. LOADING AND CLEANING REAL DATA")
csv_path = "data/METLIN_IMS_dimers_rmTM.csv" 
#Loading 500 rows for example from METLIN
df = pd.read_csv(csv_path, nrows=2000)

#Neural networks cannot process NaN values
#If a molecule is missing its SMILES, Target, or Extra Features, we must drop the entire row.
#Because we need those parameters to predict the CCS
df = df.dropna(subset=['smiles', 'CCS_AVG', 'Adduct', 'Dimer.1'])

print(f"Rows remaining after cleaning missing values: {df.shape[0]}")

print("\n2. ONE HOT ENCODING ")
# Neural Networks only do numbers. They cannot read words "Monomer" or "[M+H]".
# get_dummies converts text categories into binary columns (0.0 or 1.0)
df_encoded = pd.get_dummies(df, columns=['Adduct', 'Dimer.1'], dtype=float)

# Extracting the names of the newly created binary columns for x_d
extra_feature_columns = [col for col in df_encoded.columns if col.startswith('Adduct_') or col.startswith('Dimer.1_')]
print(f"Extra features created for x_d: {extra_feature_columns}")

print("\n 3. BUILDING DATAPOINTS")
datos = []
for index, row in df_encoded.iterrows():
    smi = row['smiles']
    ccs_target = row['CCS_AVG']
    
    # Extract the binary values as a float32 array
    x_d_values = row[extra_feature_columns].values.astype(np.float32)
    
    # x_d parameter is used to inject the features into the Chemprop pipeline
    datos.append(MoleculeDatapoint.from_smi(smi, y=np.array([ccs_target]), x_d=x_d_values))

print(f"Successfully packaged {len(datos)} MoleculeDatapoints.")

print("\n 4. DATA SPLIT AND SCALING ")
dataset_completo = MoleculeDataset(datos)
# 80% of data to update weights, 20% to evaluate performance
indices_train, indices_val, _ = make_split_indices(dataset_completo, sizes=(0.8, 0.2, 0.0), seed=42)

dataset_train = MoleculeDataset([datos[int(i)] for i in indices_train[0]])
dataset_val = MoleculeDataset([datos[int(i)] for i in indices_val[0]])

escalador = dataset_train.normalize_targets()
dataset_val.normalize_targets(escalador)

#grouping 32 molecules per batch each time
loader_train = build_dataloader(dataset_train, batch_size=32, shuffle=True, num_workers=0) 
loader_val = build_dataloader(dataset_val, batch_size=32, shuffle=False, num_workers=0)

print("\n 5. BUILDING THE PROTOTYPE MPNN ")
mp = BondMessagePassing()
agg = SumAggregation()

#The graph neural network (mp) outputs a fixed vector of 300 neurons.
#We must add the number of extra features (Adducts + Dimers) 
#so the Predictor knows exactly how many input neurons to expect.
total_input_dim = mp.output_dim + len(extra_feature_columns)

ffn = RegressionFFN(
    input_dim=total_input_dim, 
    output_transform=UnscaleTransform.from_standard_scaler(escalador)
)

modelo_ccs = MPNN(mp, agg, ffn)
print(f"Model assembled. Predictor input dimension: {total_input_dim}")

print("\n6. TRAINING (REAL PROTOTYPE) ")
entrenador = pl.Trainer(
    max_epochs=30, 
    enable_checkpointing=False, 
    logger=False,
    enable_progress_bar=True
)

entrenador.fit(modelo_ccs, loader_train, loader_val)


print("\n--- 7. VALIDATION RESULTS ---")
modelo_ccs.eval()
# Empty lists to save the results that we will obtain
todas_predicciones = []
todos_reales = []

for paquete in loader_val: # This for search in every batch (los 32 + 32 + 32 + 4) in this case because they are going to be 100 (20%of 500)
    # Prediction on the actual batch
    pred = modelo_ccs(paquete.bmg, X_d=paquete.X_d) #Becareful it is X_d=paquete.X_d
    
    #Desescalamos
    real_desescalado = escalador.inverse_transform(paquete.Y)
    
    #Add to the lists
    todas_predicciones.extend(pred.flatten().tolist())
    todos_reales.extend(real_desescalado.flatten().tolist())

print(f"\nFINAL PREDICTIONS")
for i in range(len(todos_reales)): 
    print(f"{i+1:>3}. Real CCS: {todos_reales[i]:>6.2f} | Predicted CCS: {todas_predicciones[i]:>6.2f}")
print("=============================================")


'''
This prints only 32 molecules, 1 batch
print("\n=== FINAL PREDICTIONS ON REAL METLIN DATA ===")
for i in range(len(valores_reales_desescalados)):
    print(f"Real CCS: {valores_reales_desescalados[i]:>6.2f} | Predicted CCS: {predicciones_planas[i]:>6.2f}")
print("=============================================")

'''

print(" EVALUATION METRICS  ")

# With this we can compare this model against other versions

mae = mean_absolute_error(todos_reales, todas_predicciones)
mse = mean_squared_error(todos_reales, todas_predicciones)
rmse = math.sqrt(mse)
r2 = r2_score(todos_reales, todas_predicciones)

print(f"Mean Absolute Error (MAE) : {mae:>6.2f} Å2  <-- (the Lower the better)")
print(f"Root Mean Squared (RMSE)  : {rmse:>6.2f} Å2  <-- (the Lower the better, this penalizes big errors)")
print(f"R-squared Score (R^2)      : {r2:>6.4f}     <-- (Closer to 1.0 the better)")
print("==================================================")







'''
Comparing 10 epochs with 30 epochs
For 10 epochs
EVALUATION METRICS
Mean Absolute Error (MAE) :  16.06 Å2  <-- (the Lower the better)
Root Mean Squared (RMSE)  :  21.47 Å2  <-- (the Lower the better, this penalizes big errors)
R-squared Score (R^2)      : 0.3924     <-- (Closer to 1.0 the better)


For 30 epochs
EVALUATION METRICS
Mean Absolute Error (MAE) :   9.27 Å2  <-- (the Lower the better)
Root Mean Squared (RMSE)  :  12.59 Å2  <-- (the Lower the better, this penalizes big errors)
R-squared Score (R^2)      : 0.7910     <-- (Closer to 1.0 the better)


For 30 epochs and 2000 molecules
EVALUATION METRICS
Mean Absolute Error (MAE) :   4.36 Å2  <-- (the Lower the better)
Root Mean Squared (RMSE)  :   5.92 Å2  <-- (the Lower the better, this penalizes big errors)
R-squared Score (R^2)      : 0.9373     <-- (Closer to 1.0 the better)

'''







