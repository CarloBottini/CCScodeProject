import torch
import lightning.pytorch as pl
import numpy as np
from chemprop.data import MoleculeDatapoint, MoleculeDataset, build_dataloader, make_split_indices


# CHANGED: RegressionFFN for MveFFN (Mean-Variance Estimation)
from chemprop.nn import BondMessagePassing, SumAggregation, MveFFN, UnscaleTransform

from chemprop.models import MPNN

print("--- 1. DATA PREPARATION (UNCERTAINTY) ---")
#The model will tell how CONFIDENT it is.
datos = []
for i in range(1, 11):
    datos.append(MoleculeDatapoint.from_smi("C" * i, y=np.array([i * 10.0])))

dataset_completo = MoleculeDataset(datos)
#Data split
indices_train, indices_val, _ = make_split_indices(dataset_completo, sizes=(0.8, 0.2, 0.0), seed=42)

dataset_train = MoleculeDataset([datos[int(i)] for i in indices_train[0]])
dataset_val = MoleculeDataset([datos[int(i)] for i in indices_val[0]])

escalador = dataset_train.normalize_targets()
dataset_val.normalize_targets(escalador)

loader_train = build_dataloader(dataset_train, batch_size=2, shuffle=True) 
loader_val = build_dataloader(dataset_val, batch_size=2, shuffle=False)

print("\n--- 2. BUILDING THE UNCERTAINTY BRAIN ---")
mp = BondMessagePassing()
agg = SumAggregation()

#MveFFN outputs a probability distribution.
#It automatically changes the Loss Function to Negative Log Likelihood (NLL).
#The UnscaleTransform will unscale BOTH the mean and the variance back to real units
transformacion_salida = UnscaleTransform.from_standard_scaler(escalador)
ffn = MveFFN(output_transform=transformacion_salida)

modelo_incertidumbre = MPNN(mp, agg, ffn)
print("Brain assembled. Equipped with uncertainty estimation capabilities.")

print("\n--- 3. TRAINING WITH UNCERTAINTY ---")
entrenador = pl.Trainer(
    max_epochs=10, 
    enable_checkpointing=False, 
    logger=False,
    enable_progress_bar=False
)

print("Training the model to estimate its own errors.....")
entrenador.fit(modelo_incertidumbre, loader_train, loader_val)

print("\n--- 4. Test (PREDICTION ± UNCERTAINTY) ---")
smiles_nuevos = ["CCCCCCCCCCC", "CCCCCCCCCCCC"]
datos_nuevos = [MoleculeDatapoint.from_smi(s) for s in smiles_nuevos]
dataset_ciego = MoleculeDataset(datos_nuevos)
loader_pred = build_dataloader(dataset_ciego, batch_size=2, shuffle=False)

#Predicting blindly
resultados_brutos = entrenador.predict(modelo_incertidumbre, loader_pred)
predicciones = torch.cat(resultados_brutos, dim=0)

print("\n=== PREDICTIONS WITH CONFIDENCE INTERVALS ===")
for i, smi in enumerate(smiles_nuevos):
    #MveFFN returns the Mean (index 0) and the Variance (index 1)
    valores = predicciones[i].flatten()
    ccs_predicho = valores[0].item()
    
    #Variance is the square of the standard deviation (std^2).
    varianza = valores[1].item()
    
    #Adding abs() to prevent math errors from float imprecision near zero.
    desviacion_estandar = np.sqrt(abs(varianza)) 
    
    print(f"SMILES: {smi: <15} --> CCS: {ccs_predicho:.2f} ± {desviacion_estandar:.2f} Å2")
print("=============================================")




