#importamos

from chemprop.data import MoleculeDatapoint, MoleculeDataset, build_dataloader
from chemprop.nn import BondMessagePassing, SumAggregation, RegressionFFN
from chemprop.models import MPNN
import numpy as np
# NUEVO: Importamos PyTorch para tocar activaciones y loss functions a bajo nivel
import torch 
import torch.nn as nn 

print("--- 1. PREPARANDO DATOS ---")
mol1 = MoleculeDatapoint.from_smi("C", y=np.array([10.0]))    
mol2 = MoleculeDatapoint.from_smi("CC", y=np.array([20.0]))   
mol3 = MoleculeDatapoint.from_smi("CCC", y=np.array([30.0]))  
mol4 = MoleculeDatapoint.from_smi("CCCC", y=np.array([40.0])) 

mi_dataset = MoleculeDataset([mol1, mol2, mol3, mol4])
mi_dataloader = build_dataloader(mi_dataset, batch_size=4, shuffle=False)
paquete = next(iter(mi_dataloader))

print("--- 2. CONSTRUYENDO EL CEREBRO ---")
mp = BondMessagePassing()
agg = SumAggregation()

#Activation: le decimos al Predictor que use LeakyReLU en vez del ReLU por defecto
funcion_activacion = nn.LeakyReLU()
ffn = RegressionFFN(activation=funcion_activacion)

modelo_ccs = MPNN(mp, agg, ffn)
print("Cerebro ensamblado (Usando activación LeakyReLU).")

print("\n--- 3. PREDICCIÓN ---")
modelo_ccs.eval()
predicciones = modelo_ccs(paquete.bmg)
valores_reales = paquete.Y # Guardamos los valores reales en una variable

print(f"Valores REALES:    {valores_reales.flatten().tolist()}")
print(f"Valores PREDICHOS: {predicciones.flatten().tolist()}")

print("\n--- 4. EL PROFESOR CORRIGIENDO (Loss Functions) ---")
# NUEVO TEMA (LOSS FUNCTIONS): Evaluamos lo mal que lo ha hecho el modelo sin entrenar

# Profesor 1: Error Absoluto (MAE - L1Loss en PyTorch)
profesor_mae = nn.L1Loss()
nota_mae = profesor_mae(predicciones, valores_reales)

# Profesor 2: Error Cuadrático (MSE - MSELoss en PyTorch)
profesor_mse = nn.MSELoss()
nota_mse = profesor_mse(predicciones, valores_reales)

print(f"Nota MAE (Fallo absoluto medio): {nota_mae.item():.2f}")
print(f"Nota MSE (Fallo elevado al cuadrado): {nota_mse.item():.2f}")
print("==================================")

