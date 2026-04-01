#importamos

from chemprop.data import MoleculeDatapoint, MoleculeDataset, build_dataloader
from chemprop.nn import BondMessagePassing, SumAggregation, RegressionFFN
from chemprop.models import MPNN
# NUEVO: Importamos MAE y RMSE con sus nombres actualizados de la v2
from chemprop.nn.metrics import MAE, RMSE
import numpy as np
import torch
import os

print("--- 1. PREPARANDO DATOS Y MODELO ---")
mol1 = MoleculeDatapoint.from_smi("C", y=np.array([10.0]))
mol2 = MoleculeDatapoint.from_smi("CC", y=np.array([20.0]))
mi_dataset = MoleculeDataset([mol1, mol2])
paquete = next(iter(build_dataloader(mi_dataset, batch_size=2, shuffle=False)))

# Construimos el cerebro
modelo = MPNN(BondMessagePassing(), SumAggregation(), RegressionFFN())
modelo.eval() # modo prediccion

print("\n--- 2. CALCULANDO MÉTRICAS OFICIALES ---")
predicciones = modelo(paquete.bmg)
valores_reales = paquete.Y

# Usamos las clases de métricas de Chemprop
metrica_mae = MAE()
metrica_rmse = RMSE()

# Calculamos (le pasamos predicción y realidad)
valor_mae = metrica_mae(predicciones, valores_reales)
valor_rmse = metrica_rmse(predicciones, valores_reales)

print(f"Resultado MAE: {valor_mae.item():.4f}")
print(f"Resultado RMSE: {valor_rmse.item():.4f}")

print("\n--- 3. GUARDANDO EL MODELO EN EL DISCO ---")
# Definimos un nombre de archivo para guardar
PATH_GUARDADO = "modelo_ccs_prueba.pt"

# Guardamos el "state_dict" (los pesos de las neuronas) en el archivo
torch.save(modelo.state_dict(), PATH_GUARDADO)
print(f"Cerebro guardado con éxito como: {PATH_GUARDADO}")

print("\n--- 4. CARGANDO EL MODELO (PRUEBA DE MEMORIA) ---")
# Creamos un cerebro nuevo, vacío y sin conocimientos
nuevo_modelo = MPNN(BondMessagePassing(), SumAggregation(), RegressionFFN())

# Le "inyectamos" los conocimientos del archivo que acabamos de guardar
nuevo_modelo.load_state_dict(torch.load(PATH_GUARDADO, weights_only=True))
nuevo_modelo.eval()

# Comprobamos que el nuevo modelo predice exactamente lo mismo que el original
nueva_prediccion = nuevo_modelo(paquete.bmg)
print(f"Predicción original: {predicciones.flatten().tolist()[0]:.4f}")
print(f"Predicción cargada:  {nueva_prediccion.flatten().tolist()[0]:.4f}")

if torch.allclose(predicciones, nueva_prediccion):
    print("¡ÉXITO! El modelo cargado tiene exactamente la misma memoria que el original.")



