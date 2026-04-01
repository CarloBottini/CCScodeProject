#importamos y mantenemos tambien el escalador
#usaremos 3 modelos independientes

from chemprop.data import MoleculeDatapoint, MoleculeDataset, build_dataloader
from chemprop.nn import BondMessagePassing, SumAggregation, RegressionFFN, UnscaleTransform
from chemprop.models import MPNN
import numpy as np
import torch

print("--- 1. PREPARANDO DATOS Y ESCALADOR ---")
mol1 = MoleculeDatapoint.from_smi("C", y=np.array([100.0]))    
mol2 = MoleculeDatapoint.from_smi("CC", y=np.array([200.0]))   
mol3 = MoleculeDatapoint.from_smi("CCC", y=np.array([300.0]))  
mol4 = MoleculeDatapoint.from_smi("CCCC", y=np.array([400.0])) 

dataset_train = MoleculeDataset([mol1, mol2])
dataset_val = MoleculeDataset([mol3, mol4])

escalador = dataset_train.normalize_targets()
dataset_val.normalize_targets(escalador)
paquete_val = next(iter(build_dataloader(dataset_val, batch_size=2, shuffle=False)))

print("\n--- 2. CONTRATANDO AL TRIBUNAL MÉDICO (ENSEMBLE DE 3 MODELOS) ---")
# Creamos una lista vacía para guardar a nuestros 3 "médicos" (modelos independientes)
tribunal = [] #el grupo de medicos/modelos
numero_de_modelos = 3

transformacion_salida = UnscaleTransform.from_standard_scaler(escalador)

for i in range(numero_de_modelos):
    # En cada vuelta del bucle, nace un cerebro nuevo con conexiones aleatorias distintas
    mp = BondMessagePassing()
    agg = SumAggregation()
    ffn = RegressionFFN(output_transform=transformacion_salida)
    
    modelo = MPNN(mp, agg, ffn)
    modelo.eval() #modo predicción
    tribunal.append(modelo)
    print(f"Médico {i+1} incorporado al tribunal.")

print("\n--- 3. EL DIAGNÓSTICO CONJUNTO ---")
todas_las_predicciones = []

# Le pedimos el diagnóstico a cada modelo/medico individualmente
for i, medico in enumerate(tribunal):
    # Cada modelo/medico analiza el mismo paquete de validación
    prediccion_medico = medico(paquete_val.bmg)
    todas_las_predicciones.append(prediccion_medico)
    # Mostramos qué opina cada uno (como no están entrenados, dirán cosas un poco distintas)
    print(f"-> Diagnóstico Médico {i+1}: {prediccion_medico.flatten().tolist()}")

print("\n--- 4. RESOLUCIÓN DEL TRIBUNAL (LA MEDIA) ---")
# juntamos todas las opiniones y hacer la media exacta
# torch.stack apila las respuestas, y .mean(dim=0) hace la media por columnas
diagnostico_final = torch.stack(todas_las_predicciones).mean(dim=0)

print(f"Valores REALES Originales:  [300.0, 400.0]")
print(f"DIAGNÓSTICO FINAL ENSEMBLE: {diagnostico_final.flatten().tolist()}")
print("==================================================")

