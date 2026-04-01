# Importamos
from chemprop.data import MoleculeDatapoint, MoleculeDataset
from chemprop.data import make_split_indices 
import numpy as np

print("--- 1. CREANDO DATOS DE PRUEBA ---")
datos = []
for i in range(1, 11):
    mol = MoleculeDatapoint.from_smi("C" * i, y=np.array([i * 10.0]))
    datos.append(mol)

mi_dataset = MoleculeDataset(datos)
print(f"Total de moléculas en el dataset principal: {len(mi_dataset)}")

print("\n--- 2. SEPARANDO LOS DATOS (DATA SPLITTING) ---")
indices_train, indices_val, indices_test = make_split_indices(
    mi_dataset,
    sizes=(0.8, 0.1, 0.1), 
    seed=42 
)

# Extraemos la primera división
indices_train = indices_train[0]
indices_val = indices_val[0]
indices_test = indices_test[0]

print(f"Posiciones Train: {indices_train}")
print(f"Posiciones Validation: {indices_val}")
print(f"Posiciones Test: {indices_test}")

print("\n--- 3. CREANDO LOS DATASETS DEFINITIVOS ---")
# LA CORRECCIÓN: Usamos la lista original 'datos' en lugar de 'mi_dataset'
# Así le pasamos los MoleculeDatapoints puros y originales.
dataset_train = MoleculeDataset([datos[int(i)] for i in indices_train])
dataset_val   = MoleculeDataset([datos[int(i)] for i in indices_val])
dataset_test  = MoleculeDataset([datos[int(i)] for i in indices_test])

print(f"Tamaño del Dataset Train: {len(dataset_train)} moléculas")
print(f"Tamaño del Dataset Validation: {len(dataset_val)} moléculas")
print(f"Tamaño del Dataset Test: {len(dataset_test)} moléculas")
print("==================================================")



