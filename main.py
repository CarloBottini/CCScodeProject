# Importamos las herramientas de chemprop y numpy (para arrays matematicos)
from chemprop.data import MoleculeDatapoint, MoleculeDataset, build_dataloader
import numpy as np

print("--- 1. DATAPOINTS (Datos individuales) ---")
# MoleculeDatapoint: Es la clase basica. Coge el texto de la molecula (SMILES) 
# y lo junta con su valor de CCS (la 'y') en un solo objeto para que no se pierdan.
# Usamos np.array porque las redes neuronales operan con matrices, no con numeros sueltos.
mol1 = MoleculeDatapoint.from_smi("C", y=np.array([10.0]))    # Metano (1 Carbono)
mol2 = MoleculeDatapoint.from_smi("CC", y=np.array([20.0]))   # Etano (2 Carbonos)
mol3 = MoleculeDatapoint.from_smi("CCC", y=np.array([30.0]))  # Propano (3 Carbonos)
mol4 = MoleculeDatapoint.from_smi("CCCC", y=np.array([40.0])) # Butano (4 Carbonos)
print("4 moléculas creadas con éxito.")

print("\n--- 2. DATASET (La lista oficial) ---")
# Primero las metemos en una lista normal de Python
lista_moleculas = [mol1, mol2, mol3, mol4]
# MoleculeDataset: Convierte la lista normal en un formato especial de Chemprop.
# Esto sirve para que el programa sepa exactamente cuantas moleculas hay y pueda indexarlas bien.
mi_dataset = MoleculeDataset(lista_moleculas)
print(f"Dataset creado. Tamaño total: {len(mi_dataset)} moléculas.")

print("\n--- 3. DATALOADER Y FEATURIZERS ---")
# build_dataloader: Es el iterador/empaquetador. 
# Si tuvieramos 25.000 moleculas, la RAM explotaria. Esto las agrupa en "batches" (paquetes).
# Aqui le decimos que las agrupe de 2 en 2. shuffle=False es para que no las desordene en esta prueba.
mi_dataloader = build_dataloader(mi_dataset, batch_size=2, shuffle=False)
print("DataLoader listo. ¡Iniciando bucle de lectura!")

# Bucle for: Simulamos como la red neuronal va pidiendo los paquetes uno a uno para estudiar
for numero_paquete, paquete in enumerate(mi_dataloader):
    print(f"\n Abriendo Paquete (Batch) número {numero_paquete + 1}:")
    
    # paquete.bmg (Batch Mol Graph): Aqui los Featurizers ya han actuado.
    # Han leido el texto (SMILES) y lo han transformado en un grafo (nodos y enlaces)
    grafo = paquete.bmg
    
    # .shape[0] nos dice el tamaño de la dimension 0 de la matriz (el numero total)
    # V son los Vertices (atomos) y E son las Aristas/Edges (enlaces)
    print(f"   -> Nodos (Átomos) en este paquete: {grafo.V.shape[0]}")
    print(f"   -> Enlaces en este paquete: {grafo.E.shape[0]}")
    
    # paquete.Y: Es la matriz con las respuestas (CCS) que la IA tiene que predecir.
    # Usamos Y mayuscula porque al haber agrupado 2 moleculas, ahora es un vector, no un valor suelto.
    # .flatten() simplemente aplasta la matriz para que se imprima bonito en una sola linea.
    print(f"   -> Valores objetivo (CCS) de este paquete: {paquete.Y.flatten()}")

print("\n====================================================================")