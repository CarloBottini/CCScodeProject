# Importamos la herramienta de grafos de chemprop y rdkit (la librería química base)
from chemprop.featurizers import SimpleMoleculeMolGraphFeaturizer
from rdkit import Chem

print("--- 1. CREANDO LA MOLÉCULA (RDKit) ---")
# Usamos el Etanol como ejemplo (2 carbonos, 1 oxígeno)
smiles = "CCO"
# RDKit lee el texto y crea un objeto químico real en memoria
molecula = Chem.MolFromSmiles(smiles)
print(f"Molécula {smiles} cargada en RDKit correctamente.")

print("\n--- 2. INICIANDO EL FEATURIZER (EL TRADUCTOR) ---")
# Instanciamos la clase que junta los Atom y Bond Featurizers por defecto
traductor = SimpleMoleculeMolGraphFeaturizer()

# Pasamos la molécula por el traductor para obtener el Grafo Molecular (MolGraph)
grafo_molecular = traductor(molecula)
print("¡Traducción a grafo completada!")

print("\n--- 3. INSPECCIONANDO LAS MATRICES ---")
# grafo_molecular.V es la matriz de Vértices (Átomos)
# grafo_molecular.E es la matriz de Aristas (Enlaces)

# .shape nos dice las dimensiones de la matriz: (Filas, Columnas)
# Filas = Número de átomos/enlaces. Columnas = Número de propiedades químicas extraídas.
print(f"Dimensiones de la matriz de Átomos (V): {grafo_molecular.V.shape}")
print(f"Dimensiones de la matriz de Enlaces (E): {grafo_molecular.E.shape}")

print("\n--- VISTAZO A LOS DATOS PUROS (TODA LA MOLÉCULA) ---")
# Al imprimir grafo_molecular.V entero (sin el [0]), la consola nos mostrará 
# 3 filas de números. Cada fila corresponde a un átomo de la molécula "CCO".
print(">>> MATRIZ DE ÁTOMOS (V) COMPLETA:")
print(grafo_molecular.V)

# También imprimimos la matriz de enlaces para ver las 4 direcciones (C-C, C-C, C-O, O-C)
print("\n>>> MATRIZ DE ENLACES (E) COMPLETA:")
print(grafo_molecular.E)
print("==================================================")


