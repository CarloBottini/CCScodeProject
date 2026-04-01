# Importamos
from chemprop.data import MoleculeDatapoint, MoleculeDataset, build_dataloader
from chemprop.nn import BondMessagePassing, MeanAggregation, RegressionFFN
from chemprop.models import MPNN
import numpy as np

print("--- 1. PREPARANDO LOS DATOS (La Comida) ---")
# Creamos moléculas de prueba
mol1 = MoleculeDatapoint.from_smi("C", y=np.array([10.0]))    # Metano
mol2 = MoleculeDatapoint.from_smi("CC", y=np.array([20.0]))   # Etano
mol3 = MoleculeDatapoint.from_smi("CCC", y=np.array([30.0]))  # Propano
mol4 = MoleculeDatapoint.from_smi("CCCC", y=np.array([40.0])) # Butano

mi_dataset = MoleculeDataset([mol1, mol2, mol3, mol4])

# Las metemos en un paquete usando el Dataloader (batch_size=4 para verlas todas juntas)
mi_dataloader = build_dataloader(mi_dataset, batch_size=4, shuffle=False)
paquete = next(iter(mi_dataloader)) # Extraemos el primer (y único) paquete
print("Datos empaquetados en grafos matemáticos.")

print("\n--- 2. CONSTRUYENDO EL CEREBRO (El Modelo) ---")
# Ensamblamos el MPNN
mp = BondMessagePassing()
agg = MeanAggregation()
ffn = RegressionFFN()
modelo_ccs = MPNN(mp, agg, ffn)
print("Cerebro ensamblado e instanciado.")

print("\n--- 3. EL MOMENTO DE LA VERDAD (La Predicción) ---")
# Modo evaluación: le decimos al cerebro que no aprenda, solo que prediga, NO HAY FASE DE TRAINING
modelo_ccs.eval()

# LE DAMOS DE COMER EL GRAFO AL MODELO
# Extraemos el BatchMolGraph (BMG) de nuestro paquete y se lo pasamos al modelo
grafo_matematico = paquete.bmg
predicciones = modelo_ccs(grafo_matematico)

print("\n=== RESULTADOS DEL EXPERIMENTO ===")
# Imprimimos lo que el modelo DEBERÍA haber dicho (Los valores reales 'Y')
print(f"Valores REALES de CCS:    {paquete.Y.flatten().tolist()}")

# Imprimimos lo que el modelo SE HA INVENTADO (Las predicciones), no habia training
print(f"Valores PREDICHOS por IA: {predicciones.flatten().tolist()}")
print("==================================")


