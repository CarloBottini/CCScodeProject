#importamos
from chemprop.data import MoleculeDatapoint, MoleculeDataset, build_dataloader
# NUEVO: Importamos el descompresor (UnscaleTransform)
from chemprop.nn import BondMessagePassing, SumAggregation, RegressionFFN, UnscaleTransform
from chemprop.models import MPNN
import numpy as np

print("--- 1. PREPARANDO DATOS CON NÚMEROS GIGANTES ---")
# Nos inventamos CCS enormes para ver por qué necesitamos el escalador
mol1 = MoleculeDatapoint.from_smi("C", y=np.array([100.0]))    
mol2 = MoleculeDatapoint.from_smi("CC", y=np.array([200.0]))   
mol3 = MoleculeDatapoint.from_smi("CCC", y=np.array([300.0]))  
mol4 = MoleculeDatapoint.from_smi("CCCC", y=np.array([400.0])) 

dataset_entrenamiento = MoleculeDataset([mol1, mol2])
dataset_validacion = MoleculeDataset([mol3, mol4])

# LA MAGIA DEL SCALING: Aplastamos los datos de entrenamiento
escalador = dataset_entrenamiento.normalize_targets()
print("Valores Y de entrenamiento escalados internamente (comprimidos a ~0).")

# ¡CRÍTICO! Usamos esa misma regla matemática para aplastar los de validación
dataset_validacion.normalize_targets(escalador)
print("Valores Y de validación escalados usando la misma regla.")

# Preparamos el paquete de validación para la prueba final
paquete_val = next(iter(build_dataloader(dataset_validacion, batch_size=2, shuffle=False)))

print("\n--- 2. EL CEREBRO CON DESCOMPRESOR ---")
mp = BondMessagePassing()
agg = SumAggregation()

# NUEVO: Le damos la "fórmula inversa" al predictor para que la aplique a la salida
transformacion_salida = UnscaleTransform.from_standard_scaler(escalador) #unscaleTransform
ffn = RegressionFFN(output_transform=transformacion_salida)

# Ensamblamos
modelo_ccs = MPNN(mp, agg, ffn)
modelo_ccs.eval()
print("Cerebro ensamblado con su Descompresor automático.")

print("\n--- 3. PREDICCIÓN (DE VUELTA AL MUNDO REAL) ---")
# La red predice con números pequeños internamente, pero el 'transformacion_salida'
# lo arregla en el último momento para darnos números grandes en pantalla.
predicciones = modelo_ccs(paquete_val.bmg)

print("\n=== RESULTADOS ===")
# ATENCIÓN: Imprimimos los valores originales que metimos en las moléculas 3 y 4 (300 y 400)
print(f"Valores REALES Originales: [300.0, 400.0]")
# Fíjate cómo la IA te devuelve números grandes (cientos), no números cercanos a 0.
print(f"Valores PREDICHOS por IA:  {predicciones.flatten().tolist()}")
print("==================")

'''
el codigo coge los valores 100 y 200, cuya media es 150. 150 se vuelve su nuevo centro, "su cero". 
La IA intenta predecir dando valores cercanos a 0 y leugo se descomprime por lo que obtenemos un valor cercano a 150
en este caso parece que 147.
Recordemos que aqui no hay fase de training ni una bbdd que ayude.

'''

