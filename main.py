from chemprop.data import MoleculeDatapoint, MoleculeDataset, build_dataloader, make_split_indices
from chemprop.nn import BondMessagePassing, SumAggregation, RegressionFFN, UnscaleTransform
from chemprop.models import MPNN
import numpy as np
# NUEVO: Importamos el pack para el entrenamiento
import lightning.pytorch as pl

print("--- 1. PREPARANDO LOS DATOS ---")
# Creamos 10 moléculas de prueba con valores que tienen lógica matemática
# (Cada carbono añade 10 al CCS, por ejemplo)
datos = []
for i in range(1, 11):
    datos.append(MoleculeDatapoint.from_smi("C" * i, y=np.array([i * 10.0])))

dataset_completo = MoleculeDataset(datos)

# Dividimos en Train (estudiar) y Validation (simulacro de examen)
indices_train, indices_val, _ = make_split_indices(dataset_completo, sizes=(0.8, 0.2, 0.0), seed=42) #test 0

# LA CORRECCIÓN: Usamos la lista original 'datos' en lugar de 'dataset_completo'
dataset_train = MoleculeDataset([datos[int(i)] for i in indices_train[0]])
dataset_val = MoleculeDataset([datos[int(i)] for i in indices_val[0]])

# Escalamos los datos (vital para que aprenda bien)
escalador = dataset_train.normalize_targets()
dataset_val.normalize_targets(escalador)

# Creamos las cintas transportadoras (Dataloaders)
loader_train = build_dataloader(dataset_train, batch_size=2, shuffle=True) 
loader_val = build_dataloader(dataset_val, batch_size=2, shuffle=False)

print("\n--- 2. CONSTRUYENDO EL CEREBRO ---")
mp = BondMessagePassing()
agg = SumAggregation()
transformacion_salida = UnscaleTransform.from_standard_scaler(escalador)
ffn = RegressionFFN(output_transform=transformacion_salida)

modelo_ccs = MPNN(mp, agg, ffn)
print("Cerebro listo para ir a la escuela.")

print("\n--- 3. ENTRENAMIENTO ---")
# Configuramos Trainer: Le decimos que los estudie 10 veces (max_epochs=10)
entrenador = pl.Trainer(
    max_epochs=10, 
    enable_checkpointing=False, 
    logger=False,
    enable_progress_bar=True
)

# Le pasamos el cerebro, los apuntes para estudiar (train) y los simulacros (val)
print("Iniciando las clases... (Esto puede imprimir varias líneas de progreso)")
entrenador.fit(modelo_ccs, loader_train, loader_val)

print("\n--- 4. EXAMEN FINAL (PREDICCIÓN TRAS APRENDER) ---")
# Le ponemos en modo evaluación y comprobamos si ahora acierta
modelo_ccs.eval()
paquete_examen = next(iter(loader_val))

predicciones = modelo_ccs(paquete_examen.bmg)

# LA MAGIA VISUAL: Cogemos los valores reales (Y) que estaban aplastados y los desescalamos
valores_reales_desescalados = escalador.inverse_transform(paquete_examen.Y)

print("\n=== RESULTADOS DESPUÉS DE ESTUDIAR ===")
# Ahora imprimimos los desescalados
print(f"Valores REALES esperados: {valores_reales_desescalados.flatten().tolist()}")
print(f"Valores PREDICHOS por IA: {predicciones.flatten().tolist()}")
print("======================================")


