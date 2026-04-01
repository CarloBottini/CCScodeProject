import torch
import lightning.pytorch as pl
import numpy as np
from chemprop.data import MoleculeDatapoint, MoleculeDataset, build_dataloader, make_split_indices
from chemprop.nn import BondMessagePassing, SumAggregation, RegressionFFN, UnscaleTransform
from chemprop.models import MPNN

print("--- 1. FASE DE ENTRENAMIENTO  ---")
# Creamos nuestras 10 moléculas para que el modelo aprenda 
datos_entrenamiento = [MoleculeDatapoint.from_smi("C" * i, y=np.array([i * 10.0])) for i in range(1, 11)]
dataset_train = MoleculeDataset(datos_entrenamiento)

# Escalamos (para que el modelo no colapse con numeros grandes)
escalador = dataset_train.normalize_targets()
loader_train = build_dataloader(dataset_train, batch_size=2, shuffle=True) 

# Montamos el MPNN (nuestro cerebro)
mp = BondMessagePassing()
agg = SumAggregation()
# Le acoplamos el desescalador a la salida para que nos dé el CCS en unidades reales (Å²)
ffn = RegressionFFN(output_transform=UnscaleTransform.from_standard_scaler(escalador))
modelo_ccs = MPNN(mp, agg, ffn)

# Entrenamos rapidisimo a 10 epochs
entrenador = pl.Trainer(max_epochs=10, enable_checkpointing=False, logger=False, enable_progress_bar=False)
entrenador.fit(modelo_ccs, loader_train)
print("Modelo entrenado y listo para el mundo real.")


print("\n--- 2. FASE DE PREDICCIÓN ---")
# Supongamos que se pasan estas 3 moléculas nuevas.
# No conozco su CCS y no hay 'y=...' por ninguna parte.
smiles_nuevos = ["CCCCCCCCCCC", "CCCCCCCCCCCC", "CCCCCCCCCCCCC"] # 11, 12 y 13 carbonos

# 1. Las convierto en Datapoints, pero esta vez a ciegas (sin 'y')
datos_nuevos = []
for smi in smiles_nuevos:
    # Meto el SMILES
    datos_nuevos.append(MoleculeDatapoint.from_smi(smi))

# 2. Las empaqueto en un Dataset y luego en el Dataloader
dataset_ciego = MoleculeDataset(datos_nuevos)
# shuffle=False siempre en predicción para no perder el orden de los resultados y saber quién es quién
loader_prediccion = build_dataloader(dataset_ciego, batch_size=3, shuffle=False)

print("Analizando las nuevas moléculas en el laboratorio virtual")

# En vez de usar .fit() (estudiar) o hacer un bucle manual, usamos .predict().
# Esto coge el dataloader, lo pasa por la CPU de forma ultra optimizada,
# aplica el desescalador automáticamente y te devuelve una lista de tensores.
resultados_brutos = entrenador.predict(modelo_ccs, loader_prediccion)

print("\n--- 3. RESULTADOS FINALES PARA EL INFORME ---")
# 'resultados_brutos' es una lista de lotes (batches). Hay que aplanarla para leerla bien.
predicciones_finales = torch.cat(resultados_brutos, dim=0).flatten().tolist()

# Imprimimos los resultados emparejados con su SMILES original
for smi, ccs_predicho in zip(smiles_nuevos, predicciones_finales):
    print(f"SMILES: {smi: <15} --> CCS Predicho: {ccs_predicho:.2f} Å²")

print("==================================================================")

