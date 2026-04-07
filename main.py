#Using extra  features and descriptors


import torch
import lightning.pytorch as pl
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors
from chemprop.data import MoleculeDatapoint, MoleculeDataset, build_dataloader, make_split_indices
from chemprop.nn import BondMessagePassing, SumAggregation, RegressionFFN, UnscaleTransform
from chemprop.models import MPNN

print("--- 1. DATA PREPARATION (ADDING MACROSCOPIC FEATURES) ---")
'''
Instead of relying solely on the graph, 
we will compute the exact Molecular Weight (MW) with RDKit and feed it directly 
to the neural network as an "Extra Datapoint Descriptor" (x_d).

'''

datos = []
for i in range(1, 11):
    smi = "C" * i
    y_val = i * 10.0
    
    #RDKit to build the molecule in memory
    mol = Chem.MolFromSmiles(smi)
    #Calculate the exact Molecular Weight (ex:Methane =16.04)
    mw = Descriptors.MolWt(mol) #from rdkit
    
    #Extra features (x_d) must be passed as a 1D NumPy array of floats.
    #If I had more features (like LogP or TPSA), I would append them to this list.
    caracteristicas_extra = np.array([mw], dtype=np.float32)
    
    #Inject the x_d array into the Datapoint
    datos.append(MoleculeDatapoint.from_smi(smi, y=np.array([y_val]), x_d=caracteristicas_extra))

dataset_completo = MoleculeDataset(datos)
#The return type of make_split_indices has changed in v2.1 - see help(make_split_indices)


#Data Splitting
indices_train, indices_val, _ = make_split_indices(dataset_completo, sizes=(0.8, 0.2, 0.0), seed=42)

dataset_train = MoleculeDataset([datos[int(i)] for i in indices_train[0]]) #the problem of TypeError is solve here, with indices_train[0]
dataset_val = MoleculeDataset([datos[int(i)] for i in indices_val[0]])

#Scaling the targets
escalador_y = dataset_train.normalize_targets()
dataset_val.normalize_targets(escalador_y)

loader_train = build_dataloader(dataset_train, batch_size=2, shuffle=True) 
loader_val = build_dataloader(dataset_val, batch_size=2, shuffle=False)

print("\n--- 2. BUILDING THE ENHANCED BRAIN ---")
mp = BondMessagePassing()
agg = SumAggregation()

#The graph module (mp) outputs 300 neurons. 
#My extra descriptor (x_d) adds 1 extra feature.
#The network automatically concatenates them (300 + 1 = 301). 
#Therefore, I MUST tell the Predictor FFN to expect 301 input neurons, otherwise it will crash.
numero_de_features_extra = 1
dimension_total = mp.output_dim + numero_de_features_extra

transformacion_salida = UnscaleTransform.from_standard_scaler(escalador_y)

#Predictor
ffn = RegressionFFN(
    input_dim=dimension_total, 
    output_transform=transformacion_salida
)

#Assembling the MPNN
modelo_mejorado = MPNN(mp, agg, ffn)
print("Enhanced Brain assembled. Graph info + MW injected straight into the Predictor.")

print("\n--- 3. TRAINING THE ENHANCED MODEL ---")
entrenador = pl.Trainer(
    max_epochs=10, 
    enable_checkpointing=False, 
    logger=False,
    enable_progress_bar=False
)

print("Training.....")
entrenador.fit(modelo_mejorado, loader_train, loader_val)


print("\n--- 4. PREDICTING WITH EXTRA FEATURES ---")
#Predicting on the validation set.
# The .predict() function can handle the batches AND the x_d arrays internally.
resultados_brutos = entrenador.predict(modelo_mejorado, loader_val)
predicciones_finales = torch.cat(resultados_brutos, dim=0).flatten().tolist()

#Taking the true values to compare
paquete_examen = next(iter(loader_val))
valores_reales_desescalados = escalador_y.inverse_transform(paquete_examen.Y).flatten().tolist()

print("\n=== RESULTS WITH MOLECULAR WEIGHT INJECTED ===")
for real, pred in zip(valores_reales_desescalados, predicciones_finales):
    print(f"Real Target: {real:>6.2f} | Model Predicted: {pred:>6.2f}")
print("==============================================")
