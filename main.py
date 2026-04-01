# Importamos la red Neuronal desde el módulo 'nn' (Neural Networks) de Chemprop
from chemprop.nn import BondMessagePassing, MeanAggregation, RegressionFFN
from chemprop.models import MPNN

print("--- 1. FABRICANDO LAS PIEZAS DEL CEREBRO ---")

# PIEZA 1: Message Passing (Los "Carteros" químicos)
# Su trabajo es hacer que los átomos se pasen información a través de los enlaces.
mp = BondMessagePassing()
print("Pieza 1 (Message Passing) creada.")

# PIEZA 2: Aggregation (El "Resumen Ejecutivo")
# Usamos 'MeanAggregation' para que haga la Media Matemática de todos los átomos
# y colapse la molécula en un solo vector.
agg = MeanAggregation()
print("Pieza 2 (Aggregation) creada.")

# PIEZA 3: Predictor 
# Usamos 'RegressionFFN' (Feed-Forward Network para Regresión)
# Usamos Regresión porque el CCS es un número continuo (ej: 125.4), no una categoría (ej: "Perro" o "Gato").
ffn = RegressionFFN()
print("Pieza 3 (Predictor FFN) creada.")

print("\n--- 2. ENSAMBLANDO EL MODELO COMPLETO (MPNN) ---")
# MPNN: junta las 3 piezas en el orden correcto
modelo_ccs = MPNN(mp, agg, ffn)

print("¡Cerebro ensamblado con éxito! Aquí tienes su radiografía interna:")
print("==================================================")
# Al imprimir el modelo, la librería PyTorch nos muestra todas sus capas matemáticas ocultas
print(modelo_ccs)
print("==================================================")



