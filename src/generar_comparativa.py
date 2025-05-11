# Generación de una tabla que compara los resultados de los distintos modelos
# Autor: Jorge Alvarez
# TFM - Detección de Fallos en UAVs

from pathlib import Path
import pandas as pd
import json

RUTA_PROYECTO = Path(__file__).parent.parent
RUTA_OUTPUT    = RUTA_PROYECTO / "data" / "output"

# Cargar predicciones_modelos_con_metricas.json
predicciones_path = RUTA_OUTPUT / "predicciones_modelos_con_metricas.json"
if not predicciones_path.exists():
    print(f"Error: no se encontró '{predicciones_path.name}'")
    exit()
with open(predicciones_path, "r") as f:
    resultados = json.load(f)

tabla = []
for modelo in resultados:
    info = resultados[modelo]
    tabla.append({
        'Modelo': modelo,
        'Precision': info.get('precision', 0),
        'Recall': info.get('recall', 0),
        'F1-Score': info.get('f1', 0),
        'Anomalías Detectadas': sum(info['preds']),
        'Total Muestras': len(info['preds']),
        'Umbral': 'manual' if 'Isolation' in modelo or 'SVM' in modelo else 'percentil'
    })

df = pd.DataFrame(tabla)
print(df.to_string(index=False))
ruta_archivo = RUTA_OUTPUT / "comparativa_modelos_no_supervisados.csv"
df.to_csv(ruta_archivo, index=False)
print(f"Gráfico guardado en '{ruta_archivo}'")