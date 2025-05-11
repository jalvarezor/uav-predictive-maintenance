# Generación de gráficos de señales clave (IMU AZ, throttle, altitude) con superposición de predicciones de modelos
# Autor: Jorge Alvarez
# TFM - Detección de Fallos en UAVs

from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json

RUTA_PROYECTO = Path(__file__).parent.parent
RUTA_OUTPUT    = RUTA_PROYECTO / "data" / "output"

print("Cargando datos y predicciones...")

# Cargar X_test.csv
X_test_path = RUTA_OUTPUT / "X_test.csv"
if not X_test_path.exists():
    print(f"Error: no se encontró '{X_test_path.name}'")
    exit()
try:
    X_test = pd.read_csv(X_test_path)
except Exception as e:
    print(f"Error: no se pudo leer X_test.csv: {str(e)}")
    exit()

# Cargar predicciones_modelos_con_metricas.json
predicciones_path = RUTA_OUTPUT / "predicciones_modelos_con_metricas.json"
if not predicciones_path.exists():
    print(f"Error: no se encontró '{predicciones_path.name}'")
    exit()
with open(predicciones_path, 'r') as f:
    predicciones = json.load(f)

# Señales clave para graficar
senal_x = 'imu_az_mean'    # Aceleración vertical media
senal_y = 'throttle_mean'  # Throttle medio
senal_z = 'altitude_slope' # Pendiente de altitud

# Tiempo para graficar
tiempo = X_test.index

# Graficar detecciones de anomalías
plt.figure(figsize=(16, 10))

for i, modelo in enumerate(predicciones):
    preds = np.array(predicciones[modelo]['preds'])
    idx_anomalies = np.where(preds == 1)[0]

    plt.subplot(len(predicciones), 3, i * 3 + 1)
    plt.plot(tiempo, X_test[senal_x], label=senal_x, color='blue')
    plt.scatter(idx_anomalies, X_test.iloc[idx_anomalies][senal_x], c='red', s=10, label="Anomalía detectada")
    plt.title(f"{modelo} -> {senal_x}")
    plt.legend()
    plt.grid(True)

    plt.subplot(len(predicciones), 3, i * 3 + 2)
    plt.plot(tiempo, X_test[senal_y], label=senal_y, color='green')
    plt.scatter(idx_anomalies, X_test.iloc[idx_anomalies][senal_y], c='red', s=10, label="Anomalía detectada")
    plt.title(f"{modelo} -> {senal_y}")
    plt.legend()
    plt.grid(True)

    plt.subplot(len(predicciones), 3, i * 3 + 3)
    plt.plot(tiempo, X_test[senal_z], label=senal_z, color='orange')
    plt.scatter(idx_anomalies, X_test.iloc[idx_anomalies][senal_z], c='red', s=10, label="Anomalía detectada")
    plt.title(f"{modelo} -> {senal_z}")
    plt.legend()
    plt.grid(True)

plt.tight_layout()
plt.show()
ruta_grafico = RUTA_OUTPUT / "grafico_detecciones_vs_señales.png"
plt.savefig(ruta_grafico, dpi=150, bbox_inches='tight')
print(f"Gráfico guardado en '{ruta_grafico}'")