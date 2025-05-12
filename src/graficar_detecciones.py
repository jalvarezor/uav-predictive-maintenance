# Generación de gráficos de señales clave (IMU AZ, throttle, altitude) 
# con superposición de predicciones de modelos.
# Se genera una gráfica por modelo para mejorar la legibilidad.
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

# Cargar el resultado de las evaluaciones de los modelos
predicciones_path = RUTA_OUTPUT / "predicciones_modelos_con_metricas.json"
if not predicciones_path.exists():
    print(f"Error: no se encontró '{predicciones_path.name}'")
    exit()
with open(predicciones_path, 'r') as f:
    predicciones = json.load(f)
print("Datos cargados correctamente.")

# Señales clave para graficar
senal_x = 'imu_az_mean'
senal_y = 'throttle_mean'
senal_z = 'altitude_slope'

# Tiempo para graficar
tiempo = X_test.index

# Graficar resultados por modelo
for i, modelo in enumerate(predicciones):
    info = predicciones[modelo]
    preds = np.array(info['preds'])
    idx_anomalies = np.where(preds == 1)[0]

    # Crear nueva figura para este modelo
    plt.figure(figsize=(14, 8))
    
    # IMU AZ Mean
    plt.subplot(3, 1, 1)
    plt.plot(tiempo, X_test[senal_x], label=senal_x, color='blue')
    plt.scatter(idx_anomalies, X_test.iloc[idx_anomalies][senal_x], c='red', s=10, label="Anomalía detectada")
    plt.title(f"{modelo} -> {senal_x}")
    plt.grid(True)
    plt.legend()

    # Throttle Mean
    plt.subplot(3, 1, 2)
    plt.plot(tiempo, X_test[senal_y], label=senal_y, color='green')
    plt.scatter(idx_anomalies, X_test.iloc[idx_anomalies][senal_y], c='red', s=10, label="Anomalía detectada")
    plt.title(f"{modelo} -> {senal_y}")
    plt.grid(True)
    plt.legend()

    # Altitude Slope
    plt.subplot(3, 1, 3)
    plt.plot(tiempo, X_test[senal_z], label=senal_z, color='orange')
    plt.scatter(idx_anomalies, X_test.iloc[idx_anomalies][senal_z], c='red', s=10, label="Anomalía detectada")
    plt.title(f"{modelo} -> {senal_z}")
    plt.grid(True)
    plt.legend()
    
    # Ajustar diseño
    plt.tight_layout()

    # Limpiar nombre de modelo para usarlo en el nombre del archivo
    from re import sub
    nombre_archivo = sub(r'[^\w\-]', '', modelo.replace(' ', '_'))
    
    # Guardar gráfico en disco
    ruta_grafico = RUTA_OUTPUT / 'graphs' / f"{nombre_archivo}.png"
    plt.savefig(ruta_grafico, dpi=150, bbox_inches='tight')
    print(f"Gráfico guardado: '{ruta_grafico}'")

    # Cerrar la figura antes de crear una nueva
    plt.close()

print("Todos los modelos han sido graficados y guardados individualmente.")