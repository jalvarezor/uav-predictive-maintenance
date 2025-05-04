# Preprocesamiento del dataset ALFA para ML no supervisado
# Autor: Jorge Alvarez
# TFM - Detección de Fallos en UAVs

import pandas as pd
import numpy as np
from pathlib import Path
from scipy.interpolate import interp1d

# Rutas base del proyecto
RUTA_PROYECTO = Path("C:/Users/2890960/Documents/master/tfm/uav-predictive-maintenance")
RUTA_INPUT     = RUTA_PROYECTO / "data" / "input"
RUTA_OUTPUT    = RUTA_PROYECTO / "data" / "output"
RUTA_OUTPUT.mkdir(exist_ok=True)

print("Iniciando preprocesamiento no supervisado...")


# Convierte %time a segundos desde inicio
def to_seconds(df):
    df['timestamp'] = pd.to_datetime(df['%time'], errors='coerce')
    df['timestamp_unix'] = (df['timestamp'].astype(np.int64) // 1_000_000_000).astype(float)
    return df['timestamp_unix'] - df['timestamp_unix'].iloc[0]


# Interpola señal al tiempo común
def align_signal(x, y, t_target):
    df = pd.DataFrame({'t': x, 'y': y})
    df.drop_duplicates('t', keep='first', inplace=True)
    f = interp1d(df['t'], df['y'], kind='linear', fill_value='extrapolate', assume_sorted=True)
    return pd.Series(f(t_target), index=t_target)


# Carga IMU, HUD y RC-Out de una secuencia de ALFA
def cargar_secuencia(ruta_secuencia):
    # Buscar archivos
    def encontrar(*patrones):
        for patron in patrones:
            arch = list(ruta_secuencia.glob(f"*{patron}*"))
            if arch:
                return pd.read_csv(arch[0])
        return None

    df_imu = encontrar("-mavros-imu-data.csv")
    df_hud = encontrar("-mavros-vfr_hud.csv")
    df_rc = encontrar("-mavros-rc-out.csv")

    # Tiempo común
    t_imu = to_seconds(df_imu) if df_imu is not None else np.array([])
    t_hud = to_seconds(df_hud) if df_hud is not None else np.array([])
    t_rc = to_seconds(df_rc) if df_rc is not None else np.array([])

    max_time = max(
        t_imu.max() if len(t_imu) > 0 else 100,
        t_hud.max() if len(t_hud) > 0 else 100,
        t_rc.max() if len(t_rc) > 0 else 100
    )
    t_common = np.arange(0, max_time, 0.5)

    # Señales clave
    imu_az = align_signal(to_seconds(df_imu), df_imu['field.linear_acceleration.z'], t_common) if df_imu is not None and 'field.linear_acceleration.z' in df_imu.columns else pd.Series(np.zeros_like(t_common), index=t_common)
    throttle = align_signal(to_seconds(df_rc), df_rc['field.channels2'], t_common) if df_rc is not None and 'field.channels2' in df_rc.columns else pd.Series(np.full_like(t_common, 1500), index=t_common)
    altitude = align_signal(to_seconds(df_hud), df_hud['field.altitude'], t_common) if df_hud is not None and 'field.altitude' in df_hud.columns else pd.Series(np.zeros_like(t_common), index=t_common)

    return pd.DataFrame({
        'timestamp': t_common,
        'imu_az': imu_az.values,
        'altitude': altitude.values,
        'throttle': throttle.values
    })


# Extrae características por ventana temporal
def extraer_caracteristicas_ventana(df, ventana_segundos=5, frecuencia_muestreo=0.5):
    ventana_filas = int(ventana_segundos / frecuencia_muestreo)
    cols = ['imu_az', 'altitude', 'throttle']
    df_features = df[['timestamp']].copy()

    for col in cols:
        df_features[f'{col}_mean'] = df[col].rolling(window=ventana_filas).mean()
        df_features[f'{col}_std'] = df[col].rolling(window=ventana_filas).std()
        df_features[f'{col}_slope'] = df[col].diff(periods=ventana_filas) / ventana_segundos

    return df_features.dropna().reset_index(drop=True)


# Procesa automáticamente todas las secuencias seleccionadas
def procesar_todas_las_secuencias(ruta_base):
    dfs_normales = []
    dfs_fallo = []

    for carpeta in ruta_base.iterdir():
        if not carpeta.is_dir():
            continue
        nombre = carpeta.name

        if '_no_failure' in nombre:
            df = cargar_secuencia(carpeta)
            if not df.empty:
                dfs_normales.append(extraer_caracteristicas_ventana(df))

        elif '_engine_failure' in nombre:
            df = cargar_secuencia(carpeta)
            if not df.empty:
                dfs_fallo.append(extraer_caracteristicas_ventana(df))

    X_train = pd.concat(dfs_normales, ignore_index=True).drop(columns=['timestamp'])
    X_test = pd.concat(dfs_fallo, ignore_index=True).drop(columns=['timestamp'])

    return X_train, X_test


# Exporta los conjuntos normal y anómalo
def guardar_dataset(X_train, X_test):
    pd.DataFrame(X_train).to_csv(RUTA_OUTPUT / "X_train.csv", index=False)
    pd.DataFrame(X_test).to_csv(RUTA_OUTPUT / "X_test.csv", index=False)
    print("\nArchivos exportados:")
    print(f"- X_train.csv: {len(X_train)} muestras (normales)")
    print(f"- X_test.csv: {len(X_test)} muestras (fallo)\n")


if __name__ == "__main__":
    X_train, X_test = procesar_todas_las_secuencias(RUTA_INPUT)
    if not X_train.empty and not X_test.empty:
        guardar_dataset(X_train, X_test)