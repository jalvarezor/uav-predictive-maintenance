import os
import pandas as pd
import numpy as np
from scipy.interpolate import interp1d
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from pathlib import Path


# Paso 1: Cargar una secuencia desde sus archivos .csv
def cargar_secuencia(ruta_secuencia):
    nombre_secuencia = os.path.basename(ruta_secuencia)
    
    imu_path      = os.path.join(ruta_secuencia, "mavros-imu.csv")
    position_path = os.path.join(ruta_secuencia, "mavros-local_position.csv")
    hud_path      = os.path.join(ruta_secuencia, "mavros-vfr_hud.csv")
    rc_path       = os.path.join(ruta_secuencia, "mavros-rc_out.csv")
    state_path    = os.path.join(ruta_secuencia, "mavros-state.csv")

    # Cargar cada CSV como DataFrame
    df_imu = pd.read_csv(imu_path, parse_dates=["timestamp"])
    df_pos = pd.read_csv(position_path, parse_dates=["timestamp"])
    df_hud = pd.read_csv(hud_path, parse_dates=["timestamp"])
    df_rc  = pd.read_csv(rc_path, parse_dates=["timestamp"])
    df_state = pd.read_csv(state_path, parse_dates=["timestamp"])

    return {
        'nombre': nombre_secuencia,
        'imu': df_imu,
        'position': df_pos,
        'hud': df_hud,
        'rc': df_rc,
        'state': df_state
    }


# Paso 2: Alinear tópicos por marca temporal
def alinear_datos(datos_secuencia):
    df_imu = datos_secuencia['imu']
    df_pos = datos_secuencia['position']
    df_hud = datos_secuencia['hud']
    df_rc  = datos_secuencia['rc']
    df_state = datos_secuencia['state']

    def to_seconds(timestamps):
        return (timestamps - timestamps.iloc[0]).dt.total_seconds()

    # Convertir timestamp a segundos
    t_imu = to_seconds(df_imu['timestamp'])
    t_pos = to_seconds(df_pos['timestamp'])
    t_hud = to_seconds(df_hud['timestamp'])
    t_rc  = to_seconds(df_rc['timestamp'])
    t_state = to_seconds(df_state['timestamp'])

    # Tiempo común (ejemplo: muestreo cada 0.5 segundos)
    t_common = np.linspace(0, max(
        t_imu.max(), t_pos.max(), t_hud.max(),
        t_rc.max(), t_state.max()
    ), int(1e4))

    # Interpolación de IMU
    f_imux = interp1d(t_imu, df_imu['linear_acceleration.x'], fill_value='extrapolate')
    f_imuy = interp1d(t_imu, df_imu['linear_acceleration.y'], fill_value='extrapolate')
    f_imuz = interp1d(t_imu, df_imu['linear_acceleration.z'], fill_value='extrapolate')

    # Interpolación de posición local
    f_velx = interp1d(t_pos, df_pos['velocity.x'], fill_value='extrapolate')
    f_vely = interp1d(t_pos, df_pos['velocity.y'], fill_value='extrapolate')
    f_velz = interp1d(t_pos, df_pos['velocity.z'], fill_value='extrapolate')

    # Interpolación de altitud
    f_altitude = interp1d(t_hud, df_hud['altitude'], fill_value='extrapolate')

    # Interpolación de throttle (PWM)
    f_throttle = interp1d(t_rc, df_rc['servo_pwm'], fill_value='extrapolate')

    # Crear DataFrame final
    df_aligned = pd.DataFrame({
        'timestamp': t_common,
        'imu_ax': f_imux(t_common),
        'imu_ay': f_imuy(t_common),
        'imu_az': f_imuz(t_common),
        'vel_ned_x': f_velx(t_common),
        'vel_ned_y': f_vely(t_common),
        'vel_ned_z': f_velz(t_common),
        'altitude': f_altitude(t_common),
        'throttle': f_throttle(t_common),
        'armed': np.where(np.isin(t_common, t_state), df_state['armed'].iloc[0], False)
    })

    return df_aligned


# Paso 3: Etiquetar según fallo / normal
def etiquetar_fallo(df, ruta_secuencia):
    if "_engine_failure" in ruta_secuencia:
        fs_path = os.path.join(ruta_secuencia, "failure_status-engines.csv")
        df_fs = pd.read_csv(fs_path, parse_dates=["timestamp"])
        t_fallo = (df_fs['timestamp'][0] - df_fs['timestamp'].iloc[0]).total_seconds()
        
        df['fallo'] = np.where(df['timestamp'] >= t_fallo, 1, 0)
    else:
        df['fallo'] = 0

    return df


# Paso 4: Limpieza, imputación y escalado
def limpiar_y_escalar(df):
    imputer = SimpleImputer(strategy='ffill')
    scaler = StandardScaler()
    
    features = df.columns.drop(['timestamp', 'fallo']) if 'fallo' in df.columns else df.columns.drop(['timestamp'])
    
    df_clean = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)
    df_clean[features] = scaler.fit_transform(df_clean[features])

    return df_clean


# Paso 5: Extraer características por ventana temporal
def extraer_caracteristicas_ventana(df, columnas, ventana_segundos=5, frecuencia_muestreo=0.5):
    ventana_filas = int(ventana_segundos / frecuencia_muestreo)

    df_resultados = pd.DataFrame()

    for col in columnas:
        df_resultados[f'{col}_mean'] = df[col].rolling(window=ventana_filas).mean()
        df_resultados[f'{col}_std'] = df[col].rolling(window=ventana_filas).std()
        df_resultados[f'{col}_max'] = df[col].rolling(window=ventana_filas).max()
        df_resultados[f'{col}_min'] = df[col].rolling(window=ventana_filas).min()
        df_resultados[f'{col}_range'] = df_resultados[f'{col}_max'] - df_resultados[f'{col}_min']
        df_resultados[f'{col}_median'] = df[col].rolling(window=ventana_filas).median()
        df_resultados[f'{col}_slope'] = df[col].diff(periods=ventana_filas) / ventana_segundos

    df_resultados['timestamp'] = df['timestamp'].iloc[ventana_filas - 1:].reset_index(drop=True)
    if 'fallo' in df.columns:
        df_resultados['fallo'] = df['fallo'].iloc[ventana_filas - 1:].reset_index(drop=True)

    return df_resultados.dropna().reset_index(drop=True)


# Paso 6: Procesar todas las secuencias
def procesar_todas_las_secuencias(lista_rutas):
    dfs_procesados = []

    for ruta in lista_rutas:
        print(f"Procesando secuencia: {ruta}")
        datos = cargar_secuencia(ruta)
        df_alineado = alinear_datos(datos)
        df_etiquetado = etiquetar_fallo(df_alineado, ruta)
        df_final = limpiar_y_escalar(df_etiquetado)
        dfs_procesados.append(df_final)

    return pd.concat(dfs_procesados, ignore_index=True)


def listar_rutas_secuencias():
    script_path = Path(__file__).resolve()  # ruta de este script
    root_path = script_path.parent.parent  # ruta del proyecto
    data_path = root_path / 'data'  # ruta de las secuencias
    
    lista_rutas_secuencias = []
    for item in data_path.iterdir():
        if item.is_dir():
            lista_rutas_secuencias.append(item)

    return lista_rutas_secuencias


# Paso 8: Ejecutar el preprocesamiento
print("Iniciando preprocesamiento...")
rutas_sequencias = listar_rutas_secuencias()
df_dataset = procesar_todas_las_secuencias(rutas_sequencias)

columnas_relevantes = ['imu_ax', 'imu_ay', 'imu_az', 'vel_ned_x', 'vel_ned_y', 'vel_ned_z', 'altitude', 'throttle']
df_features = extraer_caracteristicas_ventana(df_dataset, columnas_relevantes, ventana_segundos=5, frecuencia_muestreo=0.5)

# Paso 9: Balanceo de clases con SMOTE (si es necesario)
X = df_features.drop(columns=['timestamp', 'fallo']).copy()
y = df_features['fallo'].copy()

smote = SMOTE(random_state=42)
X_res, y_res = smote.fit_resample(X, y)

# Paso 10: Dividir en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.2, stratify=y_res, random_state=42)

# Paso 11: Exportar los conjuntos finales
pd.DataFrame(X_res).to_csv("dataset_preprocesado_X.csv", index=False)
pd.DataFrame(y_res).to_csv("dataset_preprocesado_y.csv", index=False)

pd.DataFrame(X_train).to_csv("X_train.csv", index=False)
pd.DataFrame(X_test).to_csv("X_test.csv", index=False)
pd.DataFrame(y_train).to_csv("y_train.csv", index=False)
pd.DataFrame(y_test).to_csv("y_test.csv", index=False)

print("Preprocesamiento completado. Archivos exportados.")