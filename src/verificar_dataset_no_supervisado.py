# Verificar que los archivos X_train.csv y X_test.csv estén listos para ML no supervisado
# Autor: Jorge Alvarez
# TFM - Detección de Fallos en UAVs

from pathlib import Path
import pandas as pd
import numpy as np

RUTA_PROYECTO = Path("C:/Users/2890960/Documents/master/tfm/uav-predictive-maintenance")
RUTA_OUTPUT    = RUTA_PROYECTO / "data" / "output"

print("Revisando dataset no supervisado...\n")


def verificar_archivo(ruta_archivo):
    if not ruta_archivo.exists():
        print(f"Error: el archivo '{ruta_archivo.name}' no existe. Revisa la salida del preprocesamiento.")
        return None

    try:
        df = pd.read_csv(ruta_archivo)
    except Exception as e:
        print(f"Error: no se pudo leer '{ruta_archivo.name}': {str(e)}")
        return None

    print(f"Archivo '{ruta_archivo.name}' cargado correctamente.")
    print(f"  - Número de filas: {len(df)}")
    print(f"  - Columnas ({len(df.columns)}): {', '.join(df.columns.tolist())}")
    
    # Comprobar si hay NaNs
    n_nans = df.isna().sum().sum()
    if n_nans > 0:
        print(f"  - Warning: faltan {n_nans} valores")
    else:
        print(f"  - No falta ningún valor")

    # Revisar si hay valores infinitos o muy grandes
    for col in df.columns:
        if np.isinf(df[col]).any():
            print(f"  - Warning: infinitos detectados en columna '{col}'")
        elif (df[col].abs() > 1e5).any():
            print(f"  - Warning: valores extremos en columna '{col}'")

    print()

    return df


if __name__ == "__main__":
    ruta_X_train = RUTA_OUTPUT / "X_train.csv"
    ruta_X_test = RUTA_OUTPUT / "X_test.csv"

    print("Verificando X_train.csv -> datos normales (sin fallo)")
    df_train = verificar_archivo(ruta_X_train)

    print("Verificando X_test.csv -> datos con fallo")
    df_test = verificar_archivo(ruta_X_test)

    if df_train is not None and df_test is not None:
        print("\Estadísticas básicas de X_train.csv:")
        print(df_train.describe())

        print("\nEstadísticas básicas de X_test.csv:")
        print(df_test.describe())

        print("\nPrimeras filas de X_train.csv:")
        print(df_train.head(5))

        print("\nPrimeras filas de X_test.csv:")
        print(df_test.head(5))

        print("\nVerificación completada. Dataset listo para ML no supervisado.\n")