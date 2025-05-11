# Entrenamiento de modelos ML no supervisados:
#    - Isolation Forest
#    - One-Class SVM
#    - LOF
#    - KMeans
# usando X_train.csv
#
# Autor: Jorge Alvarez
# TFM - Detección de Fallos en UAVs

from pathlib import Path
import pandas as pd
import joblib
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.neighbors import LocalOutlierFactor
from sklearn.cluster import KMeans

RUTA_PROYECTO = Path(__file__).parent.parent
RUTA_OUTPUT = RUTA_PROYECTO / "data" / "output"
RUTA_MODELOS = RUTA_PROYECTO / "models"
RUTA_MODELOS.mkdir(exist_ok=True)

print("Cargando datos de entrenamiento...")

X_train_path = RUTA_OUTPUT / "X_train.csv"
if not X_train_path.exists():
    print("Error: no se encontró X_train.csv")
    exit()
try:
    X_train = pd.read_csv(X_train_path)
except Exception as e:
    print(f"Error: no se pudo cargar X_train.csv: {str(e)}")
    exit()

# Solo características (sin timestamp)
X_train_clean = X_train.drop(columns=['timestamp'], errors='ignore')

# Entrenamiento de modelos no supervisados
print("\nEntrenando modelos...")

# Isolation Forest
isol_forest = IsolationForest(n_estimators=100, contamination=0.05)
isol_forest.fit(X_train_clean)
joblib.dump(isol_forest, RUTA_MODELOS / "model_isol_forest.joblib")

# One-Class SVM
oc_svm = OneClassSVM(kernel='rbf', nu=0.05)
oc_svm.fit(X_train_clean)
joblib.dump(oc_svm, RUTA_MODELOS / "model_oc_svm.joblib")

# Local Outlier Factor (LOF)
lof = LocalOutlierFactor(n_neighbors=20, novelty=True)
lof.fit(X_train_clean)
joblib.dump(lof, RUTA_MODELOS / "model_lof.joblib")

# KMeans – usado como detector de anomalías
kmeans = KMeans(n_clusters=2)
kmeans.fit(X_train_clean)
joblib.dump(kmeans, RUTA_MODELOS / "model_kmeans.joblib")

print("\nModelos entrenados y guardados en 'models'")