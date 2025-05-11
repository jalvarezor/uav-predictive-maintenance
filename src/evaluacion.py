# Evaluación de modelos no supervisados con ground truth artificial
# Autor: Jorge Alvarez
# TFM - Detección de Fallos en UAVs

from pathlib import Path
import pandas as pd
import numpy as np
import joblib
import json
from sklearn.metrics import precision_score, recall_score, f1_score

RUTA_PROYECTO = Path(__file__).parent.parent
RUTA_OUTPUT    = RUTA_PROYECTO / "data" / "output"
RUTA_MODELOS     = RUTA_PROYECTO / "models"

print("Cargando datos de prueba...")
X_test_path = RUTA_OUTPUT / "X_test.csv"
if not X_test_path.exists():
    print(f"Error: no se encontró '{X_test_path.name}'")
    exit()
try:
    X_test = pd.read_csv(X_test_path)
except Exception as e:
    print(f"Error: no se pudo leer X_test.csv: {str(e)}")
    exit()

X_test_clean = X_test.drop(columns=['timestamp'], errors='ignore')

# Ground truth artificial – suponemos fallo desde fila 500
fallo_inicio = 500
y_true = np.zeros(len(X_test))
y_true[fallo_inicio:] = 1  # Marcar como anómalo desde esta fila

resultados = {}

# Isolation Forest
try:
    isol_forest = joblib.load(RUTA_MODELOS / "model_isol_forest.joblib")
    scores_iso = isol_forest.score_samples(X_test_clean)
    
    for umbral in [-0.7, -0.5, -0.3]:
        preds_iso = np.where(scores_iso < umbral, 1, 0)
        resultados[f'Isolation Forest ({umbral})'] = {
            'scores': scores_iso.tolist(),
            'preds': preds_iso.tolist(),
            'precision': float(precision_score(y_true, preds_iso, average='binary', zero_division=0)),
            'recall': float(recall_score(y_true, preds_iso, average='binary', zero_division=0)),
            'f1': float(f1_score(y_true, preds_iso, average='binary', zero_division=0))
        }
    print("Isolation Forest evaluado.")
except Exception as e:
    print(f"Error Isolation Forest: {str(e)}")

# One-Class SVM
try:
    oc_svm = joblib.load(RUTA_MODELOS / "model_oc_svm.joblib")
    scores_ocsvm = oc_svm.score_samples(X_test_clean)

    for nu in [0.01, 0.05, 0.1]:
        model = joblib.load(RUTA_MODELOS / "model_oc_svm.joblib")
        model.set_params(nu=nu)
        preds_ocsvm = model.predict(X_test_clean)
        preds_ocsvm = np.where(preds_ocsvm == -1, 1, 0)

        resultados[f'One-Class SVM (nu={nu})'] = {
            'scores': scores_ocsvm.tolist(),
            'preds': preds_ocsvm.tolist(),
            'precision': float(precision_score(y_true, preds_ocsvm, average='binary', zero_division=0)),
            'recall': float(recall_score(y_true, preds_ocsvm, average='binary', zero_division=0)),
            'f1': float(f1_score(y_true, preds_ocsvm, average='binary', zero_division=0))
        }
    print("One-Class SVM evaluado.")
except Exception as e:
    print(f"Error One-Class SVM: {str(e)}")

# Local Outlier Factor (LOF)
try:
    lof = joblib.load(RUTA_MODELOS / "model_lof.joblib")
    scores_lof = lof.score_samples(X_test_clean)

    for contamination in [0.01, 0.05, 0.1]:
        model = joblib.load(RUTA_MODELOS / "model_lof.joblib")
        model.set_params(contamination=contamination, novelty=True)
        preds_lof = model.predict(X_test_clean)
        resultados[f'Local Outlier Factor (contam={contamination})'] = {
            'scores': scores_lof.tolist(),
            'preds': preds_lof.tolist(),
            'precision': float(precision_score(y_true, preds_lof, average='macro', zero_division=0)),
            'recall': float(recall_score(y_true, preds_lof, average='macro', zero_division=0)),
            'f1': float(f1_score(y_true, preds_lof, average='macro', zero_division=0))
        }
    print("LOF evaluado.")
except Exception as e:
    print(f"Error LOF: {str(e)}")

# KMeans
try:
    kmeans = joblib.load(RUTA_MODELOS / "model_kmeans.joblib")
    distances = kmeans.transform(X_test_clean).min(axis=1)

    for percentile in [90, 95, 99]:
        threshold = np.percentile(distances, percentile)
        preds_km = np.where(distances > threshold, 1, 0)

        resultados[f'KMeans (pct={percentile})'] = {
            'scores': distances.tolist(),
            'preds': preds_km.tolist(),
            'precision': float(precision_score(y_true, preds_km, average='binary', zero_division=0)),
            'recall': float(recall_score(y_true, preds_km, average='binary', zero_division=0)),
            'f1': float(f1_score(y_true, preds_km, average='binary', zero_division=0))
        }
    print("KMeans evaluado.")
except Exception as e:
    print(f"Error KMeans: {str(e)}")

# Guardar resultados
resultados_path = RUTA_OUTPUT / "predicciones_modelos_con_metricas.json"
with open(resultados_path, 'w') as f:
    json.dump(resultados, f)

print(f"\nPredicciones guardadas en '{resultados_path}'")
print("Incluyen distintos umbrales y métricas calculadas.")