# UAV Predictive Maintenance – TFM


## Estructura del proyecto:

	uav-predictive-maintenance/
	│
	├── src/                     # Scripts Python
	│    ├── preprocesamiento.py
	│    ├── entrenamiento.py
	│    ├── evaluacion.py
	│    └── graficar_detecciones.py
	│    └── generar_comparativa.py
	│
	├── data/
	│    ├── input/              # Todas las secuencias de vuelo
	│    └── output/             # Resultados de la ejecución de los scripts (csv, json, png)
	│
	├── models/                  # Aquí se guardan los modelos entrenados (joblib)
	│
	├── README.md                # Instrucciones de instalación y ejecución
	│
	└── requirements.txt         # Librerías que hay que instalar


## Requisito - instalación de las librerías necesarias:

	pip install -r requirements.txt


## ¿Cómo ejecutar el proyecto? 

	cd src
	python preprocesamiento.py
	python entrenamiento.py
	python evaluacion.py
	python graficar_detecciones.py
	python generar_comparativa.py