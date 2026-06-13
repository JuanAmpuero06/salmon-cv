# Contexto del Proyecto: salmon-cv

Este archivo guarda el contexto del proyecto `salmon-cv` para futuros desarrolladores y asistentes de inteligencia artificial.

## 1. Propósito del Proyecto
Detección y clasificación automática de peces de las especies **Salmon** (Salmón) y **Pollock** (Abadejo de Alaska) en videos del NOAA (National Oceanic and Atmospheric Administration) utilizando modelos **YOLOv8** de `ultralytics`.

## 2. Tecnologías y Librerías Utilizadas
- **Framework Principal:** [Kedro](https://kedro.org/) (versión `1.2.0`) para la orquestación y estructuración de pipelines de datos y machine learning.
- **Modelo de Visión Artificial:** [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics) para entrenamiento de object detection.
- **Entorno de Programación:** Python 3.11 con un entorno virtual local (`.venv`).
- **Procesamiento de Imagen:** OpenCV (`opencv-python-headless`) y PyTorch (`torch`, `torchvision`).

## 3. Estructura del Proyecto
- **`training/`**: Proyecto Kedro con pipelines de preparación de datos, EDA y entrenamiento.
  - **`conf/base/`**: Parámetros de pipelines y catálogo de datos.
    - [catalog.yml](file:///C:/Users/jampu/Desktop/salmon-cv/training/conf/base/catalog.yml): Define los datasets del reporte (ej. manifiestos, resúmenes).
    - [parameters.yml](file:///C:/Users/jampu/Desktop/salmon-cv/training/conf/base/parameters.yml): Ruta raíz de datos NOAA (`data/01_raw/noaa_salmon_pollock_2019_2020/annotated_data`).
    - [parameters_data_preparation.yml](file:///C:/Users/jampu/Desktop/salmon-cv/training/conf/base/parameters_data_preparation.yml): Clases target y splits (`train: 0.7`, `val: 0.2`, `test: 0.1`).
    - [parameters_data_understanding.yml](file:///C:/Users/jampu/Desktop/salmon-cv/training/conf/base/parameters_data_understanding.yml): Configuración del EDA (ej. número de muestras a tomar).
    - [parameters_train.yml](file:///C:/Users/jampu/Desktop/salmon-cv/training/conf/base/parameters_train.yml): Configuración de épocas (10), batch size (4), device (`cpu`) y modelo de baseline (`yolov8n.pt`).
  - **`src/training/pipelines/`**:
    - **`data_understanding`**:
      - [nodes.py](file:///C:/Users/jampu/Desktop/salmon-cv/training/src/training/pipelines/data_understanding/nodes.py): Funciones para realizar inventario de los clips COCO y generar muestras visuales con bboxes (`data/08_reporting/eda_samples/`).
    - **`data_preparation`**:
      - [nodes.py](file:///C:/Users/jampu/Desktop/salmon-cv/training/src/training/pipelines/data_preparation/nodes.py): Convierte las cajas delimitadoras de COCO JSON (esquinas absolutas) a YOLO TXT (centro, ancho y alto normalizados), separando los clips en train, val y test. Genera el archivo `data.yaml`.
    - **`train`**:
      - [nodes.py](file:///C:/Users/jampu/Desktop/salmon-cv/training/src/training/pipelines/train/nodes.py): Entrena el detector YOLOv8 a partir de la ruta del `data.yaml` y guarda los checkpoints y reportes de entrenamiento. Soporta **auto-resumen** (resume=True): si el checkpoint existe, reanuda desde ahí; si no, inicia desde cero automáticamente sin lanzar error.
- **`serving/`**: Directorio preparado para el servidor de inferencia.
- **`shared/`**: Directorio para utilidades comunes compartidas entre training y serving.

## 4. Estado de los Datos
El dataset completo contiene 184 clips de video, que tras pasar por `data_preparation` equivalen a:
- **Imágenes exportadas:** 16,998
- **Etiquetas (Bounding Boxes):** 84,983
  - **Salmon:** 11,572 annotations
  - **Pollock:** 73,394 annotations

*Nota:* Los datos de imágenes y etiquetas están excluidos en Git (`data/**`). Para poder reejecutar el pipeline, se requiere descargar la fuente de datos NOAA en la ruta `training/data/01_raw/noaa_salmon_pollock_2019_2020/annotated_data/`.

## 5. Comandos Útiles
Para activar el entorno virtual y correr pipelines en Windows (PowerShell):
```powershell
# Activar el entorno virtual
.venv\Scripts\Activate.ps1

# Ejecutar el pipeline por defecto (ejecuta todos los registrados)
kedro run --pipeline=__default__

# Ejecutar un pipeline específico
kedro run --pipeline=data_understanding
kedro run --pipeline=data_preparation
kedro run --pipeline=train
```

## 6. Registro de Experimentos (Experiment Tracking)
Cada vez que finaliza un entrenamiento (pipeline `train`), el sistema extrae automáticamente la mejor época y las métricas de validación del archivo `results.csv` de YOLO y las guarda junto con los hiperparámetros utilizados en:
- **`training/data/08_reporting/train_history.json`**: Historial completo en formato estructurado JSON.
- **`training/data/08_reporting/train_history.csv`**: Historial en formato CSV plano, ideal para abrir en Excel o cargar con herramientas de análisis de datos para comparar qué hiperparámetros (épocas, batch size, imgsz, optimizer) arrojaron el mejor `mAP50` y menores pérdidas.

