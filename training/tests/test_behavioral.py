import pytest
from pathlib import Path
import cv2
import numpy as np
from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Buscar dinámicamente el mejor modelo en yolo_runs
def _get_best_model_path():
    yolo_runs_dir = PROJECT_ROOT / "data" / "06_models" / "yolo_runs"
    # Buscar primero el run configurado por defecto
    default_s_path = yolo_runs_dir / "baseline_yolov8s_cpu" / "weights" / "best.pt"
    if default_s_path.exists():
        return default_s_path
    # Buscar cualquier otro best.pt
    candidates = list(yolo_runs_dir.glob("*/weights/best.pt"))
    if candidates:
        return candidates[0]
    return default_s_path

MODEL_PATH = _get_best_model_path()

def _get_val_image_path():
    val_dir = PROJECT_ROOT / "data" / "05_model_input" / "yolo_salmon_pollock" / "images" / "val"
    if not val_dir.exists():
        return None
    candidates = sorted(list(val_dir.glob("*.PNG")) + list(val_dir.glob("*.png")) + list(val_dir.glob("*.jpg")))
    return candidates[0] if candidates else None

@pytest.fixture
def model():
    if not MODEL_PATH.exists():
        pytest.skip(f"El modelo no existe en {MODEL_PATH}. Corre primero el entrenamiento.")
    return YOLO(str(MODEL_PATH))

@pytest.fixture
def sample_image_path():
    img_path = _get_val_image_path()
    if img_path is None:
        pytest.skip("No se encontraron imágenes en la carpeta de validación para realizar los tests.")
    return img_path

def test_model_loaded(model):
    assert model is not None
    assert model.names[0] == "Salmon"
    assert model.names[1] == "Pollock"

def test_prediction_invariance_brightness(model, sample_image_path):
    """
    Test de Invarianza: Comprobar que el modelo sigue detectando peces al cambiar el brillo de la imagen
    (simulando diferentes condiciones de luz y turbidez en el agua).
    """
    img = cv2.imread(str(sample_image_path))
    h, w = img.shape[:2]
    
    # 1. Inferencia en la imagen original
    res_orig = model(img, conf=0.15, verbose=False)[0]
    boxes_orig = res_orig.boxes
    
    # Si la imagen original no contiene peces según el modelo, saltamos el test para evitar falso negativo
    if len(boxes_orig) == 0:
        pytest.skip("La imagen de prueba seleccionada no contiene detecciones en su estado original.")
        
    # 2. Modificar el brillo (aumentar +30)
    img_bright = cv2.convertScaleAbs(img, alpha=1.0, beta=30)
    res_bright = model(img_bright, conf=0.15, verbose=False)[0]
    boxes_bright = res_bright.boxes
    
    # 3. Simular turbidez mediante desenfoque gaussiano
    img_blurred = cv2.GaussianBlur(img, (5, 5), 0)
    res_blurred = model(img_blurred, conf=0.15, verbose=False)[0]
    boxes_blurred = res_blurred.boxes

    # Assert: El modelo debe seguir detectando al menos un pez bajo estas alteraciones moderadas
    assert len(boxes_bright) > 0, "El modelo perdió todas las detecciones al aumentar el brillo."
    assert len(boxes_blurred) > 0, "El modelo perdió todas las detecciones al aplicar desenfoque (turbidez)."
    
    # Comprobar que las clases predichas coincidan mayoritariamente
    orig_classes = set(boxes_orig.cls.cpu().numpy().astype(int))
    bright_classes = set(boxes_bright.cls.cpu().numpy().astype(int))
    blurred_classes = set(boxes_blurred.cls.cpu().numpy().astype(int))
    
    assert orig_classes.intersection(bright_classes), "Las clases predichas cambiaron completamente con el brillo."
    assert orig_classes.intersection(blurred_classes), "Las clases predichas cambiaron completamente con el desenfoque."

def test_prediction_invariance_flip(model, sample_image_path):
    """
    Test de Invarianza: Comprobar que al voltear la imagen horizontalmente (Horizontal Flip),
    el modelo sigue detectando peces y sus clases se mantienen consistentes.
    """
    img = cv2.imread(str(sample_image_path))
    
    # 1. Inferencia original
    res_orig = model(img, conf=0.15, verbose=False)[0]
    if len(res_orig.boxes) == 0:
        pytest.skip("La imagen de prueba seleccionada no contiene detecciones en su estado original.")
        
    # 2. Voltear imagen horizontalmente
    img_flipped = cv2.flip(img, 1)
    res_flipped = model(img_flipped, conf=0.15, verbose=False)[0]
    
    # Assert: Se deben mantener detecciones
    assert len(res_flipped.boxes) > 0, "El modelo perdió todas las detecciones al voltear la imagen horizontalmente."
    assert len(res_orig.boxes) == len(res_flipped.boxes), "El número de detecciones difiere al voltear la imagen."

def test_predictions_within_bounds(model, sample_image_path):
    """
    Test de Robustez (Boundary Check): Comprobar que todas las cajas delimitadoras predichas
    estén dentro de los límites válidos de la imagen (0 <= x1 < x2 <= w, 0 <= y1 < y2 <= h).
    """
    img = cv2.imread(str(sample_image_path))
    h, w = img.shape[:2]
    
    results = model(img, verbose=False)[0]
    boxes = results.boxes.xyxy.cpu().numpy()  # Coordenadas [x1, y1, x2, y2]
    
    for box in boxes:
        x1, y1, x2, y2 = box
        
        # Validar rangos absolutos
        assert 0 <= x1 < w, f"Coordenada x1 ({x1}) fuera de los límites de la imagen (ancho={w})."
        assert 0 <= y1 < h, f"Coordenada y1 ({y1}) fuera de los límites de la imagen (alto={h})."
        assert 0 < x2 <= w, f"Coordenada x2 ({x2}) fuera de los límites de la imagen (ancho={w})."
        assert 0 < y2 <= h, f"Coordenada y2 ({y2}) fuera de los límites de la imagen (alto={h})."
        
        # Validar consistencia interna
        assert x1 < x2, f"Caja delimitadora corrupta: x1 ({x1}) >= x2 ({x2})."
        assert y1 < y2, f"Caja delimitadora corrupta: y1 ({y1}) >= y2 ({y2})."
