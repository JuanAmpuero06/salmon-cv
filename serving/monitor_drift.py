import os
import glob
import cv2
import numpy as np
from pathlib import Path
from scipy.stats import ks_2samp

# Directorios de comparación
SERVING_DIR = Path(__file__).resolve().parent
TRAIN_IMG_DIR = SERVING_DIR.parent / "training" / "data" / "05_model_input" / "yolo_salmon_pollock" / "images" / "train"
PRODUCTION_IMG_DIR = SERVING_DIR / "data" / "active_learning" # Imágenes de producción retenidas

def compute_image_stats(img_path):
    """
    Calcula estadísticas visuales básicas para la imagen:
    Brillo medio, contraste (desviación estándar) y saturación media (HSV).
    """
    img = cv2.imread(str(img_path))
    if img is None:
        return None
        
    # Redimensionar para acelerar cómputo
    img_resized = cv2.resize(img, (128, 128))
    
    # Brillo y contraste (escala de grises)
    gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
    mean_brightness = float(np.mean(gray))
    std_contrast = float(np.std(gray))
    
    # Saturación (HSV)
    hsv = cv2.cvtColor(img_resized, cv2.COLOR_BGR2HSV)
    mean_saturation = float(np.mean(hsv[:, :, 1]))
    
    return mean_brightness, std_contrast, mean_saturation

def run_drift_analysis():
    print("=== [Data Drift Monitor] Analizador de Desviacion de Datos ===")
    
    # 1. Recopilar muestras del dataset de entrenamiento (Referencia)
    ref_images = glob.glob(str(TRAIN_IMG_DIR / "*.jpg")) + glob.glob(str(TRAIN_IMG_DIR / "*.png")) + glob.glob(str(TRAIN_IMG_DIR / "*.PNG"))
    # Limitar a máximo 100 imágenes para un chequeo rápido
    ref_images = sorted(ref_images)[:100]
    
    if not ref_images:
        print(f"[ERROR] No se encontraron imágenes de referencia en: {TRAIN_IMG_DIR}")
        return
        
    # 2. Recopilar muestras del feed de producción (Actual)
    prod_images = glob.glob(str(PRODUCTION_IMG_DIR / "*.jpg"))
    if not prod_images:
        print(f"[INFO] No hay suficientes imágenes recolectadas en producción ({PRODUCTION_IMG_DIR.name}) para análisis de deriva.")
        print("Sugerencia: Permita que el sistema opere y guarde frames de baja confianza primero.")
        return
        
    print(f"Comparando {len(ref_images)} imágenes de entrenamiento frente a {len(prod_images)} imágenes de producción...")
    
    # Extraer estadísticas para referencia
    ref_stats = []
    for p in ref_images:
        stat = compute_image_stats(p)
        if stat:
            ref_stats.append(stat)
            
    # Extraer estadísticas para producción
    prod_stats = []
    for p in prod_images:
        stat = compute_image_stats(p)
        if stat:
            prod_stats.append(stat)
            
    if not ref_stats or not prod_stats:
        print("[ERROR] Error al procesar las estadísticas visuales.")
        return
        
    # Convertir a numpy arrays
    ref_arr = np.array(ref_stats)
    prod_arr = np.array(prod_stats)
    
    features = ["Brillo", "Contraste", "Saturación"]
    drift_detected = False
    
    report_lines = [
        "# 📊 Reporte de Monitoreo de Data Drift (Deriva de Datos)",
        f"Fecha del reporte: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "## Resumen Comparativo de Métricas Visuales",
        "| Métrica | Media Entrenamiento (Ref) | Media Producción (Actual) | P-Valor (KS-Test) | Estado |",
        "| :--- | :---: | :---: | :---: | :--- |"
    ]
    
    import time
    
    for idx, name in enumerate(features):
        ref_vals = ref_arr[:, idx]
        prod_vals = prod_arr[:, idx]
        
        ref_mean = np.mean(ref_vals)
        prod_mean = np.mean(prod_vals)
        
        # Test estadístico de Kolmogorov-Smirnov de 2 muestras
        # Si p_value < 0.05, rechazamos la hipótesis de que provienen de la misma distribución (Hay deriva!)
        ks_stat, p_val = ks_2samp(ref_vals, prod_vals)
        
        status = "✅ Estable"
        # Si el p-valor es menor a 0.05, indica que las distribuciones difieren estadísticamente
        if p_val < 0.05:
            status = "🚨 Deriva Detectada"
            drift_detected = True
            
        report_lines.append(f"| {name} | {ref_mean:.2f} | {prod_mean:.2f} | {p_val:.4f} | {status} |")
        
    report_lines.append("")
    if drift_detected:
        report_lines.append("> [!WARNING]")
        report_lines.append("> Se ha detectado una deriva estadística significativa en las características de las imágenes submarinas en producción.")
        report_lines.append("> Se recomienda recopilar más muestras, reetiquetarlas y ejecutar un nuevo reentrenamiento del modelo en Kedro.")
    else:
        report_lines.append("> [!NOTE]")
        report_lines.append("> Las características visuales de producción se mantienen consistentes con el dataset de entrenamiento.")
        
    # Guardar reporte en el directorio de producción
    report_path = SERVING_DIR / "data" / "drift_report.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(report_lines), encoding="utf-8")
    
    print(f"\nReporte de Data Drift generado exitosamente en: {report_path.resolve()}")
    if drift_detected:
        print("[ALERTA] ¡Deriva de datos detectada en condiciones subacuaticas! Reentrenamiento recomendado.")
    else:
        print("[ESTABLE] La distribucion de datos de produccion es estable.")

if __name__ == "__main__":
    import time
    run_drift_analysis()
