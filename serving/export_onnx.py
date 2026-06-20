import sys
from pathlib import Path
from ultralytics import YOLO

def main():
    project_root = Path(__file__).resolve().parent.parent
    
    # Buscar dinámicamente el mejor modelo en yolo_runs
    def _get_best_model_path():
        yolo_runs_dir = project_root / "training" / "data" / "06_models" / "yolo_runs"
        default_s_path = yolo_runs_dir / "baseline_yolov8s_cpu" / "weights" / "best.pt"
        if default_s_path.exists():
            return default_s_path
        candidates = list(yolo_runs_dir.glob("*/weights/best.pt"))
        if candidates:
            return candidates[0]
        return default_s_path

    model_path = _get_best_model_path()
    
    if not model_path.exists():
        print(f"Error: No se encontró el modelo en {model_path}")
        sys.exit(1)
        
    print(f"Cargando modelo PyTorch desde {model_path}...")
    model = YOLO(str(model_path))
    
    output_dir = project_root / "serving" / "weights"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("Exportando modelo a formato ONNX...")
    # Exportar el modelo. YOLO guarda el archivo .onnx en el mismo directorio por defecto,
    # luego lo moveremos a la carpeta de serving/weights
    onnx_path_temp = model.export(format="onnx", imgsz=512, dynamic=True)
    
    if onnx_path_temp:
        temp_path = Path(onnx_path_temp)
        dest_path = output_dir / "model.onnx"
        if dest_path.exists():
            dest_path.unlink()
        temp_path.rename(dest_path)
        print(f"Modelo exportado con éxito y guardado en: {dest_path}")
    else:
        print("Error: Falló la exportación del modelo.")
        sys.exit(1)

if __name__ == "__main__":
    main()
