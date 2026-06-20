import cv2
import numpy as np
from pathlib import Path

def main():
    project_root = Path(__file__).resolve().parent
    train_dir = project_root / "data" / "05_model_input" / "yolo_salmon_pollock" / "images" / "train"
    raw_root = project_root / "data" / "01_raw" / "noaa_salmon_pollock_2019_2020" / "annotated_data"
    output_dir = project_root / "data" / "08_reporting" / "underwater_aug_comparisons"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if not train_dir.exists():
        print(f"Error: No existe el directorio de imágenes aumentadas: {train_dir}")
        return
        
    # Buscar imágenes en el dataset de entrenamiento
    candidates = sorted(list(train_dir.glob("*.PNG")) + list(train_dir.glob("*.png")) + list(train_dir.glob("*.jpg")))
    if not candidates:
        print("Error: No se encontraron imágenes aumentadas en train.")
        return
        
    print(f"Generando imágenes comparativas en {output_dir}...")
    
    # Seleccionar unas pocas imágenes para la comparación (por ejemplo, cada 20 imágenes para tener variedad)
    count = 0
    for img_path in candidates:
        name = img_path.name
        if "__" not in name:
            continue
            
        # Desglosar clip_name e original_image_name
        parts = name.split("__")
        clip_name = parts[0]
        original_name = parts[1]
        
        # Ruta de la imagen original
        orig_img_path = raw_root / clip_name / "frames" / original_name
        
        if orig_img_path.exists():
            orig_img = cv2.imread(str(orig_img_path))
            aug_img = cv2.imread(str(img_path))
            
            if orig_img is not None and aug_img is not None:
                # Si las resoluciones difieren (por ejemplo, si se redimensionó,
                # aunque prepare_yolo_dataset no redimensiona, mantiene el tamaño original),
                # nos aseguramos de que sean iguales
                if orig_img.shape != aug_img.shape:
                    orig_img = cv2.resize(orig_img, (aug_img.shape[1], aug_img.shape[0]))
                    
                # Crear imagen lado a lado (Side-by-side)
                h, w = orig_img.shape[:2]
                
                # Agregar etiquetas de texto en las imágenes
                cv2.putText(orig_img, "ORIGINAL", (50, 80), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 0, 255), 5)
                cv2.putText(aug_img, "AUMENTADA (CLAHE + Turbidez + Copy-Paste)", (50, 80), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 255, 0), 5)
                
                comparison = np.hstack((orig_img, aug_img))
                
                # Redimensionar la comparación final para que quepa bien en los visores
                comp_resized = cv2.resize(comparison, (1920, 540))
                
                out_path = output_dir / f"comparison_{count + 1}.jpg"
                cv2.imwrite(str(out_path), comp_resized)
                print(f"Comparación {count + 1} guardada: {out_path.name}")
                
                count += 1
                if count >= 3:  # Generar 3 comparaciones
                    break

    print("Imágenes comparativas generadas con éxito.")

if __name__ == "__main__":
    main()
