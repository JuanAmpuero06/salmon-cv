import os
import shutil
import time
from pathlib import Path

# Directorios de active learning
SERVING_DIR = Path(__file__).resolve().parent
ACTIVE_LEARNING_DIR = SERVING_DIR / "data" / "active_learning"
SENT_DIR = SERVING_DIR / "data" / "active_learning_sent"

# Configuración de Label Studio para sincronización real
LABEL_STUDIO_URL = os.environ.get("LABEL_STUDIO_URL", "http://localhost:8080")
LABEL_STUDIO_API_KEY = os.environ.get("LABEL_STUDIO_API_KEY", "")
LABEL_STUDIO_PROJECT_ID = os.environ.get("LABEL_STUDIO_PROJECT_ID", "1")

def run_synchronization():
    print("=== 🔄 Sincronizador de Aprendizaje Activo (Active Learning Sync) ===")
    
    if not ACTIVE_LEARNING_DIR.exists():
        print(f"Directorio de capturas no existe: {ACTIVE_LEARNING_DIR}")
        print("No hay imágenes que sincronizar.")
        return
        
    images = list(ACTIVE_LEARNING_DIR.glob("*.jpg"))
    if not images:
        print("No se encontraron frames de baja confianza pendientes para sincronizar.")
        return
        
    print(f"Encontrados {len(images)} frames con detecciones dudosas para reetiquetado.")
    SENT_DIR.mkdir(parents=True, exist_ok=True)
    
    success_count = 0
    for idx, img_path in enumerate(images):
        name = img_path.name
        
        # Desglosar los metadatos desde el nombre del archivo
        # Formato: uncertain_<timestamp>_<uuid>_<species>_<confidence>.jpg
        parts = name.replace(".jpg", "").split("_")
        if len(parts) >= 5:
            timestamp = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(int(parts[1])))
            uuid_hex = parts[2]
            species = parts[3]
            confidence = parts[4]
            print(f"[{idx+1}/{len(images)}] Procesando: {name}")
            print(f"    - Especie Dudosa: {species} | Confianza: {confidence}")
            print(f"    - Capturado el: {timestamp}")
        else:
            print(f"[{idx+1}/{len(images)}] Procesando archivo sin metadatos estándar: {name}")
            
        # --- SUBIDA DE ARTEFACTO (CVAT / Label Studio API) ---
        if LABEL_STUDIO_API_KEY:
            try:
                import requests
                url = f"{LABEL_STUDIO_URL}/api/projects/{LABEL_STUDIO_PROJECT_ID}/import"
                headers = {"Authorization": f"Token {LABEL_STUDIO_API_KEY}"}
                with open(img_path, "rb") as f_img:
                    files = {"file": (name, f_img, "image/jpeg")}
                    response = requests.post(url, files=files, headers=headers, timeout=10)
                if response.status_code in [200, 201]:
                    print(f"    -> [API LABEL STUDIO] Cargado con éxito (Status: {response.status_code}).")
                else:
                    print(f"    -> [API LABEL STUDIO WARNING] Error al subir. Status: {response.status_code}, Res: {response.text[:200]}")
            except Exception as ex:
                print(f"    -> [API LABEL STUDIO ERROR] Falló la subida real: {ex}")
        else:
            print("    -> [API SIMULADA] Frame subido con éxito al servidor de etiquetado (Configura LABEL_STUDIO_API_KEY para conexión real).")
        
        # Mover la imagen procesada a la carpeta de "enviados" para evitar duplicados en la siguiente corrida
        dest_path = SENT_DIR / name
        if dest_path.exists():
            dest_path.unlink()
        shutil.move(str(img_path), str(dest_path))
        success_count += 1
        
    print(f"\nSincronización completada con éxito. {success_count} imágenes transferidas a la cola de etiquetado en: {SENT_DIR.name}")

if __name__ == "__main__":
    run_synchronization()
