import os
import sys
import argparse
import urllib.request
import urllib.parse
import xml.etree.ElementTree as ET
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

BUCKET_NAME = "nmfs_odp_afsc"
PREFIX = "RACE/MACE/salmon_pollock_object_detection/annotated_data/"
BASE_URL = f"https://storage.googleapis.com/{BUCKET_NAME}"

def get_project_root() -> Path:
    cwd = Path.cwd()
    # Si estamos en la raíz del repo salmon-cv, entrar a training
    if (cwd / "training").exists():
        return cwd / "training"
    # Si ya estamos en training
    if (cwd / "src" / "training").exists():
        return cwd
    return cwd

def list_gcs_files(prefix: str):
    print("Obteniendo lista de archivos desde el bucket público de NOAA GCS...")
    keys = []
    marker = ""
    ns = {'ns': 'http://doc.s3.amazonaws.com/2006-03-01'}
    
    while True:
        url = f"{BASE_URL}?prefix={urllib.parse.quote(prefix)}"
        if marker:
            url += f"&marker={urllib.parse.quote(marker)}"
            
        try:
            with urllib.request.urlopen(url) as response:
                xml_data = response.read()
            root = ET.fromstring(xml_data)
            contents = root.findall('ns:Contents', ns)
            if not contents:
                break
                
            for c in contents:
                key = c.find('ns:Key', ns).text
                size = int(c.find('ns:Size', ns).text)
                keys.append((key, size))
                
            is_truncated = root.find('ns:IsTruncated', ns)
            if is_truncated is not None and is_truncated.text == "true":
                next_marker = root.find('ns:NextMarker', ns)
                if next_marker is not None:
                    marker = next_marker.text
                else:
                    marker = keys[-1][0]
            else:
                break
        except Exception as e:
            print(f"Error al listar archivos: {e}")
            break
            
    return keys

def download_file(url: str, dest_path: Path, size_bytes: int):
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    if dest_path.exists() and dest_path.stat().st_size == size_bytes:
        # Ya existe y tiene el mismo tamaño
        return True, None
    try:
        urllib.request.urlretrieve(url, str(dest_path))
        return True, None
    except Exception as e:
        return False, str(e)

def parse_args():
    parser = argparse.ArgumentParser(description="Descarga el dataset de NOAA Salmon & Pollock.")
    parser.add_argument("--all", action="store_true", help="Descargar todos los clips del dataset.")
    parser.add_argument("--clips", type=int, default=None, help="Número de clips a descargar (para desarrollo/pruebas).")
    parser.add_argument("--workers", type=int, default=8, help="Número de hilos concurrentes para la descarga.")
    return parser.parse_args()

def main():
    args = parse_args()
    project_root = get_project_root()
    dest_root = project_root / "data" / "01_raw" / "noaa_salmon_pollock_2019_2020" / "annotated_data"
    
    # Listar todos los archivos
    all_files = list_gcs_files(PREFIX)
    if not all_files:
        print("No se pudieron obtener archivos del bucket. Abortando.")
        sys.exit(1)
        
    # Agrupar archivos por clip
    clips = {}
    for key, size in all_files:
        # El key tiene formato: RACE/MACE/salmon_pollock_object_detection/annotated_data/nombre_clip/file
        rel_parts = key.replace(PREFIX, "").split("/")
        if len(rel_parts) < 2:
            continue
        clip_name = rel_parts[0]
        clips.setdefault(clip_name, []).append((key, size))
        
    clip_names = sorted(clips.keys())
    total_clips = len(clip_names)
    print(f"Se encontraron {total_clips} clips en el dataset de NOAA.")
    
    # Determinar qué descargar
    n_clips_to_download = args.clips
    if not args.all and n_clips_to_download is None:
        # Modo interactivo básico si no se especifican argumentos
        print("\nOpciones de descarga:")
        print("1. Descargar todo el dataset (Atención: ~25 GB de datos)")
        print("2. Descargar solo un subset de prueba (ej. 2 clips)")
        print("3. Cancelar")
        try:
            choice = input("Elige una opción (1-3): ").strip()
            if choice == "1":
                args.all = True
            elif choice == "2":
                n_clips_to_download = int(input("¿Cuántos clips deseas descargar? (Recomendado: 1-5): ").strip())
            else:
                print("Descarga cancelada.")
                sys.exit(0)
        except Exception:
            print("Entrada no válida. Cancelando.")
            sys.exit(1)
            
    if args.all:
        selected_clips = clip_names
        print(f"Preparando la descarga completa de {total_clips} clips...")
    else:
        selected_clips = clip_names[:n_clips_to_download]
        print(f"Preparando la descarga de {len(selected_clips)} clips de prueba...")
        
    # Compilar lista de descargas
    download_list = []
    total_size = 0
    for clip in selected_clips:
        for key, size in clips[clip]:
            # Convertir key de GCS a ruta local
            rel_path = key.replace(PREFIX, "")
            local_path = dest_root / rel_path
            download_list.append((key, local_path, size))
            total_size += size
            
    print(f"Total de archivos a descargar: {len(download_list)} ({total_size / (1024*1024):.2f} MB)")
    
    # Descargar concurrentemente
    success_count = 0
    fail_count = 0
    
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {}
        for key, local_path, size in download_list:
            url = f"{BASE_URL}/{urllib.parse.quote(key)}"
            futures[executor.submit(download_file, url, local_path, size)] = (key, local_path)
            
        completed = 0
        total_files = len(download_list)
        for future in as_completed(futures):
            key, local_path = futures[future]
            success, err = future.result()
            completed += 1
            if success:
                success_count += 1
            else:
                fail_count += 1
                print(f"\n[ERROR] Falló la descarga de {key}: {err}")
                
            # Mostrar progreso básico
            if completed % 10 == 0 or completed == total_files:
                print(f"Progreso: {completed}/{total_files} archivos procesados...", end="\r")
                
    print(f"\n\nDescarga finalizada.")
    print(f"Archivos descargados con éxito: {success_count}")
    print(f"Errores: {fail_count}")
    print(f"Los datos se guardaron en: {dest_root.resolve()}")

if __name__ == "__main__":
    main()
