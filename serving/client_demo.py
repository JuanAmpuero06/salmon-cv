import argparse
import sys
from pathlib import Path
import cv2
import requests
from tracker import IoUTracker

def run_tracking_client(input_source: str, api_url: str, output_path: str, line_position: float):
    # Inicializar el Tracker
    tracker = IoUTracker(iou_threshold=0.3, max_lost_frames=15)
    
    input_path = Path(input_source)
    is_dir = input_path.is_dir()
    
    # 1. Configurar la captura según si es directorio o archivo de video
    if is_dir:
        # Buscar imágenes ordenadas
        valid_extensions = {".png", ".jpg", ".jpeg", ".png", ".jpg", ".PNG", ".JPG", ".JPEG"}
        frame_files = sorted(
            [p for p in input_path.iterdir() if p.suffix in valid_extensions]
        )
        if not frame_files:
            print(f"Error: No se encontraron imágenes válidas en el directorio {input_source}")
            sys.exit(1)
            
        print(f"Abriendo secuencia de imágenes en directorio: {input_source}")
        # Leer primera imagen para dimensiones
        first_img = cv2.imread(str(frame_files[0]))
        if first_img is None:
            print(f"Error: No se pudo leer la primera imagen en {frame_files[0]}")
            sys.exit(1)
        height, width = first_img.shape[:2]
        fps = 10.0  # Asumimos 10 FPS por defecto para secuencias de imágenes
        total_frames = len(frame_files)
        print(f"Resolución de imágenes: {width}x{height} | Total de imágenes: {total_frames}")
    else:
        # Usar VideoCapture estándar
        cap = cv2.VideoCapture(input_source)
        if not cap.isOpened():
            print(f"Error: No se pudo abrir el video en {input_source}")
            sys.exit(1)
            
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print(f"Abriendo archivo de video: {input_source}")
        print(f"Resolución: {width}x{height} | FPS: {fps} | Total Frames: {total_frames}")
    
    # Configurar el escritor de video de salida
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # Posición de la línea de conteo (e.g., a la mitad de la pantalla)
    line_x = int(width * line_position)
    
    # Contadores
    counts = {
        "Salmon": 0,
        "Pollock": 0
    }
    
    # Colores para dibujar (BGR)
    colors = {
        "Salmon": (255, 0, 0),    # Azul
        "Pollock": (0, 255, 0),   # Verde
        "line": (0, 0, 255)       # Rojo para la línea
    }
    
    frame_idx = 0
    
    try:
        while True:
            # 2. Leer frame según la fuente
            if is_dir:
                if frame_idx >= len(frame_files):
                    break
                frame = cv2.imread(str(frame_files[frame_idx]))
                if frame is None:
                    print(f"[WARNING] No se pudo leer la imagen {frame_files[frame_idx]}")
                    frame_idx += 1
                    continue
                ret = True
            else:
                ret, frame = cap.read()
                if not ret:
                    break
            
            frame_idx += 1
            if frame_idx % 10 == 0 or frame_idx == 1:
                print(f"Procesando frame {frame_idx}/{total_frames}...")
                
            # Codificar el frame a JPEG
            _, img_encoded = cv2.imencode('.jpg', frame)
            files = {"file": ("frame.jpg", img_encoded.tobytes(), "image/jpeg")}
            
            # Consumir la API de detección
            try:
                response = requests.post(f"{api_url}/detect", files=files, timeout=5)
                response.raise_for_status()
                result = response.json()
                predictions = result["predictions"]
            except Exception as e:
                print(f"[WARNING] Error al conectar con la API en frame {frame_idx}: {e}")
                predictions = []
                
            # Formatear detecciones para el tracker
            detections = []
            for pred in predictions:
                detections.append({
                    "box": pred["box"],  # [x1, y1, x2, y2]
                    "class_id": pred["class_id"],
                    "score": pred["confidence"]
                })
                
            # Actualizar el tracker
            tracks = tracker.update(detections)
            
            # Lógica de conteo al cruzar la línea vertical
            for track in tracks:
                if len(track.history) < 2:
                    continue
                    
                prev_x, prev_y = track.history[-2]
                curr_x, curr_y = track.history[-1]
                
                # Nombre de la clase
                class_name = "Salmon" if track.class_id == 0 else "Pollock"
                
                # Detectar cruce de izquierda a derecha o derecha a izquierda
                if not track.counted:
                    # Cruce de izquierda a derecha
                    if prev_x < line_x <= curr_x:
                        counts[class_name] += 1
                        track.counted = True
                        print(f"[CONTEO] {class_name} ID {track.track_id} cruzó la línea (swimming right)!")
                    # Cruce de derecha a izquierda
                    elif prev_x > line_x >= curr_x:
                        counts[class_name] += 1
                        track.counted = True
                        print(f"[CONTEO] {class_name} ID {track.track_id} cruzó la línea (swimming left)!")
            
            # Dibujar elementos gráficos en el frame
            # 1. Dibujar la línea de conteo
            cv2.line(frame, (line_x, 0), (line_x, height), colors["line"], 3)
            cv2.putText(frame, "LINEA DE CONTEO", (line_x + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, colors["line"], 2)
            
            # 2. Dibujar tracks activos
            for track in tracks:
                if track.lost_count > 0:
                    continue
                    
                x1, y1, x2, y2 = [int(val) for track_val in track.box for val in [track_val]] if not isinstance(track.box[0], (list, tuple)) else [int(v) for v in track.box]
                # Desenredar coordenadas si es necesario
                x1, y1, x2, y2 = int(track.box[0]), int(track.box[1]), int(track.box[2]), int(track.box[3])
                class_name = "Salmon" if track.class_id == 0 else "Pollock"
                color = colors[class_name]
                
                # Dibujar bounding box
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                
                # Etiqueta
                label = f"{class_name} #{track.track_id} ({track.score:.2f})"
                cv2.putText(frame, label, (x1, max(0, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                
                # Dibujar centroide
                cx, cy = [int(val) for val in track.get_centroid()]
                cv2.circle(frame, (cx, cy), 4, color, -1)
                
            # 3. Dibujar marcador de conteo acumulado
            cv2.rectangle(frame, (10, 10), (320, 100), (0, 0, 0), -1)
            cv2.putText(frame, f"Salmones: {counts['Salmon']}", (20, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.8, colors["Salmon"], 2)
            cv2.putText(frame, f"Pollocks: {counts['Pollock']}", (20, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.8, colors["Pollock"], 2)
            
            # Guardar frame en el video de salida
            out.write(frame)
            
    finally:
        if not is_dir:
            cap.release()
        out.release()
        print("Procesamiento de video completado.")
        print(f"Video final guardado con éxito en: {output_path}")
        print("--- Conteo Final ---")
        print(f"Salmones contados: {counts['Salmon']}")
        print(f"Pollocks contados: {counts['Pollock']}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Cliente para simular streaming de video/secuencia de imágenes y tracking de peces.")
    parser.add_argument("--video", type=str, required=True, help="Ruta al video .mp4 o directorio de imágenes PNG/JPG.")
    parser.add_argument("--api-url", type=str, default="http://localhost:8000", help="URL base de la API de FastAPI.")
    parser.add_argument("--output", type=str, default="output_tracked.mp4", help="Ruta del video final con tracking.")
    parser.add_argument("--line", type=float, default=0.5, help="Posición relativa de la línea vertical (0.0 a 1.0).")
    
    args = parser.parse_args()
    
    run_tracking_client(args.video, args.api_url, args.output, args.line)
