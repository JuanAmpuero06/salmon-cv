import io
import time
from pathlib import Path
from typing import List, Dict, Any
import os
import uuid
import tempfile
import shutil

import cv2
import numpy as np
import onnxruntime as ort
from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks, WebSocket, WebSocketDisconnect
from pydantic import BaseModel
from tracker import IoUTracker

app = FastAPI(
    title="Salmon-CV Fish Detection API",
    description="Microservicio local de inferencia de costo cero para detectar Salmon y Pollock usando YOLOv8 en ONNX Runtime.",
    version="1.0.0"
)

# Definir la ruta del modelo
PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / "serving" / "weights" / "model.onnx"

# Nombres de las clases
CLASS_NAMES = {
    0: "Salmon",
    1: "Pollock"
}

# Inicializar sesión de ONNX Runtime
session = None
input_name = None
output_names = None
imgsz = 512

ACTIVE_LEARNING_DIR = PROJECT_ROOT / "serving" / "data" / "active_learning"

def check_and_save_uncertain_frame(img_bgr, predictions):
    """
    Guarda de forma preventiva los frames donde el modelo tiene dudas (confianza entre 0.35 y 0.55)
    para el ciclo de Aprendizaje Activo (Active Learning).
    """
    if img_bgr is None:
        return
        
    for pred in predictions:
        if 0.35 <= pred.confidence <= 0.55:
            # Crear directorio si no existe
            ACTIVE_LEARNING_DIR.mkdir(parents=True, exist_ok=True)
            
            # Limitar a máximo 200 imágenes para evitar llenar el disco
            existing_files = list(ACTIVE_LEARNING_DIR.glob("*.jpg"))
            if len(existing_files) >= 200:
                break
                
            # Generar nombre único
            timestamp = int(time.time())
            unique_id = uuid.uuid4().hex[:6]
            conf_str = f"{pred.confidence:.2f}"
            filename = f"uncertain_{timestamp}_{unique_id}_{pred.class_name}_{conf_str}.jpg"
            filepath = ACTIVE_LEARNING_DIR / filename
            
            # Guardar imagen original
            cv2.imwrite(str(filepath), img_bgr)
            print(f"[ACTIVE LEARNING] Capturado frame de baja confianza ({pred.confidence:.4f}) para reetiquetar: {filename}")
            break # Solo guardar una imagen por frame para evitar duplicidad masiva


@app.on_event("startup")
def load_model():
    global session, input_name, output_names
    if not MODEL_PATH.exists():
        print(f"[ERROR] No se encontró el modelo ONNX en: {MODEL_PATH}")
        return
        
    print(f"Cargando sesión de ONNX Runtime desde {MODEL_PATH}...")
    # Ejecución en CPU
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 4  # Optimizar para multithreading en CPU
    session = ort.InferenceSession(str(MODEL_PATH), opts, providers=['CPUExecutionProvider'])
    
    inputs = session.get_inputs()
    input_name = inputs[0].name
    
    outputs = session.get_outputs()
    output_names = [o.name for o in outputs]
    print("Modelo ONNX cargado exitosamente.")

class Prediction(BaseModel):
    class_id: int
    class_name: str
    confidence: float
    box: List[float]  # [x1, y1, x2, y2] en coordenadas originales de la imagen

class DetectionResponse(BaseModel):
    predictions: List[Prediction]
    inference_time_ms: float

def preprocess(img_bgr, target_size=512):
    """
    Preprocesa la imagen de entrada para YOLOv8.
    Retorna la imagen preprocesada (float32, [1, 3, target_size, target_size])
    y la relación de escala/offsets para reescalar las cajas de vuelta.
    """
    h_orig, w_orig = img_bgr.shape[:2]
    
    # Redimensionar manteniendo la relación de aspecto (Letterbox)
    r = target_size / max(h_orig, w_orig)
    new_unpad = (int(round(w_orig * r)), int(round(h_orig * r)))
    
    # Redimensionar
    if (w_orig, h_orig) != new_unpad:
        img_resized = cv2.resize(img_bgr, new_unpad, interpolation=cv2.INTER_LINEAR)
    else:
        img_resized = img_bgr.copy()
        
    # Calcular relleno (padding)
    dw = target_size - new_unpad[0]
    dh = target_size - new_unpad[1]
    
    # Rellenar uniformemente por ambos lados
    top, bottom = dh // 2, dh - (dh // 2)
    left, right = dw // 2, dw - (dw // 2)
    
    # Agregar borde gris (114, 114, 114 es el estándar en YOLO)
    img_padded = cv2.copyMakeBorder(
        img_resized, top, bottom, left, right, 
        cv2.BORDER_CONSTANT, value=(114, 114, 114)
    )
    
    # Conversión BGR -> RGB y Normalización a [0.0, 1.0]
    img_rgb = cv2.cvtColor(img_padded, cv2.COLOR_BGR2RGB)
    img_data = img_rgb.astype(np.float32) / 255.0
    
    # Transposición: HWC -> CHW y agregar batch dimension: BCHW
    img_data = np.transpose(img_data, (2, 0, 1))
    img_data = np.expand_dims(img_data, axis=0)
    
    return img_data, r, (left, top)

def postprocess(outputs, r, pad, orig_shape, conf_threshold=0.25, iou_threshold=0.45):
    """
    Procesa las salidas de YOLOv8 y aplica Non-Maximum Suppression (NMS).
    """
    predictions = []
    # Output tensor shape: (1, 6, 5376) -> (6, 5376)
    output = np.squeeze(outputs[0])
    
    # Transponer a (5376, 6) para que cada fila sea una detección
    # Cada fila: [x_center, y_center, w, h, score_class0, score_class1]
    output = output.T
    
    boxes = []
    confidences = []
    class_ids = []
    
    left_pad, top_pad = pad
    h_orig, w_orig = orig_shape
    
    for row in output:
        # Puntuaciones de clase están de la columna 4 en adelante
        scores = row[4:]
        class_id = np.argmax(scores)
        confidence = scores[class_id]
        
        if confidence >= conf_threshold:
            # Coordenadas en la imagen de 512x512
            xc, yc, w, h = row[0], row[1], row[2], row[3]
            
            # Convertir a [x1, y1, x2, y2] en el espacio padding de 512
            x1 = xc - w / 2.0
            y1 = yc - h / 2.0
            x2 = xc + w / 2.0
            y2 = yc + h / 2.0
            
            # Remover el padding
            x1_unpad = x1 - left_pad
            y1_unpad = y1 - top_pad
            x2_unpad = x2 - left_pad
            y2_unpad = y2 - top_pad
            
            # Reescalar a las dimensiones originales de la imagen
            x1_orig = x1_unpad / r
            y1_orig = y1_unpad / r
            x2_orig = x2_unpad / r
            y2_orig = y2_unpad / r
            
            # Limitar dentro de la imagen original
            x1_orig = max(0, min(x1_orig, w_orig))
            y1_orig = max(0, min(y1_orig, h_orig))
            x2_orig = max(0, min(x2_orig, w_orig))
            y2_orig = max(0, min(y2_orig, h_orig))
            
            boxes.append([int(x1_orig), int(y1_orig), int(x2_orig - x1_orig), int(y2_orig - y1_orig)])
            confidences.append(float(confidence))
            class_ids.append(int(class_id))
            
    # Aplicar Non-Maximum Suppression (NMS) usando OpenCV
    indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, iou_threshold)
    
    if len(indices) > 0:
        for i in indices.flatten():
            box = boxes[i]
            x, y, w, h = box
            predictions.append(
                Prediction(
                    class_id=class_ids[i],
                    class_name=CLASS_NAMES.get(class_ids[i], "Desconocido"),
                    confidence=confidences[i],
                    box=[x, y, x + w, y + h]
                )
            )
            
    return predictions

@app.get("/health")
def health_check():
    if session is None:
        return {"status": "unhealthy", "error": "El modelo ONNX no está cargado."}
    return {"status": "healthy", "model": str(MODEL_PATH.name)}

@app.post("/detect", response_model=DetectionResponse)
async def detect_fish(file: UploadFile = File(...)):
    if session is None:
        raise HTTPException(status_code=503, detail="El modelo ONNX no está inicializado.")
        
    try:
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            raise HTTPException(status_code=400, detail="El archivo enviado no es una imagen válida.")
            
        orig_shape = img.shape[:2]
        
        # Preprocesar
        start_time = time.time()
        img_data, scale_r, padding = preprocess(img, imgsz)
        
        # Inferencia ONNX
        outputs = session.run(output_names, {input_name: img_data})
        
        # Postprocesar
        predictions = postprocess(outputs, scale_r, padding, orig_shape)
        
        # --- FASE 4: APRENDIZAJE ACTIVO ---
        check_and_save_uncertain_frame(img, predictions)
        
        inference_time = (time.time() - start_time) * 1000
        
        return DetectionResponse(
            predictions=predictions,
            inference_time_ms=round(inference_time, 2)
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error interno al procesar inferencia: {e}")

# --- FASE 2: ESCALABILIDAD Y SERVIDO ---

# Base de datos en memoria para tareas de procesamiento de video
tasks_db: Dict[str, Dict[str, Any]] = {}

def process_video_task(task_id: str, temp_video_path: str):
    global session, input_name, output_names
    try:
        tasks_db[task_id]["status"] = "processing"
        
        cap = cv2.VideoCapture(temp_video_path)
        if not cap.isOpened():
            raise ValueError("No se pudo abrir el archivo de video.")
            
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames <= 0:
            total_frames = 1
            
        tracker = IoUTracker(iou_threshold=0.3, max_lost_frames=15)
        unique_salmon_ids = set()
        unique_pollock_ids = set()
        
        frame_idx = 0
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            frame_idx += 1
            orig_shape = frame.shape[:2]
            
            # Preprocesar
            img_data, scale_r, padding = preprocess(frame, imgsz)
            
            # Inferencia ONNX
            outputs = session.run(output_names, {input_name: img_data})
            
            # Postprocesar
            predictions = postprocess(outputs, scale_r, padding, orig_shape, conf_threshold=0.25)
            
            # Formatear para el tracker
            detections = []
            for pred in predictions:
                detections.append({
                    "box": pred.box,
                    "class_id": pred.class_id,
                    "score": pred.confidence
                })
                
            # Actualizar tracker
            tracks = tracker.update(detections)
            
            # Registrar IDs únicos
            for t in tracks:
                if t.class_id == 0:
                    unique_salmon_ids.add(t.track_id)
                elif t.class_id == 1:
                    unique_pollock_ids.add(t.track_id)
            
            # Actualizar progreso
            if frame_idx % 10 == 0 or frame_idx == total_frames:
                progress = int((frame_idx / total_frames) * 100)
                tasks_db[task_id]["progress"] = min(progress, 99)
                
        cap.release()
        
        tasks_db[task_id]["status"] = "completed"
        tasks_db[task_id]["progress"] = 100
        tasks_db[task_id]["results"] = {
            "total_frames_processed": frame_idx,
            "counts": {
                "Salmon": len(unique_salmon_ids),
                "Pollock": len(unique_pollock_ids)
            }
        }
    except Exception as e:
        tasks_db[task_id]["status"] = "failed"
        tasks_db[task_id]["error"] = str(e)
    finally:
        if os.path.exists(temp_video_path):
            try:
                os.remove(temp_video_path)
            except Exception:
                pass

@app.post("/detect/video")
async def detect_fish_video(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    if session is None:
        raise HTTPException(status_code=503, detail="El modelo ONNX no está inicializado.")
        
    filename = file.filename
    if not filename.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
        raise HTTPException(status_code=400, detail="Formato de video no soportado. Suba MP4, AVI, MOV o MKV.")
        
    try:
        fd, temp_path = tempfile.mkstemp(suffix=".mp4")
        os.close(fd)
        
        with open(temp_path, "wb") as f:
            shutil.copyfileobj(file.file, f)
            
        task_id = str(uuid.uuid4())
        tasks_db[task_id] = {
            "status": "queued",
            "progress": 0,
            "results": None,
            "filename": filename
        }
        
        background_tasks.add_task(process_video_task, task_id, temp_path)
        return {"task_id": task_id, "status": "queued"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"No se pudo encolar la tarea: {e}")

@app.get("/task/{task_id}")
async def get_task_status(task_id: str):
    if task_id not in tasks_db:
        raise HTTPException(status_code=404, detail="Tarea no encontrada.")
    return tasks_db[task_id]

@app.websocket("/detect/ws")
async def detect_fish_ws(websocket: WebSocket):
    await websocket.accept()
    tracker = IoUTracker(iou_threshold=0.3, max_lost_frames=15)
    unique_salmon_ids = set()
    unique_pollock_ids = set()
    
    try:
        while True:
            data = await websocket.receive_bytes()
            nparr = np.frombuffer(data, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if img is None:
                await websocket.send_json({"error": "Imagen inválida."})
                continue
                
            orig_shape = img.shape[:2]
            img_data, scale_r, padding = preprocess(img, imgsz)
            outputs = session.run(output_names, {input_name: img_data})
            predictions = postprocess(outputs, scale_r, padding, orig_shape)
            
            # --- FASE 4: APRENDIZAJE ACTIVO ---
            check_and_save_uncertain_frame(img, predictions)
            
            detections = []
            for pred in predictions:
                detections.append({
                    "box": pred.box,
                    "class_id": pred.class_id,
                    "score": pred.confidence
                })
                
            tracks = tracker.update(detections)
            
            active_tracks_res = []
            for t in tracks:
                if t.lost_count == 0:
                    active_tracks_res.append({
                        "track_id": t.track_id,
                        "class_name": CLASS_NAMES.get(t.class_id, "Desconocido"),
                        "confidence": float(t.score),
                        "box": [float(v) for v in t.box]
                    })
                if t.class_id == 0:
                    unique_salmon_ids.add(t.track_id)
                elif t.class_id == 1:
                    unique_pollock_ids.add(t.track_id)
                    
            await websocket.send_json({
                "tracks": active_tracks_res,
                "counts": {
                    "Salmon": len(unique_salmon_ids),
                    "Pollock": len(unique_pollock_ids)
                }
            })
    except WebSocketDisconnect:
        pass
    except Exception as e:
        try:
            await websocket.send_json({"error": str(e)})
        except Exception:
            pass

# --- ENDPOINTS ADICIONALES PARA DASHBOARD Y ACTIVE LEARNING ---
from fastapi.responses import HTMLResponse, FileResponse
import contextlib
import io as python_io

@app.get("/", response_class=HTMLResponse)
async def get_dashboard():
    dashboard_path = Path(__file__).resolve().parent / "dashboard.html"
    if not dashboard_path.exists():
        raise HTTPException(status_code=404, detail="Dashboard HTML no encontrado.")
    return HTMLResponse(content=dashboard_path.read_text(encoding="utf-8"))

@app.get("/active-learning/images")
async def list_active_learning_images():
    if not ACTIVE_LEARNING_DIR.exists():
        return []
    # Ordenar por fecha de modificación descendente
    files = sorted(list(ACTIVE_LEARNING_DIR.glob("*.jpg")), key=os.path.getmtime, reverse=True)
    images_list = []
    for f in files:
        parts = f.name.replace(".jpg", "").split("_")
        species = "Desconocido"
        confidence = "0.0"
        if len(parts) >= 5:
            species = parts[3]
            confidence = parts[4]
        images_list.append({
            "name": f.name,
            "species": species,
            "confidence": confidence,
            "url": f"/active-learning/image/{f.name}"
        })
    return images_list

@app.get("/active-learning/image/{filename}")
async def get_active_learning_image(filename: str):
    filepath = ACTIVE_LEARNING_DIR / filename
    if not filepath.exists():
        raise HTTPException(status_code=404, detail="Imagen no encontrada.")
    return FileResponse(str(filepath))

@app.post("/active-learning/sync")
async def trigger_active_learning_sync():
    try:
        from active_learning_sync import run_synchronization
        f = python_io.StringIO()
        with contextlib.redirect_stdout(f):
            run_synchronization()
        logs = f.getvalue()
        return {"status": "success", "logs": logs}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
