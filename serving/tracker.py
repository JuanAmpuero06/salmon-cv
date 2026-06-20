import numpy as np

def compute_iou(box1, box2):
    """
    Calcula la Intersección sobre Unión (IoU) entre dos cajas delimitadoras.
    Formato de caja: [x1, y1, x2, y2]
    """
    x1_1, y1_1, x2_1, y2_1 = box1
    x1_2, y1_2, x2_2, y2_2 = box2
    
    # Intersección
    x1_i = max(x1_1, x1_2)
    y1_i = max(y1_1, y1_2)
    x2_i = min(x2_1, x2_2)
    y2_i = min(y2_1, y2_2)
    
    if x2_i < x1_i or y2_i < y1_i:
        return 0.0
        
    intersection_area = (x2_i - x1_i) * (y2_i - y1_i)
    
    # Unión
    area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
    area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
    union_area = area1 + area2 - intersection_area
    
    if union_area == 0.0:
        return 0.0
        
    return intersection_area / union_area

class Track:
    def __init__(self, track_id, box, class_id, score):
        self.track_id = track_id
        self.box = box  # [x1, y1, x2, y2]
        self.class_id = class_id
        self.score = score
        self.lost_count = 0
        self.counted = False
        self.history = [self.get_centroid()]

    def get_centroid(self):
        x1, y1, x2, y2 = self.box
        return (x1 + x2) / 2.0, (y1 + y2) / 2.0

    def update(self, box, score):
        self.box = box
        self.score = score
        self.lost_count = 0
        self.history.append(self.get_centroid())
        # Limitar historial para no consumir memoria infinita
        if len(self.history) > 30:
            self.history.pop(0)

class IoUTracker:
    def __init__(self, iou_threshold=0.3, max_lost_frames=10):
        self.iou_threshold = iou_threshold
        self.max_lost_frames = max_lost_frames
        self.next_track_id = 0
        self.tracks = []

    def update(self, detections):
        """
        Detections es una lista de diccionarios:
        [{"box": [x1, y1, x2, y2], "class_id": int, "score": float}, ...]
        """
        updated_tracks = []
        matched_detections = set()
        
        # 1. Intentar emparejar tracks activos con nuevas detecciones
        # Ordenamos los tracks de mayor a menor confianza
        self.tracks.sort(key=lambda t: t.score, reverse=True)
        
        for track in self.tracks:
            best_iou = -1.0
            best_det_idx = -1
            
            for idx, det in enumerate(detections):
                if idx in matched_detections:
                    continue
                if det["class_id"] != track.class_id:
                    continue  # Solo emparejar la misma especie
                    
                iou = compute_iou(track.box, det["box"])
                if iou > best_iou:
                    best_iou = iou
                    best_det_idx = idx
            
            if best_iou >= self.iou_threshold:
                # Emparejado
                det = detections[best_det_idx]
                track.update(det["box"], det["score"])
                updated_tracks.append(track)
                matched_detections.add(best_det_idx)
            else:
                # No emparejado en este frame
                track.lost_count += 1
                if track.lost_count <= self.max_lost_frames:
                    updated_tracks.append(track)
                    
        # 2. Registrar nuevas detecciones no emparejadas como nuevos tracks
        for idx, det in enumerate(detections):
            if idx not in matched_detections:
                new_track = Track(
                    track_id=self.next_track_id,
                    box=det["box"],
                    class_id=det["class_id"],
                    score=det["score"]
                )
                self.next_track_id += 1
                updated_tracks.append(new_track)
                
        self.tracks = updated_tracks
        return self.tracks
