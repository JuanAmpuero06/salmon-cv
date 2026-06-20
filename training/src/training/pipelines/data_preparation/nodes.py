from __future__ import annotations

import json
import random
import shutil
from pathlib import Path
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np


def _validate_coco_schema(coco: Dict[str, Any], path: Path):
    """
    Valida que el archivo JSON COCO tenga la estructura mínima requerida para Salmon-CV.
    """
    for key in ["images", "annotations", "categories"]:
        if key not in coco:
            raise ValueError(f"Esquema COCO inválido en {path.name}: Falta la clave '{key}'.")
            
    # Validar categorías
    categories = coco["categories"]
    cat_names = {cat.get("name") for cat in categories if "name" in cat}
    required_classes = {"Salmon", "Pollock"}
    if not required_classes.intersection(cat_names):
        print(f"[WARNING] El archivo {path.name} no contiene las clases esperadas ('Salmon', 'Pollock'). Categorías encontradas: {cat_names}")

    # Validar imágenes
    for img in coco["images"]:
        for field in ["id", "file_name", "width", "height"]:
            if field not in img:
                raise ValueError(f"Esquema COCO inválido en {path.name} (Imagen ID: {img.get('id', 'desconocido')}): Falta el campo '{field}'.")

    # Validar anotaciones
    for ann in coco["annotations"]:
        for field in ["id", "image_id", "category_id", "bbox"]:
            if field not in ann:
                raise ValueError(f"Esquema COCO inválido en {path.name} (Anotación ID: {ann.get('id', 'desconocido')}): Falta el campo '{field}'.")
        bbox = ann["bbox"]
        if not isinstance(bbox, list) or len(bbox) != 4:
            raise ValueError(f"Esquema COCO inválido en {path.name} (Anotación ID: {ann.get('id')}): 'bbox' debe ser una lista de 4 elementos. Encontrado: {bbox}")


def _load_coco(coco_json_path: Path) -> Dict[str, Any]:
    with coco_json_path.open("r", encoding="utf-8") as f:
        coco_data = json.load(f)
    _validate_coco_schema(coco_data, coco_json_path)
    return coco_data


def _index_coco(
    coco: Dict[str, Any]
) -> Tuple[Dict[int, Dict[str, Any]], Dict[int, List[Dict[str, Any]]], Dict[int, str]]:
    images_by_id = {img["id"]: img for img in coco.get("images", [])}

    ann_by_image_id: Dict[int, List[Dict[str, Any]]] = {}
    for ann in coco.get("annotations", []):
        ann_by_image_id.setdefault(ann["image_id"], []).append(ann)

    cat_name_by_id = {cat["id"]: cat.get("name", str(cat["id"])) for cat in coco.get("categories", [])}
    return images_by_id, ann_by_image_id, cat_name_by_id


def _resolve_image_path(frames_dir: Path, file_name: str | None) -> Path | None:
    if not file_name:
        return None

    base = Path(file_name).name
    p = frames_dir / base
    if p.exists():
        return p

    stem = Path(base).stem
    candidates = list(frames_dir.glob(stem + ".*"))
    if candidates:
        return candidates[0]

    return None


def _coco_bbox_to_yolo(
    bbox_xywh: List[float],
    img_width: int,
    img_height: int,
) -> Tuple[float, float, float, float]:
    x, y, w, h = bbox_xywh
    x_center = x + (w / 2.0)
    y_center = y + (h / 2.0)

    return (
        x_center / img_width,
        y_center / img_height,
        w / img_width,
        h / img_height,
    )


def _discover_clip_jsons(annotated_root: Path) -> List[Path]:
    clip_jsons: List[Path] = []
    for clip_dir in sorted(p for p in annotated_root.iterdir() if p.is_dir()):
        json_candidates = sorted(clip_dir.glob("*.json"))
        if json_candidates:
            clip_jsons.append(json_candidates[0])
    return clip_jsons


def _split_clip_jsons(
    clip_jsons: List[Path],
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> Dict[str, List[Path]]:
    if abs((train_ratio + val_ratio + test_ratio) - 1.0) > 1e-6:
        raise ValueError("Los porcentajes de split deben sumar 1.0")

    items = clip_jsons[:]
    random.seed(seed)
    random.shuffle(items)

    n = len(items)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    n_test = n - n_train - n_val

    return {
        "train": items[:n_train],
        "val": items[n_train:n_train + n_val],
        "test": items[n_train + n_val:n_train + n_val + n_test],
    }


def _build_salmon_crop_bank(clip_jsons, annotated_root, frames_dirname):
    """
    Recopila recortes de salmones de todas las imágenes de entrenamiento para el Copy-Paste sintético.
    """
    salmon_bank = []
    print("[INFO] Recopilando banco de salmones de forma dinámica...")
    for coco_path in clip_jsons:
        coco = _load_coco(coco_path)
        images_by_id, ann_by_image_id, cat_name_by_id = _index_coco(coco)
        clip_dir = coco_path.parent
        frames_dir = clip_dir / frames_dirname
        
        for img_info in images_by_id.values():
            anns = ann_by_image_id.get(img_info["id"], [])
            for ann in anns:
                cat_id = ann.get("category_id")
                class_name = cat_name_by_id.get(cat_id, str(cat_id))
                
                if class_name == "Salmon":
                    bbox = ann.get("bbox")
                    if bbox and len(bbox) == 4:
                        file_name = img_info.get("file_name")
                        img_path = _resolve_image_path(frames_dir, file_name)
                        if img_path and img_path.exists():
                            img = cv2.imread(str(img_path))
                            if img is not None:
                                x, y, w, h = [int(round(v)) for v in bbox]
                                h_img, w_img = img.shape[:2]
                                x1 = max(0, x)
                                y1 = max(0, y)
                                x2 = min(w_img, x + w)
                                y2 = min(h_img, y + h)
                                
                                if (x2 - x1) > 15 and (y2 - y1) > 15:
                                    crop = img[y1:y2, x1:x2].copy()
                                    salmon_bank.append(crop)
    print(f"[INFO] Recopilados {len(salmon_bank)} recortes de salmón para el Copy-Paste sintético.")
    return salmon_bank

def _box_overlap(box1, box2):
    """
    Calcula la superposición entre dos cajas en formato absolute [x1, y1, x2, y2].
    """
    x1_1, y1_1, x2_1, y2_1 = box1
    x1_2, y1_2, x2_2, y2_2 = box2
    
    x1_i = max(x1_1, x1_2)
    y1_i = max(y1_1, y1_2)
    x2_i = min(x2_1, x2_2)
    y2_i = min(y2_1, y2_2)
    
    if x2_i <= x1_i or y2_i <= y1_i:
        return 0.0
        
    inter = (x2_i - x1_i) * (y2_i - y1_i)
    area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
    area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
    return inter / min(area1, area2)

def prepare_yolo_dataset(
    annotated_data_root_relpath: str,
    output_yolo_root_relpath: str,
    target_classes: List[str],
    class_to_idx: Dict[str, int],
    split: Dict[str, float],
    seed: int,
    frames_dirname: str,
    copy_images: bool,
) -> Dict[str, Any]:
    project_root = Path.cwd()

    annotated_root = project_root / annotated_data_root_relpath
    output_root = project_root / output_yolo_root_relpath

    if not annotated_root.exists():
        raise FileNotFoundError(f"No existe annotated_data: {annotated_root}")

    clip_jsons = _discover_clip_jsons(annotated_root)
    if not clip_jsons:
        raise ValueError("No se encontraron archivos JSON de clips.")

    # Limpiar directorios previos para evitar acumulación de archivos viejos
    if (output_root / "images").exists():
        shutil.rmtree(output_root / "images")
    if (output_root / "labels").exists():
        shutil.rmtree(output_root / "labels")

    # crear estructura YOLO
    for split_name in ["train", "val", "test"]:
        (output_root / "images" / split_name).mkdir(parents=True, exist_ok=True)
        (output_root / "labels" / split_name).mkdir(parents=True, exist_ok=True)

    exported_images = 0
    exported_labels = 0
    filtered_out_annotations = 0
    empty_label_files = 0

    split_stats: Dict[str, Dict[str, Any]] = {
        "train": {"clips": 0, "images": 0, "labels": 0},
        "val": {"clips": 0, "images": 0, "labels": 0},
        "test": {"clips": 0, "images": 0, "labels": 0},
    }

    train_ratio = split["train"]
    val_ratio = split["val"]
    test_ratio = split["test"]

    if abs((train_ratio + val_ratio + test_ratio) - 1.0) > 1e-6:
        raise ValueError("Los porcentajes de split deben sumar 1.0")

    # Recopilar el banco de salmones de entrenamiento
    salmon_bank = _build_salmon_crop_bank(clip_jsons, annotated_root, frames_dirname)
    random.seed(seed)

    # Procesar cada clip y dividir sus frames cronológicamente
    for coco_path in clip_jsons:
        coco = _load_coco(coco_path)
        images_by_id, ann_by_image_id, cat_name_by_id = _index_coco(coco)

        clip_dir = coco_path.parent
        frames_dir = clip_dir / frames_dirname

        # Obtener y ordenar las imágenes cronológicamente
        image_items = list(images_by_id.values())
        image_items.sort(key=lambda x: x.get("file_name", ""))

        n_images = len(image_items)
        n_train = int(n_images * train_ratio)
        n_val = int(n_images * val_ratio)

        # Particionar los frames por segmento temporal
        partitioned_images = {
            "train": image_items[:n_train],
            "val": image_items[n_train:n_train + n_val],
            "test": image_items[n_train + n_val:],
        }

        for split_name, split_images in partitioned_images.items():
            if len(split_images) > 0:
                split_stats[split_name]["clips"] += 1

            for img_info in split_images:
                file_name = img_info.get("file_name")
                img_path = _resolve_image_path(frames_dir, file_name)
                if img_path is None:
                    continue

                img_width = int(img_info["width"])
                img_height = int(img_info["height"])

                anns = ann_by_image_id.get(img_info["id"], [])
                
                # --- VALIDACIÓN PREVENTIVA DE CORRUPCIÓN EN IMÁGENES ---
                if not img_path.exists() or img_path.stat().st_size == 0:
                    raise ValueError(f"Fallo de Calidad de Datos: La imagen {img_path.name} no existe o está vacía (0 bytes).")
                
                # Intentar leer la imagen para verificar que no esté corrupta
                test_img = cv2.imread(str(img_path))
                if test_img is None:
                    raise ValueError(f"Fallo de Calidad de Datos: La imagen {img_path.name} está corrupta o tiene un formato inválido.")
                
                img = None
                if split_name == "train" and copy_images:
                    img = test_img

                # Registrar coordenadas absolutas originales primero
                img_boxes = []
                img_classes = []
                has_salmon = False

                for ann in anns:
                    cat_id = ann.get("category_id")
                    class_name = cat_name_by_id.get(cat_id, str(cat_id))

                    if class_name not in target_classes:
                        filtered_out_annotations += 1
                        continue

                    bbox = ann.get("bbox")
                    if not bbox or len(bbox) != 4:
                        continue

                    cls_idx = class_to_idx[class_name]
                    if class_name == "Salmon":
                        has_salmon = True
                    
                    x, y, w, h = bbox
                    img_boxes.append([x, y, x + w, y + h])
                    img_classes.append(cls_idx)

                # --- 1. APLICAR COPY-PASTE SINTÉTICO (Solo en entrenamiento, si no tiene Salmon) ---
                if split_name == "train" and not has_salmon and len(salmon_bank) > 0 and img is not None:
                    # Decidir si pegamos con 85% de probabilidad para balancear la clase minoritaria (Salmon)
                    if random.random() < 0.85:
                        num_pastes = random.randint(1, 3)
                        for _ in range(num_pastes):
                            crop = random.choice(salmon_bank)
                            ch, cw = crop.shape[:2]
                            
                            # Escalar aleatoriamente el recorte del salmón
                            scale = random.uniform(0.7, 1.3)
                            n_ch, n_cw = int(round(ch * scale)), int(round(cw * scale))
                            if n_ch < 10 or n_cw < 10 or n_ch >= img_height or n_cw >= img_width:
                                continue
                                
                            crop_resized = cv2.resize(crop, (n_cw, n_ch), interpolation=cv2.INTER_LINEAR)
                            
                            # Intentar colocar en una posición sin solapamiento
                            placed = False
                            for _ in range(15):
                                px = random.randint(0, img_width - n_cw)
                                py = random.randint(0, img_height - n_ch)
                                candidate_box = [px, py, px + n_cw, py + n_ch]
                                
                                # Comprobar solapamiento
                                overlap = False
                                for existing in img_boxes:
                                    if _box_overlap(candidate_box, existing) > 0.05:
                                        overlap = True
                                        break
                                
                                if not overlap:
                                    # Pegar usando mezcla de bordes suaves
                                    mask = np.ones((n_ch, n_cw), dtype=np.float32)
                                    mask[0, :] = 0.2
                                    mask[-1, :] = 0.2
                                    mask[:, 0] = 0.2
                                    mask[:, -1] = 0.2
                                    mask = cv2.GaussianBlur(mask, (3, 3), 0)
                                    
                                    for c in range(3):
                                        img[py:py+n_ch, px:px+n_cw, c] = (
                                            mask * crop_resized[:, :, c] + (1.0 - mask) * img[py:py+n_ch, px:px+n_cw, c]
                                        ).astype(np.uint8)
                                        
                                    img_boxes.append(candidate_box)
                                    img_classes.append(0)  # 0 es la clase Salmon
                                    placed = True
                                    break

                # Convertir las coordenadas absolutas finales a formato YOLO normalizado
                yolo_bboxes = []
                for box in img_boxes:
                    x1, y1, x2, y2 = box
                    w = x2 - x1
                    h = y2 - y1
                    xc = x1 + w / 2.0
                    yc = y1 + h / 2.0
                    yolo_bboxes.append([xc / img_width, yc / img_height, w / img_width, h / img_height])

                # --- 2. APLICAR ALBUMENTATIONS UNDERWATER AUGMENTATIONS (Solo en entrenamiento) ---
                if split_name == "train" and img is not None and len(yolo_bboxes) > 0:
                    try:
                        import albumentations as A
                        transform = A.Compose([
                            A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=0.6),
                            A.RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.15, p=0.5),
                            A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=15, val_shift_limit=10, p=0.5),
                            A.GaussianBlur(blur_limit=(3, 5), p=0.3),
                        ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))
                        
                        transformed = transform(image=img, bboxes=yolo_bboxes, class_labels=img_classes)
                        img = transformed['image']
                        yolo_bboxes = transformed['bboxes']
                        img_classes = transformed['class_labels']
                    except Exception as e:
                        print(f"[WARNING] Falló la aumentación de Albumentations: {e}")

                # Generar líneas de salida en formato YOLO texto
                yolo_lines = []
                for cls_idx, bbox in zip(img_classes, yolo_bboxes):
                    xc, yc, w, h = bbox
                    yolo_lines.append(f"{cls_idx} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}")

                clip_name = clip_dir.name
                out_image_name = f"{clip_name}__{img_path.name}"
                out_label_name = f"{clip_name}__{img_path.stem}.txt"

                out_image_path = output_root / "images" / split_name / out_image_name
                out_label_path = output_root / "labels" / split_name / out_label_name

                # Guardar imagen y etiquetas
                if copy_images:
                    if split_name == "train" and img is not None:
                        cv2.imwrite(str(out_image_path), img)
                    else:
                        shutil.copy2(img_path, out_image_path)

                with out_label_path.open("w", encoding="utf-8") as f:
                    for line in yolo_lines:
                        f.write(line + "\n")

                if len(yolo_lines) == 0:
                    empty_label_files += 1

                exported_images += 1
                exported_labels += len(yolo_lines)
                split_stats[split_name]["images"] += 1
                split_stats[split_name]["labels"] += len(yolo_lines)

    data_yaml_path = output_root / "data.yaml"
    data_yaml_content = "\n".join(
        [
            f"path: {output_root.as_posix()}",
            "train: images/train",
            "val: images/val",
            "test: images/test",
            "names:",
            "  0: Salmon",
            "  1: Pollock",
        ]
    )
    data_yaml_path.write_text(data_yaml_content, encoding="utf-8")

    return {
        "annotated_data_root": str(annotated_root),
        "output_root": str(output_root),
        "target_classes": target_classes,
        "class_to_idx": class_to_idx,
        "n_clips_total": len(clip_jsons),
        "split_stats": split_stats,
        "exported_images": exported_images,
        "exported_labels": exported_labels,
        "filtered_out_annotations": filtered_out_annotations,
        "empty_label_files": empty_label_files,
        "data_yaml": str(data_yaml_path),
    }