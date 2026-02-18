from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Dict, Any, List, Tuple

import cv2


def _load_coco(coco_json_path: Path) -> Dict[str, Any]:
    with coco_json_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _index_coco(coco: Dict[str, Any]) -> Tuple[Dict[int, Dict[str, Any]], Dict[int, List[Dict[str, Any]]], Dict[int, str]]:
    images_by_id = {img["id"]: img for img in coco.get("images", [])}

    ann_by_image_id: Dict[int, List[Dict[str, Any]]] = {}
    for ann in coco.get("annotations", []):
        ann_by_image_id.setdefault(ann["image_id"], []).append(ann)

    cat_name_by_id = {cat["id"]: cat.get("name", str(cat["id"])) for cat in coco.get("categories", [])}
    return images_by_id, ann_by_image_id, cat_name_by_id


def _draw_bbox(img, bbox_xywh, label: str):
    x, y, w, h = bbox_xywh
    x1, y1 = int(round(x)), int(round(y))
    x2, y2 = int(round(x + w)), int(round(y + h))
    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.putText(img, label, (x1, max(0, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)


def create_coco_eda_samples(
    coco_json_relpath: str,
    frames_dirname: str,
    n_samples: int,
    seed: int,
    output_dir_relpath: str,
) -> Dict[str, Any]:
    """
    Lee 1 JSON COCO (del clip), toma N frames aleatorios, dibuja bboxes y guarda imágenes en 08_reporting.
    Devuelve un manifest (útil para debugging y trazabilidad).
    """
    project_root = Path.cwd()  # esto será .../training cuando corres kedro run
    coco_path = project_root / coco_json_relpath
    if not coco_path.exists():
        raise FileNotFoundError(f"No existe COCO JSON: {coco_path}")

    coco = _load_coco(coco_path)
    images_by_id, ann_by_image_id, cat_name_by_id = _index_coco(coco)

    # La carpeta frames está al mismo nivel que el json en tu estructura
    clip_dir = coco_path.parent
    frames_dir = clip_dir / frames_dirname
    if not frames_dir.exists():
        raise FileNotFoundError(f"No existe carpeta frames: {frames_dir}")

    # Elegimos muestras de las imágenes listadas en COCO (más robusto que listar archivos sueltos)
    image_items = list(images_by_id.values())
    if not image_items:
        raise ValueError("El COCO no trae 'images' o está vacío.")

    random.seed(seed)
    sample_items = random.sample(image_items, k=min(n_samples, len(image_items)))

    out_dir = project_root / output_dir_relpath
    out_dir.mkdir(parents=True, exist_ok=True)

    saved: List[Dict[str, Any]] = []
    for item in sample_items:
        file_name = item.get("file_name")
        img_path = frames_dir / file_name if file_name else None

        # fallback: si el COCO usa file_name distinto, intentamos encontrar por patrón
        if img_path is None or not img_path.exists():
            # intenta con el nombre exacto en frames/
            candidates = list(frames_dir.glob(file_name)) if file_name else []
            img_path = candidates[0] if candidates else None

        if img_path is None or not img_path.exists():
            continue

        img = cv2.imread(str(img_path))
        if img is None:
            continue

        anns = ann_by_image_id.get(item["id"], [])
        for ann in anns:
            bbox = ann.get("bbox")  # COCO bbox = [x,y,width,height] en pixeles :contentReference[oaicite:2]{index=2}
            if not bbox or len(bbox) != 4:
                continue
            cat_id = ann.get("category_id")
            label = cat_name_by_id.get(cat_id, str(cat_id))
            _draw_bbox(img, bbox, label)

        out_path = out_dir / f"eda_{img_path.stem}.jpg"
        cv2.imwrite(str(out_path), img)

        saved.append(
            {
                "image_id": item["id"],
                "file_name": file_name,
                "source_path": str(img_path),
                "output_path": str(out_path),
                "n_annotations": len(anns),
            }
        )

    manifest = {
        "coco_json": str(coco_path),
        "frames_dir": str(frames_dir),
        "output_dir": str(out_dir),
        "n_requested": n_samples,
        "n_saved": len(saved),
        "saved": saved,
    }
    return manifest