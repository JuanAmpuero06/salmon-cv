from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Dict, Any, List, Tuple

import cv2


def _load_coco(coco_json_path: Path) -> Dict[str, Any]:
    with coco_json_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _index_coco(
    coco: Dict[str, Any]
) -> Tuple[Dict[int, Dict[str, Any]], Dict[int, List[Dict[str, Any]]], Dict[int, str]]:
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


def resolve_image_path(frames_dir: Path, file_name: str | None) -> Path | None:
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


def _summarize_single_coco(coco_path: Path, frames_dirname: str) -> Dict[str, Any]:
    coco = _load_coco(coco_path)
    images_by_id, ann_by_image_id, cat_name_by_id = _index_coco(coco)

    clip_dir = coco_path.parent
    frames_dir = clip_dir / frames_dirname

    image_items = list(images_by_id.values())
    annotations = coco.get("annotations", [])

    n_images = len(image_items)
    n_annotations = len(annotations)

    class_counts: Dict[str, int] = {}
    bbox_widths: List[float] = []
    bbox_heights: List[float] = []
    bbox_areas: List[float] = []

    frames_with_annotations = 0
    frames_without_annotations = 0
    unresolved_images = 0
    resolutions: Dict[str, int] = {}

    for item in image_items:
        anns = ann_by_image_id.get(item["id"], [])
        if anns:
            frames_with_annotations += 1
        else:
            frames_without_annotations += 1

        width = item.get("width")
        height = item.get("height")
        if width is not None and height is not None:
            key = f"{width}x{height}"
            resolutions[key] = resolutions.get(key, 0) + 1

        file_name = item.get("file_name")
        img_path = resolve_image_path(frames_dir, file_name)
        if img_path is None:
            unresolved_images += 1

        for ann in anns:
            cat_id = ann.get("category_id")
            class_name = cat_name_by_id.get(cat_id, str(cat_id))
            class_counts[class_name] = class_counts.get(class_name, 0) + 1

            bbox = ann.get("bbox")
            if bbox and len(bbox) == 4:
                _, _, w, h = bbox
                bbox_widths.append(float(w))
                bbox_heights.append(float(h))
                bbox_areas.append(float(w) * float(h))

    return {
        "clip_name": coco_path.parent.name,
        "coco_json": str(coco_path),
        "frames_dir": str(frames_dir),
        "n_images": n_images,
        "n_annotations": n_annotations,
        "classes_present": sorted(class_counts.keys()),
        "class_counts": class_counts,
        "frames_with_annotations": frames_with_annotations,
        "frames_without_annotations": frames_without_annotations,
        "unresolved_images": unresolved_images,
        "resolutions": resolutions,
        "bbox_stats": {
            "avg_width": round(sum(bbox_widths) / len(bbox_widths), 2) if bbox_widths else 0.0,
            "avg_height": round(sum(bbox_heights) / len(bbox_heights), 2) if bbox_heights else 0.0,
            "avg_area": round(sum(bbox_areas) / len(bbox_areas), 2) if bbox_areas else 0.0,
            "min_area": round(min(bbox_areas), 2) if bbox_areas else 0.0,
            "max_area": round(max(bbox_areas), 2) if bbox_areas else 0.0,
        },
    }


def create_coco_eda_samples(
    coco_json_relpath: str,
    frames_dirname: str,
    n_samples: int,
    seed: int,
    output_dir_relpath: str,
) -> Dict[str, Any]:
    project_root = Path.cwd()
    coco_path = project_root / coco_json_relpath
    if not coco_path.exists():
        raise FileNotFoundError(f"No existe COCO JSON: {coco_path}")

    coco = _load_coco(coco_path)
    images_by_id, ann_by_image_id, cat_name_by_id = _index_coco(coco)

    clip_dir = coco_path.parent
    frames_dir = clip_dir / frames_dirname
    if not frames_dir.exists():
        raise FileNotFoundError(f"No existe carpeta frames: {frames_dir}")

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
        img_path = resolve_image_path(frames_dir, file_name)
        if img_path is None:
            continue

        img = cv2.imread(str(img_path))
        if img is None:
            continue

        anns = ann_by_image_id.get(item["id"], [])
        for ann in anns:
            bbox = ann.get("bbox")
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

    return {
        "coco_json": str(coco_path),
        "frames_dir": str(frames_dir),
        "output_dir": str(out_dir),
        "n_requested": n_samples,
        "n_saved": len(saved),
        "saved": saved,
    }


def summarize_coco_clip(
    coco_json_relpath: str,
    frames_dirname: str,
) -> Dict[str, Any]:
    project_root = Path.cwd()
    coco_path = project_root / coco_json_relpath
    if not coco_path.exists():
        raise FileNotFoundError(f"No existe COCO JSON: {coco_path}")

    return _summarize_single_coco(coco_path, frames_dirname)


def summarize_dataset_clips(
    annotated_data_root_relpath: str,
    frames_dirname: str,
) -> Dict[str, Any]:
    """
    Recorre todos los clips dentro de annotated_data, encuentra su JSON
    y genera un inventario consolidado del dataset.
    """
    project_root = Path.cwd()
    annotated_root = project_root / annotated_data_root_relpath
    if not annotated_root.exists():
        raise FileNotFoundError(f"No existe la carpeta annotated_data: {annotated_root}")

    clip_summaries: List[Dict[str, Any]] = []
    skipped_dirs: List[str] = []

    for clip_dir in sorted(p for p in annotated_root.iterdir() if p.is_dir()):
        json_candidates = sorted(clip_dir.glob("*.json"))

        if not json_candidates:
            skipped_dirs.append(str(clip_dir))
            continue

        # En este dataset, cada carpeta debería tener un único json principal
        coco_path = json_candidates[0]

        try:
            clip_summary = _summarize_single_coco(coco_path, frames_dirname)
            clip_summaries.append(clip_summary)
        except Exception as e:
            skipped_dirs.append(f"{clip_dir} -> ERROR: {e}")

    total_images = sum(item["n_images"] for item in clip_summaries)
    total_annotations = sum(item["n_annotations"] for item in clip_summaries)

    global_class_counts: Dict[str, int] = {}
    clips_per_class: Dict[str, int] = {}

    for item in clip_summaries:
        for class_name, count in item["class_counts"].items():
            global_class_counts[class_name] = global_class_counts.get(class_name, 0) + count

        for class_name in item["classes_present"]:
            clips_per_class[class_name] = clips_per_class.get(class_name, 0) + 1

    inventory = {
        "annotated_data_root": str(annotated_root),
        "n_clips_found": len(clip_summaries),
        "n_clips_skipped": len(skipped_dirs),
        "total_images": total_images,
        "total_annotations": total_annotations,
        "global_class_counts": global_class_counts,
        "clips_per_class": clips_per_class,
        "skipped_dirs": skipped_dirs,
        "clips": clip_summaries,
    }
    return inventory