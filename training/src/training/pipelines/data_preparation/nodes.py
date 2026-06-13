from __future__ import annotations

import json
import random
import shutil
from pathlib import Path
from typing import Any, Dict, List, Tuple


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
                yolo_lines: List[str] = []

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
                    x_c, y_c, w, h = _coco_bbox_to_yolo(bbox, img_width, img_height)

                    yolo_lines.append(
                        f"{cls_idx} {x_c:.6f} {y_c:.6f} {w:.6f} {h:.6f}"
                    )

                clip_name = clip_dir.name
                out_image_name = f"{clip_name}__{img_path.name}"
                out_label_name = f"{clip_name}__{img_path.stem}.txt"

                out_image_path = output_root / "images" / split_name / out_image_name
                out_label_path = output_root / "labels" / split_name / out_label_name

                if copy_images:
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