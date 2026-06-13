from __future__ import annotations

import csv
import datetime
import json
from pathlib import Path
from typing import Any, Dict

from ultralytics import YOLO


def _parse_yolo_metrics(results_csv_path: Path) -> Dict[str, float]:
    if not results_csv_path.exists():
        return {}
    try:
        with results_csv_path.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        if not rows:
            return {}
        
        # Clean column names (strip spaces)
        cleaned_rows = []
        for r in rows:
            cleaned_row = {k.strip(): v.strip() for k, v in r.items() if k is not None}
            cleaned_rows.append(cleaned_row)
            
        # Encontrar la mejor época basada en mAP50(B)
        best_row = None
        best_map50 = -1.0
        for r in cleaned_rows:
            try:
                map50 = float(r.get("metrics/mAP50(B)", 0))
                if map50 > best_map50:
                    best_map50 = map50
                    best_row = r
            except (ValueError, TypeError):
                continue
                
        # Si no se encontró, usar la última fila
        if best_row is None:
            best_row = cleaned_rows[-1]
            
        return {
            "best_epoch": int(best_row.get("epoch", 0)),
            "train_box_loss": float(best_row.get("train/box_loss", 0)),
            "train_cls_loss": float(best_row.get("train/cls_loss", 0)),
            "val_box_loss": float(best_row.get("val/box_loss", 0)),
            "val_cls_loss": float(best_row.get("val/cls_loss", 0)),
            "metrics_precision": float(best_row.get("metrics/precision(B)", 0)),
            "metrics_recall": float(best_row.get("metrics/recall(B)", 0)),
            "metrics_mAP50": float(best_row.get("metrics/mAP50(B)", 0)),
            "metrics_mAP50_95": float(best_row.get("metrics/mAP50-95(B)", 0)),
        }
    except Exception as e:
        print(f"[WARNING] Error al parsear results.csv: {e}")
        return {}


def _append_to_history(history_json_path: Path, new_entry: Dict[str, Any]):
    history_json_path.parent.mkdir(parents=True, exist_ok=True)
    history_data = []
    if history_json_path.exists():
        try:
            with history_json_path.open("r", encoding="utf-8") as f:
                history_data = json.load(f)
                if not isinstance(history_data, list):
                    history_data = []
        except Exception:
            history_data = []
            
    history_data.append(new_entry)
    
    with history_json_path.open("w", encoding="utf-8") as f:
        json.dump(history_data, f, indent=2, ensure_ascii=False)


def _append_to_history_csv(history_csv_path: Path, new_entry: Dict[str, Any]):
    history_csv_path.parent.mkdir(parents=True, exist_ok=True)
    file_exists = history_csv_path.exists()
    headers = list(new_entry.keys())
    
    try:
        with history_csv_path.open("a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=headers)
            if not file_exists:
                writer.writeheader()
            writer.writerow(new_entry)
    except Exception as e:
        print(f"[WARNING] Error al escribir en history.csv: {e}")


def train_yolo_detector(
    data_yaml_relpath: str,
    model_name: str,
    imgsz: int,
    epochs: int,
    batch: int,
    workers: int,
    patience: int,
    device: str,
    pretrained: bool,
    project_relpath: str,
    run_name: str,
    save: bool,
    save_period: int,
    plots: bool,
    verbose: bool,
    resume: bool,
    resume_checkpoint_relpath: str,
) -> Dict[str, Any]:
    project_root = Path.cwd()

    data_yaml_path = project_root / data_yaml_relpath
    if not data_yaml_path.exists():
        raise FileNotFoundError(f"No existe data.yaml: {data_yaml_path}")

    runs_root = project_root / project_relpath
    runs_root.mkdir(parents=True, exist_ok=True)

    resume_ckpt = project_root / resume_checkpoint_relpath
    should_resume = resume and resume_ckpt.exists()

    if resume and not resume_ckpt.exists():
        print(f"[WARNING] Se especifico 'resume=True' pero no se encontro el checkpoint en '{resume_ckpt}'. Iniciando entrenamiento desde cero.")

    if should_resume:
        print(f"Reanudando entrenamiento desde el checkpoint: {resume_ckpt}")
        model = YOLO(str(resume_ckpt))
        results = model.train(
            resume=True,
        )
    else:
        print(f"Iniciando entrenamiento desde cero usando modelo base: {model_name}")
        model = YOLO(model_name)
        results = model.train(
            data=str(data_yaml_path),
            imgsz=imgsz,
            epochs=epochs,
            batch=batch,
            workers=workers,
            patience=patience,
            device=device,
            pretrained=pretrained,
            project=str(runs_root),
            name=run_name,
            save=save,
            save_period=save_period,
            plots=plots,
            verbose=verbose,
        )

    run_dir = Path(results.save_dir)
    weights_dir = run_dir / "weights"
    best_pt = weights_dir / "best.pt"
    last_pt = weights_dir / "last.pt"
    results_csv = run_dir / "results.csv"

    # Registrar el historico del experimento
    metrics = {}
    if results_csv.exists():
        metrics = _parse_yolo_metrics(results_csv)

    new_entry = {
        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "run_name": run_name,
        "model_name": model_name,
        "imgsz": imgsz,
        "epochs": epochs,
        "batch": batch,
        "device": device,
        "pretrained": pretrained,
        "resume": resume,
        "best_epoch": metrics.get("best_epoch", None),
        "metrics_mAP50": metrics.get("metrics_mAP50", None),
        "metrics_mAP50_95": metrics.get("metrics_mAP50_95", None),
        "metrics_precision": metrics.get("metrics_precision", None),
        "metrics_recall": metrics.get("metrics_recall", None),
        "train_box_loss": metrics.get("train_box_loss", None),
        "train_cls_loss": metrics.get("train_cls_loss", None),
        "val_box_loss": metrics.get("val_box_loss", None),
        "val_cls_loss": metrics.get("val_cls_loss", None),
        "best_pt": str(best_pt) if best_pt.exists() else None,
        "run_dir": str(run_dir),
    }

    # Guardar en JSON e historico CSV en 08_reporting
    history_json_path = project_root / "data" / "08_reporting" / "train_history.json"
    history_csv_path = project_root / "data" / "08_reporting" / "train_history.csv"
    
    _append_to_history(history_json_path, new_entry)
    _append_to_history_csv(history_csv_path, new_entry)
    print(f"Resultado del experimento guardado en el historico:")
    print(f" - JSON: {history_json_path.resolve()}")
    print(f" - CSV: {history_csv_path.resolve()}")

    return {
        "data_yaml": str(data_yaml_path) if data_yaml_path.exists() else None,
        "model_name": model_name,
        "imgsz": imgsz,
        "epochs": epochs,
        "batch": batch,
        "workers": workers,
        "patience": patience,
        "device": device,
        "pretrained": pretrained,
        "resume": resume,
        "resume_checkpoint": str(project_root / resume_checkpoint_relpath) if resume else None,
        "run_dir": str(run_dir),
        "weights_dir": str(weights_dir),
        "best_pt": str(best_pt) if best_pt.exists() else None,
        "last_pt": str(last_pt) if last_pt.exists() else None,
        "results_csv": str(results_csv) if results_csv.exists() else None,
    }