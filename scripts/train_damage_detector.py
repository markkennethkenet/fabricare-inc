#!/usr/bin/env python3
"""Train a candidate YOLOv8 detector and report per-class test metrics."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "dataset_sources" / "prepared_damage_yolov8" / "data.yaml"
CURRENT_MODEL = ROOT / "model" / "best.pt"


def sha256(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def evaluate(checkpoint, data, output, device, batch, image_size):
    from ultralytics import YOLO

    model = YOLO(str(checkpoint), task="detect")
    metrics = model.val(
        data=str(data), split="test", imgsz=image_size, batch=batch,
        device=device, workers=0, plots=True, verbose=True,
        conf=0.001, project=str(output), name="test", exist_ok=True,
    )
    by_class = {int(class_id): i for i, class_id in enumerate(metrics.ap_class_index)}
    if set(by_class) != set(model.names):
        raise RuntimeError("The held-out test metrics must contain all six classes.")
    rows = []
    for class_id, name in model.names.items():
        precision, recall, ap50, ap = metrics.class_result(by_class[class_id])
        rows.append({
            "class": name,
            "test_images": int(metrics.nt_per_image[class_id]),
            "test_boxes": int(metrics.nt_per_class[class_id]),
            "precision": float(precision),
            "recall": float(recall),
            "AP50": float(ap50),
            "AP50-95": float(ap),
        })
    with (output / "per-class-test.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return {
        "checkpoint": str(checkpoint), "checkpoint_sha256": sha256(checkpoint),
        "split": "test", "evaluation_image_size": image_size,
        "metrics": metrics.results_dict, "classes": rows,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--data", type=Path, default=DATASET)
    parser.add_argument("--train-data", type=Path,
                        help="Optional resized training copy; --data remains the test source.")
    parser.add_argument("--device", default="0")
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--train-size", type=int, default=416)
    parser.add_argument("--test-size", type=int, default=640)
    args = parser.parse_args()

    weights, data, output = (p.resolve(strict=must_exist) for p, must_exist in (
        (args.weights, True), (args.data, True), (args.output, False)
    ))
    if weights == CURRENT_MODEL.resolve():
        raise ValueError("Supply the pretrained YOLOv8 weights for this separate experiment.")
    output.mkdir(parents=True, exist_ok=False)
    train_data = args.train_data.resolve(strict=True) if args.train_data else data
    os.environ.setdefault("YOLO_CONFIG_DIR", str(output.parent / "ultralytics-config"))
    from ultralytics import YOLO

    active_hash = sha256(CURRENT_MODEL)
    candidate = YOLO(str(weights), task="detect")
    candidate.train(
        data=str(train_data), epochs=args.epochs, batch=args.batch, imgsz=args.train_size,
        device=args.device, workers=0, cache=False, seed=42, deterministic=True,
        amp=True, patience=0, close_mosaic=min(3, args.epochs // 4),
        project=str(output), name="train", exist_ok=True, plots=True,
    )
    checkpoint = output / "train" / "weights" / "best.pt"
    report = evaluate(checkpoint, data, output, args.device, min(4, args.batch), args.test_size)
    if active_hash != sha256(CURRENT_MODEL):
        raise RuntimeError("Active detector checksum changed during training.")
    report.update({
        "active_model_sha256": active_hash, "pretrained_weights": str(weights),
        "pretrained_weights_sha256": sha256(weights), "data": str(data),
        "training_data": str(train_data),
        "train_epochs_requested": args.epochs, "train_image_size": args.train_size,
    })
    (output / "test-metrics.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
