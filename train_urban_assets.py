"""Train and validate one Ultralytics model for visible urban assets and waterlogging."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

REQUIRED_CLASSES = frozenset(
    {
        "road_divider",
        "zebra_crossing",
        "traffic_signboard",
        "damaged_divider",
        "damaged_zebra_crossing",
        "damaged_traffic_signboard",
        "waterlogging",
    }
)


def dataset_class_names(data: dict[str, Any]) -> list[str]:
    names = data.get("names")
    if isinstance(names, list):
        return [str(value) for value in names]
    if isinstance(names, dict):
        try:
            return [str(names[key]) for key in sorted(names, key=lambda value: int(value))]
        except (TypeError, ValueError) as exc:
            raise ValueError("dataset class-name keys must be integer-like") from exc
    raise ValueError("dataset YAML must define names as a list or index mapping")


def validate_dataset_classes(names: list[str]) -> None:
    normalized = [name.strip().lower() for name in names]
    if len(normalized) != len(set(normalized)):
        raise ValueError("urban-assets dataset class names must be unique")
    missing = REQUIRED_CLASSES - set(normalized)
    if missing:
        raise ValueError(f"urban-assets dataset is missing: {', '.join(sorted(missing))}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, help="YOLO dataset YAML")
    parser.add_argument("--model", default="yolov8n.pt")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", default="0")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--project", default="runs/urban_assets")
    parser.add_argument("--name", default="yolov8n_urban_assets")
    parser.add_argument("--seed", type=int, default=26124)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_path = Path(args.data)
    if not data_path.is_file():
        raise SystemExit(f"Dataset YAML not found: {data_path}")
    try:
        import yaml
        from ultralytics import YOLO
    except ImportError as exc:
        raise SystemExit("Install dependencies with: pip install -r requirements.txt") from exc
    value = yaml.safe_load(data_path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise SystemExit("Dataset YAML must contain an object")
    try:
        validate_dataset_classes(dataset_class_names(value))
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    model = YOLO(args.model)
    result = model.train(
        data=str(data_path),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        project=args.project,
        name=args.name,
        seed=args.seed,
    )
    best_path = Path(result.save_dir) / "weights" / "best.pt"
    validation = YOLO(str(best_path)).val(data=str(data_path), device=args.device)
    metrics = {
        "precision": float(validation.box.mp),
        "recall": float(validation.box.mr),
        "mAP50": float(validation.box.map50),
        "mAP50-95": float(validation.box.map),
        "best_checkpoint": str(best_path),
    }
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
