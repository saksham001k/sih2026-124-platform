"""Train and validate a YOLOv8 road-hazard model on prepared RDD2022 data."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data",
        required=True,
        help="Path to generated Ultralytics data.yaml",
    )
    parser.add_argument("--model", default="yolov8n.pt", help="Base Ultralytics model")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", default="", help="Ultralytics device string, e.g. 0 or cpu")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--project", default="runs/road_hazard")
    parser.add_argument("--name", default="yolov8n_rdd2022_india")
    parser.add_argument("--seed", type=int, default=26124)
    return parser.parse_args()


def format_metric(value: object) -> str:
    if value is None:
        return "n/a"
    try:
        return f"{float(value):.4f}"
    except (TypeError, ValueError):
        return str(value)


def main() -> None:
    args = parse_args()
    data_path = Path(args.data)
    if not data_path.is_file():
        print(f"Error: dataset YAML not found: {data_path}", file=sys.stderr)
        raise SystemExit(1)

    try:
        from ultralytics import YOLO
    except ImportError as exc:
        print(
            "Error: ultralytics is not installed. Run: pip install -r requirements.txt",
            file=sys.stderr,
        )
        raise SystemExit(1) from exc

    try:
        model = YOLO(args.model)
        train_kwargs = {
            "data": str(data_path),
            "epochs": args.epochs,
            "imgsz": args.imgsz,
            "batch": args.batch,
            "workers": args.workers,
            "project": args.project,
            "name": args.name,
            "seed": args.seed,
            "exist_ok": True,
        }
        if args.device:
            train_kwargs["device"] = args.device

        train_results = model.train(**train_kwargs)
        run_dir = Path(getattr(train_results, "save_dir", Path(args.project) / args.name))
        best_path = run_dir / "weights" / "best.pt"
        if not best_path.is_file():
            print(f"Error: best checkpoint not found at {best_path}", file=sys.stderr)
            raise SystemExit(1)

        print(f"Best checkpoint: {best_path.resolve()}")

        val_model = YOLO(str(best_path))
        val_kwargs = {"data": str(data_path)}
        if args.device:
            val_kwargs["device"] = args.device
        metrics = val_model.val(**val_kwargs)

        box_metrics = getattr(metrics, "box", None)
        if box_metrics is not None:
            print("Validation metrics:")
            if hasattr(box_metrics, "mp"):
                print(f"  precision: {format_metric(box_metrics.mp)}")
            if hasattr(box_metrics, "mr"):
                print(f"  recall: {format_metric(box_metrics.mr)}")
            if hasattr(box_metrics, "map50"):
                print(f"  mAP50: {format_metric(box_metrics.map50)}")
            if hasattr(box_metrics, "map"):
                print(f"  mAP50-95: {format_metric(box_metrics.map)}")
        else:
            print("Validation completed, but box metrics were not returned by Ultralytics.")
    except Exception as exc:
        message = str(exc).lower()
        if "cuda" in message or "mps" in message or "device" in message:
            print(f"Error: requested device is unavailable: {exc}", file=sys.stderr)
        else:
            print(f"Error: training or validation failed: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
