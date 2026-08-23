"""Export and benchmark an Ultralytics model for an edge runtime."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="yolov8n.pt")
    parser.add_argument("--format", choices=("ncnn", "onnx"), default="ncnn")
    parser.add_argument("--source", default="bus.jpg", help="Benchmark image or video")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--int8", action="store_true")
    parser.add_argument("--data", help="Calibration dataset YAML required for INT8")
    parser.add_argument("--benchmark-frames", type=int, default=30)
    parser.add_argument("--report", default="artifacts/optimization_report.json")
    return parser.parse_args()


def artifact_size(path: Path) -> int:
    if path.is_file():
        return path.stat().st_size
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def benchmark(model_path: str, source: str, imgsz: int, limit: int) -> dict:
    from ultralytics import YOLO

    model = YOLO(model_path)
    timings_ms: list[float] = []
    results = model.predict(source=source, imgsz=imgsz, stream=True, verbose=False)
    for index, result in enumerate(results):
        speed = result.speed or {}
        timings_ms.append(float(speed.get("inference", 0.0)))
        if index + 1 >= limit:
            break
    timings_ms = [value for value in timings_ms if value > 0]
    median_ms = statistics.median(timings_ms) if timings_ms else 0.0
    return {
        "samples": len(timings_ms),
        "median_latency_ms": round(median_ms, 3),
        "estimated_fps": round(1000 / median_ms, 3) if median_ms else 0.0,
    }


def main() -> None:
    args = parse_args()
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise SystemExit("Install dependencies with: pip install -r requirements.txt") from exc

    if args.int8 and not args.data:
        raise SystemExit("--int8 requires --data with a representative calibration dataset")

    source_model = Path(args.model)
    model = YOLO(args.model)
    export_arguments = {
        "format": args.format,
        "imgsz": args.imgsz,
        "int8": args.int8,
    }
    if args.int8:
        export_arguments["data"] = args.data
    exported = Path(str(model.export(**export_arguments)))

    report = {
        "format": args.format,
        "int8": args.int8,
        "source_model": str(source_model),
        "exported_model": str(exported),
        "source_size_bytes": artifact_size(source_model),
        "exported_size_bytes": artifact_size(exported),
        "source_benchmark": benchmark(
            str(source_model), args.source, args.imgsz, args.benchmark_frames
        ),
        "exported_benchmark": benchmark(
            str(exported), args.source, args.imgsz, args.benchmark_frames
        ),
        "benchmark_device": "current_machine",
    }
    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
