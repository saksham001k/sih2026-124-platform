"""Export and benchmark an Ultralytics model for an edge runtime."""

from __future__ import annotations

import argparse
import json
import time
import warnings
from pathlib import Path
from typing import Any

from urban_intelligence.optimization import (
    REPORT_LIMITATIONS,
    ParityDetection,
    aggregate_parity,
    artifact_record,
    build_optimization_report,
    collect_runtime_provenance,
    per_frame_parity,
    quantize_for_precision,
    summarize_timings,
    validate_benchmark_settings,
    validation_not_run,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="Source Ultralytics .pt model")
    parser.add_argument("--format", choices=("ncnn", "onnx"), default="onnx")
    parser.add_argument("--precision", choices=("fp32", "fp16", "int8"), default="fp32")
    parser.add_argument("--source", required=True, help="Benchmark image or video")
    parser.add_argument("--device", default="cpu", help="Inference device, default cpu")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--warmup-runs", type=int, default=5)
    parser.add_argument("--benchmark-frames", type=int, default=50)
    parser.add_argument("--confidence", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.70)
    parser.add_argument(
        "--data",
        help="Representative calibration YAML required for INT8 export",
    )
    parser.add_argument(
        "--validation-data",
        help="Optional YOLO dataset YAML for source/exported validation metrics",
    )
    parser.add_argument(
        "--int8",
        action="store_true",
        help="Deprecated alias for --precision int8",
    )
    parser.add_argument(
        "--report",
        default="artifacts/edge_bench/optimization_report.json",
        help="JSON benchmark report path",
    )
    return parser.parse_args()


def resolve_precision(args: argparse.Namespace) -> str:
    if args.int8 and args.precision != "fp32":
        raise SystemExit("Use either --precision int8 or deprecated --int8, not both")
    if args.int8:
        warnings.warn(
            "--int8 is deprecated; use --precision int8",
            DeprecationWarning,
            stacklevel=2,
        )
        return "int8"
    return args.precision


def load_benchmark_frames(source: Path, frame_count: int) -> list[Any]:
    import cv2

    if not source.is_file():
        raise SystemExit(f"Benchmark source not found: {source}")

    if source.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
        image = cv2.imread(str(source))
        if image is None:
            raise SystemExit(f"Unable to read benchmark image: {source}")
        return [image]

    capture = cv2.VideoCapture(str(source))
    if not capture.isOpened():
        raise SystemExit(f"Unable to open benchmark video: {source}")

    frames: list[Any] = []
    try:
        while len(frames) < frame_count:
            ok, frame = capture.read()
            if not ok:
                break
            frames.append(frame)
    finally:
        capture.release()

    if not frames:
        raise SystemExit(f"No frames could be read from benchmark source: {source}")
    return frames


def extract_parity_detections(result) -> list[ParityDetection]:
    boxes = result.boxes
    if boxes is None or len(boxes) == 0:
        return []
    xyxy = boxes.xyxy.cpu().tolist()
    confidences = boxes.conf.cpu().tolist()
    classes = boxes.cls.cpu().tolist()
    names = result.names
    detections: list[ParityDetection] = []
    for coords, confidence, class_id in zip(xyxy, confidences, classes, strict=True):
        class_index = int(class_id)
        detections.append(
            ParityDetection(
                class_id=class_index,
                class_name=str(names.get(class_index, class_index)),
                confidence=float(confidence),
                bbox=(float(coords[0]), float(coords[1]), float(coords[2]), float(coords[3])),
            )
        )
    return detections


def run_predict(model, frame, *, imgsz: int, confidence: float, iou: float, device: str):
    return model.predict(
        frame,
        imgsz=imgsz,
        conf=confidence,
        iou=iou,
        device=device,
        verbose=False,
    )


def benchmark_model(
    model,
    frames: list[Any],
    *,
    imgsz: int,
    confidence: float,
    iou: float,
    device: str,
    warmup_runs: int,
    benchmark_frames: int,
) -> dict[str, Any]:
    total_needed = warmup_runs + benchmark_frames
    inference_ms: list[float] = []
    wall_ms: list[float] = []

    for index in range(total_needed):
        frame = frames[index % len(frames)]
        if index < warmup_runs:
            run_predict(model, frame, imgsz=imgsz, confidence=confidence, iou=iou, device=device)
            continue
        start = time.perf_counter()
        results = run_predict(
            model, frame, imgsz=imgsz, confidence=confidence, iou=iou, device=device
        )
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        wall_ms.append(elapsed_ms)
        speed = results[0].speed or {}
        inference = float(speed.get("inference", 0.0))
        if inference > 0:
            inference_ms.append(inference)

    return summarize_timings(inference_ms, wall_ms)


def compare_prediction_parity(
    source_model,
    exported_model,
    frames: list[Any],
    *,
    imgsz: int,
    confidence: float,
    iou: float,
    device: str,
    frame_count: int,
) -> dict[str, Any]:
    frame_results: list[dict[str, float | int | bool]] = []
    for index in range(frame_count):
        frame = frames[index % len(frames)]
        source_result = run_predict(
            source_model, frame, imgsz=imgsz, confidence=confidence, iou=iou, device=device
        )[0]
        exported_result = run_predict(
            exported_model, frame, imgsz=imgsz, confidence=confidence, iou=iou, device=device
        )[0]
        frame_results.append(
            per_frame_parity(
                extract_parity_detections(source_result),
                extract_parity_detections(exported_result),
            )
        )
    aggregated = aggregate_parity(frame_results)
    aggregated["note"] = (
        "prediction_parity compares postprocessed detections on the same frames; "
        "it is not a validation accuracy claim"
    )
    return aggregated


def run_optional_validation(model, validation_data: str, device: str) -> dict[str, float]:
    metrics = model.val(data=validation_data, device=device, verbose=False)
    return {
        "precision": round(float(getattr(metrics.box, "mp", 0.0)), 4),
        "recall": round(float(getattr(metrics.box, "mr", 0.0)), 4),
        "map50": round(float(getattr(metrics.box, "map50", 0.0)), 4),
        "map50_95": round(float(getattr(metrics.box, "map50_95", 0.0)), 4),
    }


def package_validation(
    source_metrics: dict[str, float] | None,
    exported_metrics: dict[str, float] | None,
) -> dict[str, Any]:
    if source_metrics is None or exported_metrics is None:
        return validation_not_run()
    differences = {
        key: round(exported_metrics[key] - source_metrics[key], 4)
        for key in source_metrics
    }
    return {
        "status": "completed",
        "source": source_metrics,
        "exported": exported_metrics,
        "differences": differences,
    }


def export_model(
    model,
    *,
    export_format: str,
    precision: str,
    imgsz: int,
    calibration_data: str | None,
) -> Path:
    export_kwargs: dict[str, Any] = {"format": export_format, "imgsz": imgsz}
    quantize = quantize_for_precision(precision)
    if quantize is not None:
        export_kwargs["quantize"] = quantize
    if precision == "int8":
        export_kwargs["data"] = calibration_data
    exported = Path(str(model.export(**export_kwargs)))
    if not exported.exists():
        raise SystemExit(f"Export failed; artifact not found: {exported}")
    return exported


def get_package_version(module_name: str) -> str:
    try:
        module = __import__(module_name)
    except ImportError:
        return "not_installed"
    return str(getattr(module, "__version__", "unknown"))


def main() -> None:
    args = parse_args()
    precision = resolve_precision(args)
    source_path = Path(args.model)
    source_media = Path(args.source)
    report_path = Path(args.report)

    if not source_path.is_file():
        raise SystemExit(f"Model not found: {source_path}")

    calibration_data = args.data
    if precision == "int8" and calibration_data:
        calibration_path = Path(calibration_data)
        if not calibration_path.is_file():
            raise SystemExit(f"Calibration YAML not found: {calibration_path}")

    try:
        validate_benchmark_settings(
            warmup_runs=args.warmup_runs,
            benchmark_frames=args.benchmark_frames,
            imgsz=args.imgsz,
            confidence=args.confidence,
            iou=args.iou,
            device=args.device,
            calibration_data=calibration_data,
            precision=precision,
            export_format=args.format,
        )
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise SystemExit("Install dependencies with: pip install -r requirements.txt") from exc

    frames = load_benchmark_frames(source_media, args.benchmark_frames)
    source_model = YOLO(str(source_path))
    export_arguments = {
        "format": args.format,
        "imgsz": args.imgsz,
        "precision": precision,
        "quantize": quantize_for_precision(precision),
    }
    if precision == "int8":
        export_arguments["data"] = calibration_data

    exported_path = export_model(
        source_model,
        export_format=args.format,
        precision=precision,
        imgsz=args.imgsz,
        calibration_data=calibration_data,
    )
    exported_model = YOLO(str(exported_path))

    benchmark_configuration = {
        "source_media": str(source_media),
        "warmup_runs": args.warmup_runs,
        "benchmark_frames": args.benchmark_frames,
        "confidence": args.confidence,
        "iou": args.iou,
        "device": args.device,
        "imgsz": args.imgsz,
    }

    source_benchmark = benchmark_model(
        source_model,
        frames,
        imgsz=args.imgsz,
        confidence=args.confidence,
        iou=args.iou,
        device=args.device,
        warmup_runs=args.warmup_runs,
        benchmark_frames=args.benchmark_frames,
    )
    exported_benchmark = benchmark_model(
        exported_model,
        frames,
        imgsz=args.imgsz,
        confidence=args.confidence,
        iou=args.iou,
        device=args.device,
        warmup_runs=args.warmup_runs,
        benchmark_frames=args.benchmark_frames,
    )
    prediction_parity = compare_prediction_parity(
        source_model,
        exported_model,
        frames,
        imgsz=args.imgsz,
        confidence=args.confidence,
        iou=args.iou,
        device=args.device,
        frame_count=args.benchmark_frames,
    )

    validation = validation_not_run()
    if args.validation_data:
        validation_path = Path(args.validation_data)
        if not validation_path.is_file():
            raise SystemExit(f"Validation YAML not found: {validation_path}")
        source_metrics = run_optional_validation(
            source_model, str(validation_path), args.device
        )
        exported_metrics = run_optional_validation(
            exported_model, str(validation_path), args.device
        )
        validation = package_validation(source_metrics, exported_metrics)

    onnxruntime_version = get_package_version("onnxruntime")
    runtime_provenance = collect_runtime_provenance(
        device=args.device,
        imgsz=args.imgsz,
        benchmark_config=benchmark_configuration,
        ultralytics_version=get_package_version("ultralytics"),
        torch_version=get_package_version("torch"),
        onnxruntime_version=None if onnxruntime_version == "not_installed" else onnxruntime_version,
    )

    report = build_optimization_report(
        source_artifact=artifact_record(source_path),
        exported_artifact=artifact_record(exported_path),
        export_format=args.format,
        precision=precision,
        export_arguments=export_arguments,
        runtime_provenance=runtime_provenance,
        benchmark_configuration=benchmark_configuration,
        source_benchmark=source_benchmark,
        exported_benchmark=exported_benchmark,
        prediction_parity=prediction_parity,
        validation=validation,
        limitations=REPORT_LIMITATIONS,
    )

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
