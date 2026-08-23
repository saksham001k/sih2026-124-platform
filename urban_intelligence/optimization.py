"""Pure helpers for edge export benchmarking and prediction parity."""

from __future__ import annotations

import hashlib
import json
import math
import os
import platform
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

EXPORT_FORMATS = ("onnx", "ncnn")
PRECISIONS = ("fp32", "fp16", "int8")
SUPPORTED_PRECISIONS = {
    "onnx": ("fp32", "fp16", "int8"),
    "ncnn": ("fp32", "fp16"),
}
SCHEMA_VERSION = "1.0"
PARITY_IOU_THRESHOLD = 0.50


@dataclass(frozen=True, slots=True)
class ParityDetection:
    class_id: int
    class_name: str
    confidence: float
    bbox: tuple[float, float, float, float]


def quantize_for_precision(precision: str) -> int | None:
    if precision == "fp32":
        return None
    if precision == "fp16":
        return 16
    if precision == "int8":
        return 8
    raise ValueError(f"Unsupported precision: {precision}")


def validate_precision_format(export_format: str, precision: str) -> None:
    if export_format not in EXPORT_FORMATS:
        raise ValueError(f"format must be one of: {', '.join(EXPORT_FORMATS)}")
    if precision not in PRECISIONS:
        raise ValueError(f"precision must be one of: {', '.join(PRECISIONS)}")
    allowed = SUPPORTED_PRECISIONS[export_format]
    if precision not in allowed:
        if export_format == "ncnn" and precision == "int8":
            raise ValueError("NCNN export does not support INT8 quantization in Ultralytics")
        raise ValueError(f"{export_format} does not support precision {precision}")


def validate_benchmark_settings(
    *,
    warmup_runs: int,
    benchmark_frames: int,
    imgsz: int,
    confidence: float,
    iou: float,
    device: str,
    calibration_data: str | None = None,
    precision: str = "fp32",
    export_format: str = "onnx",
) -> None:
    validate_precision_format(export_format, precision)
    if warmup_runs < 0:
        raise ValueError("warmup_runs must be >= 0")
    if benchmark_frames < 1:
        raise ValueError("benchmark_frames must be at least 1")
    if imgsz < 32:
        raise ValueError("imgsz must be at least 32")
    if not 0.0 <= confidence <= 1.0:
        raise ValueError("confidence must be between 0 and 1")
    if not 0.0 <= iou <= 1.0:
        raise ValueError("iou must be between 0 and 1")
    if not device:
        raise ValueError("device must not be empty")
    if precision == "int8" and not calibration_data:
        raise ValueError("INT8 export requires --data with representative calibration YAML")


def artifact_size_bytes(path: Path) -> int:
    if path.is_file():
        return path.stat().st_size
    if path.is_dir():
        return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())
    return 0


def artifact_size_mib(path: Path) -> float:
    return round(artifact_size_bytes(path) / (1024 * 1024), 4)


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def artifact_manifest(path: Path) -> list[dict[str, Any]]:
    if path.is_file():
        return [
            {
                "relative_path": path.name,
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        ]
    entries: list[dict[str, Any]] = []
    for item in sorted(path.rglob("*")):
        if not item.is_file():
            continue
        entries.append(
            {
                "relative_path": str(item.relative_to(path)),
                "bytes": item.stat().st_size,
                "sha256": sha256_file(item),
            }
        )
    return entries


def artifact_record(path: Path) -> dict[str, Any]:
    total_bytes = artifact_size_bytes(path)
    record: dict[str, Any] = {
        "path": str(path),
        "type": "file" if path.is_file() else "directory",
        "total_bytes": total_bytes,
        "mib": round(total_bytes / (1024 * 1024), 4),
        "sha256": sha256_file(path) if path.is_file() else None,
        "manifest": artifact_manifest(path) if path.exists() else [],
    }
    if path.is_dir() and record["manifest"]:
        manifest_json = json.dumps(record["manifest"], sort_keys=True).encode("utf-8")
        record["manifest_sha256"] = hashlib.sha256(manifest_json).hexdigest()
    return record


def size_reduction(source_bytes: int, exported_bytes: int) -> dict[str, float | int]:
    difference = source_bytes - exported_bytes
    percent = (difference / source_bytes * 100.0) if source_bytes > 0 else 0.0
    return {
        "difference_bytes": difference,
        "reduction_percent": round(percent, 4),
    }


def percentile_95(values: list[float]) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, math.ceil(0.95 * len(ordered)) - 1)
    return ordered[index]


def summarize_timings(
    inference_ms: list[float],
    wall_ms: list[float],
) -> dict[str, float | int]:
    if not wall_ms:
        raise ValueError("zero valid benchmark samples collected")
    inference = [value for value in inference_ms if value > 0]
    wall_median = float(sorted(wall_ms)[len(wall_ms) // 2]) if wall_ms else 0.0
    if len(wall_ms) % 2 == 0 and wall_ms:
        mid = len(wall_ms) // 2
        wall_median = (wall_ms[mid - 1] + wall_ms[mid]) / 2.0
    inference_median = float(sorted(inference)[len(inference) // 2]) if inference else 0.0
    if len(inference) % 2 == 0 and inference:
        mid = len(inference) // 2
        inference_median = (inference[mid - 1] + inference[mid]) / 2.0
    return {
        "sample_count": len(wall_ms),
        "inference_median_ms": round(inference_median, 3),
        "inference_p95_ms": round(percentile_95(inference), 3),
        "wall_median_ms": round(wall_median, 3),
        "wall_p95_ms": round(percentile_95(wall_ms), 3),
        "end_to_end_fps": round(1000.0 / wall_median, 3) if wall_median > 0 else 0.0,
    }


def bbox_iou(
    left: tuple[float, float, float, float],
    right: tuple[float, float, float, float],
) -> float:
    left_x1, left_y1, left_x2, left_y2 = left
    right_x1, right_y1, right_x2, right_y2 = right
    inter_x1 = max(left_x1, right_x1)
    inter_y1 = max(left_y1, right_y1)
    inter_x2 = min(left_x2, right_x2)
    inter_y2 = min(left_y2, right_y2)
    inter_width = max(0.0, inter_x2 - inter_x1)
    inter_height = max(0.0, inter_y2 - inter_y1)
    intersection = inter_width * inter_height
    if intersection <= 0:
        return 0.0
    left_area = max(0.0, left_x2 - left_x1) * max(0.0, left_y2 - left_y1)
    right_area = max(0.0, right_x2 - right_x1) * max(0.0, right_y2 - right_y1)
    union = left_area + right_area - intersection
    if union <= 0:
        return 0.0
    return intersection / union


def greedy_match_same_class(
    source: list[ParityDetection],
    exported: list[ParityDetection],
    *,
    iou_threshold: float = PARITY_IOU_THRESHOLD,
) -> list[tuple[ParityDetection, ParityDetection, float]]:
    ordered_source = sorted(source, key=lambda item: (-item.confidence, item.class_id))
    used_exported: set[int] = set()
    matches: list[tuple[ParityDetection, ParityDetection, float]] = []
    for left in ordered_source:
        best_index: int | None = None
        best_iou = 0.0
        for index, right in enumerate(exported):
            if index in used_exported or right.class_id != left.class_id:
                continue
            overlap = bbox_iou(left.bbox, right.bbox)
            if overlap >= iou_threshold and overlap > best_iou:
                best_iou = overlap
                best_index = index
        if best_index is not None:
            used_exported.add(best_index)
            matches.append((left, exported[best_index], best_iou))
    return matches


def per_frame_parity(
    source: list[ParityDetection],
    exported: list[ParityDetection],
    *,
    iou_threshold: float = PARITY_IOU_THRESHOLD,
) -> dict[str, float | int | bool]:
    matches = greedy_match_same_class(source, exported, iou_threshold=iou_threshold)
    both_empty = not source and not exported
    mean_iou = sum(item[2] for item in matches) / len(matches) if matches else 0.0
    mean_conf_diff = (
        sum(abs(left.confidence - right.confidence) for left, right, _ in matches) / len(matches)
        if matches
        else 0.0
    )
    return {
        "source_detection_count": len(source),
        "exported_detection_count": len(exported),
        "matched_detections": len(matches),
        "both_empty": both_empty,
        "mean_matched_iou": round(mean_iou, 4),
        "mean_abs_confidence_difference": round(mean_conf_diff, 4),
    }


def aggregate_parity(frame_results: list[dict[str, float | int | bool]]) -> dict[str, float | int]:
    source_total = sum(int(item["source_detection_count"]) for item in frame_results)
    exported_total = sum(int(item["exported_detection_count"]) for item in frame_results)
    matched_total = sum(int(item["matched_detections"]) for item in frame_results)
    both_empty_frames = sum(1 for item in frame_results if item["both_empty"])
    iou_values = [
        float(item["mean_matched_iou"])
        for item in frame_results
        if int(item["matched_detections"]) > 0
    ]
    conf_values = [
        float(item["mean_abs_confidence_difference"])
        for item in frame_results
        if int(item["matched_detections"]) > 0
    ]
    return {
        "frames_compared": len(frame_results),
        "source_detection_count": source_total,
        "exported_detection_count": exported_total,
        "matched_detections": matched_total,
        "source_match_recall": round(matched_total / source_total, 4) if source_total else 0.0,
        "exported_match_precision": round(matched_total / exported_total, 4)
        if exported_total
        else 0.0,
        "mean_matched_iou": round(sum(iou_values) / len(iou_values), 4) if iou_values else 0.0,
        "mean_abs_confidence_difference": round(sum(conf_values) / len(conf_values), 4)
        if conf_values
        else 0.0,
        "both_empty_frames": both_empty_frames,
    }


def collect_runtime_provenance(
    *,
    device: str,
    imgsz: int,
    benchmark_config: dict[str, Any],
    ultralytics_version: str,
    torch_version: str,
    onnxruntime_version: str | None,
) -> dict[str, Any]:
    operating_system = f"{platform.system()} {platform.release()}"
    return {
        "operating_system": operating_system,
        "machine_architecture": platform.machine(),
        "processor": platform.processor() or "unknown",
        "logical_cpu_count": os.cpu_count(),
        "python_version": platform.python_version(),
        "ultralytics_version": ultralytics_version,
        "torch_version": torch_version,
        "onnxruntime_version": onnxruntime_version,
        "requested_inference_device": device,
        "model_image_size": imgsz,
        "benchmark_configuration": benchmark_config,
        "hardware_scope": "current_machine_only",
        "raspberry_pi_benchmarked": False,
    }


def validation_not_run() -> dict[str, Any]:
    return {"status": "not_run"}


def build_optimization_report(
    *,
    source_artifact: dict[str, Any],
    exported_artifact: dict[str, Any],
    export_format: str,
    precision: str,
    export_arguments: dict[str, Any],
    runtime_provenance: dict[str, Any],
    benchmark_configuration: dict[str, Any],
    source_benchmark: dict[str, Any],
    exported_benchmark: dict[str, Any],
    prediction_parity: dict[str, Any],
    validation: dict[str, Any],
    limitations: list[str],
) -> dict[str, Any]:
    size_stats = size_reduction(
        int(source_artifact["total_bytes"]),
        int(exported_artifact["total_bytes"]),
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "generated_at_utc": datetime.now(UTC).replace(microsecond=0).isoformat().replace(
            "+00:00", "Z"
        ),
        "source_model": source_artifact,
        "exported_model": exported_artifact,
        "export": {
            "format": export_format,
            "precision": precision,
            "arguments": export_arguments,
        },
        "runtime_provenance": runtime_provenance,
        "artifact_comparison": size_stats,
        "benchmark_configuration": benchmark_configuration,
        "source_benchmark": source_benchmark,
        "exported_benchmark": exported_benchmark,
        "prediction_parity": prediction_parity,
        "validation": validation,
        "limitations": limitations,
        "raspberry_pi_benchmarked": False,
    }


REPORT_LIMITATIONS = [
    "Benchmarks run on the current machine only; not Raspberry Pi results.",
    "Backend inference timings come from Ultralytics result.speed.",
    "End-to-end FPS is measured from prediction-call wall time on sampled frames.",
    "Prediction parity compares postprocessed detections; not validation accuracy.",
    "Empty-vs-empty frames are reported separately and do not prove model quality.",
    "FP16 or INT8 speedups must be read from measured numbers in this report.",
]
