"""Pure helpers for edge export benchmarking and prediction parity."""

from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import statistics
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
SAMPLING_MODES = ("uniform", "sequential")
SCHEMA_VERSION = "1.1"
PARITY_IOU_THRESHOLD = 0.50
DEVICE_TREE_MODEL_PATH = Path("/proc/device-tree/model")


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
    sampling: str = "uniform",
) -> None:
    validate_precision_format(export_format, precision)
    if sampling not in SAMPLING_MODES:
        raise ValueError(f"sampling must be one of: {', '.join(SAMPLING_MODES)}")
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


def size_reduction(source_bytes: int, exported_bytes: int) -> dict[str, float | int | str]:
    difference = source_bytes - exported_bytes
    percent = (difference / source_bytes * 100.0) if source_bytes > 0 else 0.0
    if difference >= 0:
        label = "size_reduction"
        display_percent = round(percent, 4)
    else:
        label = "size_increase"
        display_percent = round(abs(percent), 4)
    return {
        "difference_bytes": difference,
        "reduction_percent": round(percent, 4),
        "change_label": label,
        "change_percent": display_percent,
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
    wall_median = float(statistics.median(wall_ms))
    wall_mean = float(statistics.mean(wall_ms))
    total_wall_time_ms = float(sum(wall_ms))
    inference_median = float(statistics.median(inference)) if inference else 0.0
    measured_fps = (
        len(wall_ms) / (total_wall_time_ms / 1000.0) if total_wall_time_ms > 0 else 0.0
    )
    fps_from_median = (1000.0 / wall_median) if wall_median > 0 else 0.0
    return {
        "sample_count": len(wall_ms),
        "inference_median_ms": round(inference_median, 3),
        "inference_p95_ms": round(percentile_95(inference), 3),
        "total_wall_time_ms": round(total_wall_time_ms, 3),
        "wall_mean_ms": round(wall_mean, 3),
        "wall_median_ms": round(wall_median, 3),
        "wall_p95_ms": round(percentile_95(wall_ms), 3),
        "measured_end_to_end_fps": round(measured_fps, 3),
        "fps_from_median_wall_ms": round(fps_from_median, 3),
    }


def uniform_frame_indices(total_frames: int, sample_count: int) -> list[int]:
    """Deterministic unique frame indices spread across the video."""
    if total_frames < 1:
        raise ValueError("total_frames must be at least 1")
    if sample_count < 1:
        raise ValueError("sample_count must be at least 1")
    count = min(sample_count, total_frames)
    if count == 1:
        return [0]
    if count == total_frames:
        return list(range(total_frames))
    # Integer arithmetic guarantees unique, inclusive endpoints when count <= total.
    return [((index * (total_frames - 1)) // (count - 1)) for index in range(count)]


def sequential_frame_indices(total_frames: int, sample_count: int) -> list[int]:
    if total_frames < 1:
        raise ValueError("total_frames must be at least 1")
    if sample_count < 1:
        raise ValueError("sample_count must be at least 1")
    return list(range(min(sample_count, total_frames)))


def select_frame_indices(
    total_frames: int,
    sample_count: int,
    *,
    sampling: str = "uniform",
) -> list[int]:
    if sampling == "uniform":
        return uniform_frame_indices(total_frames, sample_count)
    if sampling == "sequential":
        return sequential_frame_indices(total_frames, sample_count)
    raise ValueError(f"sampling must be one of: {', '.join(SAMPLING_MODES)}")


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
    matched_iou_sum = sum(item[2] for item in matches)
    matched_conf_diff_sum = sum(
        abs(left.confidence - right.confidence) for left, right, _ in matches
    )
    matched_count = len(matches)
    return {
        "source_detection_count": len(source),
        "exported_detection_count": len(exported),
        "matched_detections": matched_count,
        "both_empty": both_empty,
        "matched_iou_sum": round(matched_iou_sum, 6),
        "matched_confidence_difference_sum": round(matched_conf_diff_sum, 6),
        "mean_matched_iou": round(matched_iou_sum / matched_count, 4) if matched_count else 0.0,
        "mean_abs_confidence_difference": (
            round(matched_conf_diff_sum / matched_count, 4) if matched_count else 0.0
        ),
    }


def aggregate_parity(frame_results: list[dict[str, float | int | bool]]) -> dict[str, float | int]:
    source_total = sum(int(item["source_detection_count"]) for item in frame_results)
    exported_total = sum(int(item["exported_detection_count"]) for item in frame_results)
    matched_total = sum(int(item["matched_detections"]) for item in frame_results)
    both_empty_frames = sum(1 for item in frame_results if item["both_empty"])
    total_matched_iou = sum(float(item.get("matched_iou_sum", 0.0)) for item in frame_results)
    total_conf_diff = sum(
        float(item.get("matched_confidence_difference_sum", 0.0)) for item in frame_results
    )
    return {
        "frames_compared": len(frame_results),
        "source_detection_count": source_total,
        "exported_detection_count": exported_total,
        "matched_detections": matched_total,
        "matched_iou_sum": round(total_matched_iou, 6),
        "matched_confidence_difference_sum": round(total_conf_diff, 6),
        "source_match_recall": round(matched_total / source_total, 4) if source_total else 0.0,
        "exported_match_precision": round(matched_total / exported_total, 4)
        if exported_total
        else 0.0,
        "mean_matched_iou": (
            round(total_matched_iou / matched_total, 4) if matched_total else 0.0
        ),
        "mean_abs_confidence_difference": (
            round(total_conf_diff / matched_total, 4) if matched_total else 0.0
        ),
        "both_empty_frames": both_empty_frames,
    }


def detect_raspberry_pi(
    *,
    model_path: Path = DEVICE_TREE_MODEL_PATH,
    system_name: str | None = None,
) -> dict[str, Any]:
    """Conservatively detect Raspberry Pi from device-tree model text."""
    system = system_name if system_name is not None else platform.system()
    result: dict[str, Any] = {
        "raspberry_pi_benchmarked": False,
        "device_model": None,
    }
    if system != "Linux":
        return result
    try:
        raw = model_path.read_bytes()
    except (FileNotFoundError, PermissionError, OSError):
        return result
    text = raw.split(b"\x00", 1)[0].decode("utf-8", errors="ignore").strip()
    result["device_model"] = text or None
    if text and "raspberry pi" in text.lower():
        result["raspberry_pi_benchmarked"] = True
    return result


def collect_runtime_provenance(
    *,
    device: str,
    imgsz: int,
    benchmark_config: dict[str, Any],
    ultralytics_version: str,
    torch_version: str,
    onnxruntime_version: str | None,
    raspberry_pi_info: dict[str, Any] | None = None,
) -> dict[str, Any]:
    operating_system = f"{platform.system()} {platform.release()}"
    pi_info = raspberry_pi_info if raspberry_pi_info is not None else detect_raspberry_pi()
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
        "raspberry_pi_benchmarked": bool(pi_info["raspberry_pi_benchmarked"]),
        "device_model": pi_info.get("device_model"),
    }


def validation_not_run() -> dict[str, Any]:
    return {"status": "not_run"}


def package_validation_metrics(
    source_metrics: dict[str, float],
    exported_metrics: dict[str, float],
) -> dict[str, Any]:
    differences: dict[str, dict[str, float]] = {}
    for key in source_metrics:
        signed = round(exported_metrics[key] - source_metrics[key], 4)
        differences[key] = {
            "signed_difference": signed,
            "absolute_difference": round(abs(signed), 4),
        }
    return {
        "status": "completed",
        "source": source_metrics,
        "exported": exported_metrics,
        "differences": differences,
    }


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
    raspberry_pi = bool(runtime_provenance.get("raspberry_pi_benchmarked", False))
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
        "raspberry_pi_benchmarked": raspberry_pi,
    }


REPORT_LIMITATIONS = [
    "Benchmarks apply only to the current machine (hardware_scope).",
    "Backend inference timings come from Ultralytics result.speed.",
    "measured_end_to_end_fps uses total wall time across timed samples.",
    "fps_from_median_wall_ms is median-derived theoretical throughput only.",
    "Prediction parity is not validation accuracy.",
    "A single matched detection is insufficient parity evidence.",
    "Empty-vs-empty frames do not prove model quality.",
    "FP16 or INT8 speedups must be read from measured numbers.",
]
