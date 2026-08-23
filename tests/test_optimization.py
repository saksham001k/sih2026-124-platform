"""Tests for edge export benchmarking helpers."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from urban_intelligence.optimization import (
    ParityDetection,
    aggregate_parity,
    artifact_manifest,
    artifact_record,
    artifact_size_bytes,
    build_optimization_report,
    collect_runtime_provenance,
    detect_raspberry_pi,
    greedy_match_same_class,
    package_validation_metrics,
    per_frame_parity,
    percentile_95,
    quantize_for_precision,
    select_frame_indices,
    sha256_file,
    size_reduction,
    summarize_timings,
    uniform_frame_indices,
    validate_benchmark_settings,
    validate_precision_format,
    validation_not_run,
)


def test_artifact_size_for_file_and_directory(tmp_path: Path) -> None:
    file_path = tmp_path / "model.bin"
    file_path.write_bytes(b"abc")
    directory = tmp_path / "export"
    directory.mkdir()
    (directory / "a.bin").write_bytes(b"12")
    (directory / "nested" / "b.bin").parent.mkdir(parents=True)
    (directory / "nested" / "b.bin").write_bytes(b"345")
    assert artifact_size_bytes(file_path) == 3
    assert artifact_size_bytes(directory) == 5


def test_sha256_file_and_directory_manifest(tmp_path: Path) -> None:
    left = tmp_path / "left.txt"
    right = tmp_path / "right.txt"
    left.write_text("same", encoding="utf-8")
    right.write_text("same", encoding="utf-8")
    assert sha256_file(left) == sha256_file(right)
    export_dir = tmp_path / "export"
    export_dir.mkdir()
    (export_dir / "one.bin").write_bytes(b"1")
    (export_dir / "two.bin").write_bytes(b"2")
    manifest = artifact_manifest(export_dir)
    assert len(manifest) == 2
    assert manifest[0]["relative_path"] < manifest[1]["relative_path"]
    record = artifact_record(export_dir)
    assert record["type"] == "directory"
    assert record["manifest_sha256"]


def test_percentile_and_timing_summary_with_unsorted_samples() -> None:
    assert percentile_95([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) == pytest.approx(10.0)
    # Deliberately unordered even-length wall times to catch unsorted median bugs.
    summary = summarize_timings(
        [12.0, 10.0, 14.0, 11.0],
        [40.0, 10.0, 30.0, 20.0],
    )
    assert summary["sample_count"] == 4
    assert summary["inference_median_ms"] == pytest.approx(11.5)
    assert summary["wall_median_ms"] == pytest.approx(25.0)
    assert summary["wall_mean_ms"] == pytest.approx(25.0)
    assert summary["total_wall_time_ms"] == pytest.approx(100.0)
    assert summary["measured_end_to_end_fps"] == pytest.approx(40.0)
    assert summary["fps_from_median_wall_ms"] == pytest.approx(40.0)
    with pytest.raises(ValueError, match="zero valid benchmark samples"):
        summarize_timings([], [])


def test_measured_fps_uses_total_wall_time_not_median() -> None:
    # Uneven samples: median would give a different FPS than total throughput.
    summary = summarize_timings([], [10.0, 10.0, 10.0, 70.0])
    assert summary["wall_median_ms"] == pytest.approx(10.0)
    assert summary["measured_end_to_end_fps"] == pytest.approx(40.0)
    assert summary["fps_from_median_wall_ms"] == pytest.approx(100.0)


def test_valid_precision_format_combinations() -> None:
    validate_precision_format("onnx", "fp32")
    validate_precision_format("onnx", "fp16")
    validate_precision_format("onnx", "int8")
    validate_precision_format("ncnn", "fp16")
    assert quantize_for_precision("fp32") is None
    assert quantize_for_precision("fp16") == 16
    assert quantize_for_precision("int8") == 8


def test_rejects_ncnn_int8_and_requires_calibration_data() -> None:
    with pytest.raises(ValueError, match="NCNN export does not support INT8"):
        validate_precision_format("ncnn", "int8")
    with pytest.raises(ValueError, match="INT8 export requires"):
        validate_benchmark_settings(
            warmup_runs=1,
            benchmark_frames=10,
            imgsz=640,
            confidence=0.25,
            iou=0.7,
            device="cpu",
            precision="int8",
            export_format="onnx",
        )


def test_rejects_invalid_benchmark_settings() -> None:
    base = dict(
        warmup_runs=1,
        benchmark_frames=10,
        imgsz=640,
        confidence=0.25,
        iou=0.7,
        device="cpu",
    )
    validate_benchmark_settings(**base)
    with pytest.raises(ValueError, match="warmup_runs"):
        validate_benchmark_settings(**{**base, "warmup_runs": -1})
    with pytest.raises(ValueError, match="benchmark_frames"):
        validate_benchmark_settings(**{**base, "benchmark_frames": 0})
    with pytest.raises(ValueError, match="imgsz"):
        validate_benchmark_settings(**{**base, "imgsz": 16})
    with pytest.raises(ValueError, match="device"):
        validate_benchmark_settings(**{**base, "device": ""})
    with pytest.raises(ValueError, match="sampling"):
        validate_benchmark_settings(**{**base, "sampling": "random"})


def test_bbox_iou_and_same_class_matching() -> None:
    source = [
        ParityDetection(0, "car", 0.9, (0, 0, 10, 10)),
        ParityDetection(1, "truck", 0.8, (20, 20, 30, 30)),
    ]
    exported = [
        ParityDetection(0, "car", 0.85, (1, 1, 11, 11)),
        ParityDetection(0, "car", 0.7, (100, 100, 110, 110)),
    ]
    matches = greedy_match_same_class(source, exported, iou_threshold=0.5)
    assert len(matches) == 1
    assert matches[0][0].class_name == "car"
    assert matches[0][2] > 0.5


def test_wrong_class_boxes_are_not_matched() -> None:
    source = [ParityDetection(0, "car", 0.9, (0, 0, 10, 10))]
    exported = [ParityDetection(1, "truck", 0.9, (0, 0, 10, 10))]
    assert greedy_match_same_class(source, exported) == []


def test_parity_aggregation_and_empty_frames() -> None:
    frame_results = [
        per_frame_parity(
            [ParityDetection(0, "car", 0.9, (0, 0, 10, 10))],
            [ParityDetection(0, "car", 0.8, (1, 1, 11, 11))],
        ),
        per_frame_parity([], []),
    ]
    aggregated = aggregate_parity(frame_results)
    assert aggregated["frames_compared"] == 2
    assert aggregated["matched_detections"] == 1
    assert aggregated["both_empty_frames"] == 1
    assert aggregated["source_match_recall"] == pytest.approx(1.0)


def test_parity_aggregation_is_detection_weighted_not_frame_weighted() -> None:
    # Frame A: one match with IoU 1.0
    # Frame B: three matches with IoU 0.5 each
    # Frame-weighted mean would be (1.0 + 0.5) / 2 = 0.75
    # Detection-weighted mean is (1.0 + 0.5 + 0.5 + 0.5) / 4 = 0.625
    frame_a = {
        "source_detection_count": 1,
        "exported_detection_count": 1,
        "matched_detections": 1,
        "both_empty": False,
        "matched_iou_sum": 1.0,
        "matched_confidence_difference_sum": 0.10,
        "mean_matched_iou": 1.0,
        "mean_abs_confidence_difference": 0.10,
    }
    frame_b = {
        "source_detection_count": 3,
        "exported_detection_count": 3,
        "matched_detections": 3,
        "both_empty": False,
        "matched_iou_sum": 1.5,
        "matched_confidence_difference_sum": 0.30,
        "mean_matched_iou": 0.5,
        "mean_abs_confidence_difference": 0.10,
    }
    aggregated = aggregate_parity([frame_a, frame_b])
    assert aggregated["matched_detections"] == 4
    assert aggregated["mean_matched_iou"] == pytest.approx(0.625)
    assert aggregated["mean_abs_confidence_difference"] == pytest.approx(0.1)
    assert aggregated["matched_iou_sum"] == pytest.approx(2.5)


def test_uniform_and_sequential_frame_indices() -> None:
    assert uniform_frame_indices(100, 5) == [0, 24, 49, 74, 99]
    assert uniform_frame_indices(10, 10) == list(range(10))
    assert uniform_frame_indices(1, 50) == [0]
    assert select_frame_indices(8, 4, sampling="sequential") == [0, 1, 2, 3]
    assert select_frame_indices(8, 4, sampling="uniform") == [0, 2, 4, 7]
    # Short video: never duplicate indices for parity.
    assert len(set(uniform_frame_indices(3, 50))) == 3
    # Requested count is honored when the video is long enough.
    assert len(uniform_frame_indices(1899, 50)) == 50
    assert len(set(uniform_frame_indices(1899, 50))) == 50


def test_size_reduction_and_increase_labels() -> None:
    reduction = size_reduction(6_000_000, 4_000_000)
    assert reduction["difference_bytes"] == 2_000_000
    assert reduction["change_label"] == "size_reduction"
    assert reduction["change_percent"] == pytest.approx(33.3333, rel=1e-3)

    increase = size_reduction(6_000_000, 12_000_000)
    assert increase["difference_bytes"] == -6_000_000
    assert increase["change_label"] == "size_increase"
    assert increase["change_percent"] == pytest.approx(100.0)


def test_validation_differences_signed_and_absolute() -> None:
    higher = package_validation_metrics(
        {"precision": 0.80, "recall": 0.70, "map50": 0.60, "map50_95": 0.40},
        {"precision": 0.85, "recall": 0.68, "map50": 0.60, "map50_95": 0.35},
    )
    assert higher["differences"]["precision"]["signed_difference"] == pytest.approx(0.05)
    assert higher["differences"]["precision"]["absolute_difference"] == pytest.approx(0.05)
    assert higher["differences"]["recall"]["signed_difference"] == pytest.approx(-0.02)
    assert higher["differences"]["recall"]["absolute_difference"] == pytest.approx(0.02)
    assert higher["differences"]["map50_95"]["signed_difference"] == pytest.approx(-0.05)
    assert higher["differences"]["map50_95"]["absolute_difference"] == pytest.approx(0.05)


def test_runtime_metadata_excludes_hostname(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("platform.node", lambda: "secret-hostname")
    provenance = collect_runtime_provenance(
        device="cpu",
        imgsz=640,
        benchmark_config={"benchmark_frames": 10},
        ultralytics_version="8.4.60",
        torch_version="2.2.0",
        onnxruntime_version=None,
        raspberry_pi_info={"raspberry_pi_benchmarked": False, "device_model": None},
    )
    encoded = json.dumps(provenance)
    assert "secret-hostname" not in encoded
    assert provenance["hardware_scope"] == "current_machine_only"
    assert provenance["raspberry_pi_benchmarked"] is False


def test_detect_raspberry_pi_linux_and_non_linux(
    tmp_path: Path,
) -> None:
    model_file = tmp_path / "model"
    model_file.write_bytes(b"Raspberry Pi 5 Model B Rev 1.0\x00extra")
    detected = detect_raspberry_pi(model_path=model_file, system_name="Linux")
    assert detected["raspberry_pi_benchmarked"] is True
    assert "Raspberry Pi 5" in detected["device_model"]

    mac = detect_raspberry_pi(model_path=model_file, system_name="Darwin")
    assert mac["raspberry_pi_benchmarked"] is False
    assert mac["device_model"] is None

    missing = detect_raspberry_pi(
        model_path=tmp_path / "missing",
        system_name="Linux",
    )
    assert missing["raspberry_pi_benchmarked"] is False
    assert missing["device_model"] is None


def test_report_schema_uses_detected_raspberry_pi_flag(tmp_path: Path) -> None:
    source_path = tmp_path / "source.pt"
    source_path.write_bytes(b"source")
    exported_path = tmp_path / "export.onnx"
    exported_path.write_bytes(b"export")
    provenance = {
        "hardware_scope": "current_machine_only",
        "raspberry_pi_benchmarked": True,
        "device_model": "Raspberry Pi 4 Model B",
    }
    report = build_optimization_report(
        source_artifact=artifact_record(source_path),
        exported_artifact=artifact_record(exported_path),
        export_format="onnx",
        precision="fp32",
        export_arguments={"format": "onnx", "precision": "fp32", "device": "cpu"},
        runtime_provenance=provenance,
        benchmark_configuration={"benchmark_frames": 10},
        source_benchmark={"sample_count": 10},
        exported_benchmark={"sample_count": 10},
        prediction_parity={"frames_compared": 10},
        validation=validation_not_run(),
        limitations=["test limitation"],
    )
    assert report["schema_version"] == "1.1"
    assert report["validation"]["status"] == "not_run"
    assert report["raspberry_pi_benchmarked"] is True
    assert report["runtime_provenance"]["raspberry_pi_benchmarked"] is True
    assert "generated_at_utc" in report


def test_optimize_module_imports_without_runtime() -> None:
    import optimize_model

    assert hasattr(optimize_model, "parse_args")
    assert hasattr(optimize_model, "resolve_precision")
    assert hasattr(optimize_model, "load_benchmark_frames")
