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
    greedy_match_same_class,
    per_frame_parity,
    percentile_95,
    quantize_for_precision,
    sha256_file,
    size_reduction,
    summarize_timings,
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


def test_percentile_and_timing_summary() -> None:
    assert percentile_95([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) == pytest.approx(10.0)
    summary = summarize_timings([10.0, 12.0, 11.0], [20.0, 22.0, 24.0])
    assert summary["sample_count"] == 3
    assert summary["inference_median_ms"] == pytest.approx(11.0)
    assert summary["wall_median_ms"] == pytest.approx(22.0)
    assert summary["end_to_end_fps"] == pytest.approx(45.455, rel=1e-3)
    with pytest.raises(ValueError, match="zero valid benchmark samples"):
        summarize_timings([], [])


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


def test_size_reduction_calculation() -> None:
    stats = size_reduction(6_000_000, 4_000_000)
    assert stats["difference_bytes"] == 2_000_000
    assert stats["reduction_percent"] == pytest.approx(33.3333, rel=1e-3)


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
    )
    encoded = json.dumps(provenance)
    assert "secret-hostname" not in encoded
    assert provenance["hardware_scope"] == "current_machine_only"
    assert provenance["raspberry_pi_benchmarked"] is False


def test_report_schema_and_validation_not_run(tmp_path: Path) -> None:
    source_path = tmp_path / "source.pt"
    source_path.write_bytes(b"source")
    exported_path = tmp_path / "export.onnx"
    exported_path.write_bytes(b"export")
    source = artifact_record(source_path)
    exported = artifact_record(exported_path)
    report = build_optimization_report(
        source_artifact=source,
        exported_artifact=exported,
        export_format="onnx",
        precision="fp32",
        export_arguments={"format": "onnx", "precision": "fp32"},
        runtime_provenance={"hardware_scope": "current_machine_only"},
        benchmark_configuration={"benchmark_frames": 10},
        source_benchmark={"sample_count": 10},
        exported_benchmark={"sample_count": 10},
        prediction_parity={"frames_compared": 10},
        validation=validation_not_run(),
        limitations=["test limitation"],
    )
    assert report["schema_version"] == "1.0"
    assert report["validation"]["status"] == "not_run"
    assert report["raspberry_pi_benchmarked"] is False
    assert "generated_at_utc" in report


def test_optimize_module_imports_without_runtime() -> None:
    import optimize_model

    assert hasattr(optimize_model, "parse_args")
    assert hasattr(optimize_model, "resolve_precision")
