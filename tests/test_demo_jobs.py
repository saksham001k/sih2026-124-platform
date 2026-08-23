import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from urban_intelligence.demo_jobs import (
    AnalysisRequest,
    UploadValidationError,
    VideoMetadata,
    correlate_incidents,
    generate_run_id,
    list_completed_runs,
    load_manifest,
    modules_for_profile,
    prepare_run,
    quality_profile_settings,
    run_analysis,
    validate_upload,
)


def fake_video_probe(path: Path) -> VideoMetadata:
    assert path.name == "video.mp4"
    assert path.read_bytes() == b"video-data"
    return VideoMetadata(
        frame_count=300,
        fps=30.0,
        duration_s=10.0,
        width=1280,
        height=720,
    )


def gps_payload() -> bytes:
    return b"timestamp,lat,lon\n0,28.6,77.2\n1,28.7,77.3\n"


def test_scan_profiles_are_stable_and_custom_is_ordered() -> None:
    assert modules_for_profile("quick") == ("road", "traffic")
    assert modules_for_profile("full") == ("road", "traffic", "assets", "anpr")
    assert modules_for_profile("custom", ("anpr", "road")) == ("road", "anpr")


def test_quality_profiles_make_recall_precision_tradeoff_explicit() -> None:
    recall = quality_profile_settings("high_recall")
    strict = quality_profile_settings("strict")
    assert recall["road_confidence"] < strict["road_confidence"]
    assert recall["road_image_size"] > strict["road_image_size"]
    assert recall["road_min_hits"] < strict["road_min_hits"]
    with pytest.raises(ValueError, match="quality profile"):
        quality_profile_settings("perfect")


def test_incident_correlation_links_masked_anpr_only(tmp_path: Path) -> None:
    traffic = tmp_path / "traffic"
    anpr = tmp_path / "anpr"
    traffic.mkdir()
    anpr.mkdir()
    (traffic / "safety_events.json").write_text(
        json.dumps(
            [
                {
                    "event_id": "safety-1",
                    "event_type": "suspected_hit_and_run",
                    "video_time_s": 10,
                    "lat": 28.6,
                    "lon": 77.2,
                }
            ]
        ),
        encoding="utf-8",
    )
    (anpr / "anpr_events.json").write_text(
        json.dumps(
            [
                {
                    "event_id": "anpr-1",
                    "video_time_s": 11,
                    "lat": 28.6,
                    "lon": 77.2,
                    "masked_plate": "KA*****65",
                    "plate_text": "MUST-NOT-COPY",
                }
            ]
        ),
        encoding="utf-8",
    )
    summary = correlate_incidents(tmp_path)
    records = json.loads((tmp_path / "incidents/incidents.json").read_text())
    assert summary["anpr_candidates_linked"] == 1
    assert records[0]["masked_plate"] == "KA*****65"
    assert "plate_text" not in records[0]


def test_custom_profile_rejects_empty_and_unknown_modules() -> None:
    with pytest.raises(ValueError, match="at least one"):
        modules_for_profile("custom")
    with pytest.raises(ValueError, match="unsupported"):
        modules_for_profile("custom", ("road", "cloud_upload"))


def test_run_id_contains_only_safe_components() -> None:
    value = generate_run_id(
        datetime(2026, 8, 23, 12, 34, 56, tzinfo=UTC),
        token="A1B2C3",
    )
    assert value == "run-20260823T123456Z-a1b2c3"


def test_prepare_run_rejects_path_traversal_run_id(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="run_id"):
        prepare_run(
            runs_root=tmp_path / "runs",
            original_video_name="clip.mp4",
            video_payload=b"video-data",
            gps_source_type="synthetic_demo",
            modules=("road",),
            scan_profile="custom",
            synthetic_gps_path=tmp_path / "gps.csv",
            run_id="../../escape",
            video_probe=fake_video_probe,
        )


def test_upload_validation_rejects_empty_large_and_unsupported_files() -> None:
    with pytest.raises(UploadValidationError, match="empty"):
        validate_upload("clip.mp4", b"", allowed_extensions={".mp4"})
    with pytest.raises(UploadValidationError, match="exceeds"):
        validate_upload("clip.mp4", b"12", allowed_extensions={".mp4"}, max_bytes=1)
    with pytest.raises(UploadValidationError, match="Unsupported"):
        validate_upload("clip.exe", b"1", allowed_extensions={".mp4"})


def test_prepare_run_uses_safe_names_and_real_gps(tmp_path: Path) -> None:
    request = prepare_run(
        runs_root=tmp_path / "runs",
        original_video_name="../../private/incident.mp4",
        video_payload=b"video-data",
        gps_source_type="real_telemetry",
        modules=("road", "traffic"),
        scan_profile="quick",
        gps_payload=gps_payload(),
        original_gps_name="../../telemetry.csv",
        run_id="run-safe",
        video_probe=fake_video_probe,
    )

    assert request.input_path == tmp_path / "runs/run-safe/input/video.mp4"
    assert request.gps_path == tmp_path / "runs/run-safe/input/gps.csv"
    manifest = load_manifest(request.run_dir)
    assert manifest["input"]["video"]["original_name"] == "incident.mp4"
    assert manifest["input"]["gps"]["original_name"] == "telemetry.csv"
    assert manifest["gps_source_type"] == "real_telemetry"
    assert manifest["quality_profile"] == "high_recall"
    assert set(manifest["stages"]) == {"road", "traffic"}


def test_prepare_run_copies_and_labels_synthetic_gps(tmp_path: Path) -> None:
    synthetic = tmp_path / "source.csv"
    synthetic.write_bytes(gps_payload())
    request = prepare_run(
        runs_root=tmp_path / "runs",
        original_video_name="clip.mp4",
        video_payload=b"video-data",
        gps_source_type="synthetic_demo",
        modules=("road",),
        scan_profile="custom",
        synthetic_gps_path=synthetic,
        run_id="run-demo",
        video_probe=fake_video_probe,
    )
    assert request.gps_path.read_bytes() == gps_payload()
    assert load_manifest(request.run_dir)["gps_source_type"] == "synthetic_demo"


def test_prepare_run_removes_new_directory_after_invalid_gps(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="requires lat/lon"):
        prepare_run(
            runs_root=tmp_path / "runs",
            original_video_name="clip.mp4",
            video_payload=b"video-data",
            gps_source_type="real_telemetry",
            modules=("traffic",),
            scan_profile="custom",
            gps_payload=b"timestamp,speed\n0,20\n",
            run_id="run-invalid",
            video_probe=fake_video_probe,
        )
    assert not (tmp_path / "runs/run-invalid").exists()


def test_run_analysis_preserves_success_when_another_stage_fails(tmp_path: Path) -> None:
    run_dir = tmp_path / "run-test"
    run_dir.mkdir()
    manifest = {
        "run_id": "run-test",
        "status": "ready",
        "stages": {
            "road": {"status": "queued"},
            "traffic": {"status": "queued"},
        },
        "outputs": {},
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    request = AnalysisRequest(
        run_id="run-test",
        run_dir=run_dir,
        input_path=run_dir / "video.mp4",
        gps_path=run_dir / "gps.csv",
        gps_source_type="synthetic_demo",
        modules=("road", "traffic"),
        scan_profile="quick",
    )
    updates: list[tuple[str, str, int]] = []

    def road_runner(_: AnalysisRequest) -> dict[str, int]:
        return {"confirmed_events": 2}

    def traffic_runner(_: AnalysisRequest) -> dict[str, int]:
        raise RuntimeError("traffic model unavailable")

    result = run_analysis(
        request,
        stage_runners={"road": road_runner, "traffic": traffic_runner},
        progress=lambda module, status, message, completed, total: updates.append(
            (module, status, completed)
        ),
    )

    assert result["status"] == "completed_with_errors"
    assert result["successful_stages"] == 1
    assert result["failed_stages"] == 1
    assert result["stages"]["road"]["status"] == "completed"
    assert result["stages"]["traffic"]["status"] == "failed"
    assert result["outputs"]["road"]["summary"]["confirmed_events"] == 2
    assert updates[-1] == ("traffic", "failed", 2)


def test_list_runs_requires_a_manifest_and_returns_newest_first(tmp_path: Path) -> None:
    older = tmp_path / "older"
    newer = tmp_path / "newer"
    ignored = tmp_path / "ignored"
    for path in (older, newer, ignored):
        path.mkdir()
    (older / "manifest.json").write_text("{}", encoding="utf-8")
    (newer / "manifest.json").write_text("{}", encoding="utf-8")
    older.touch()
    newer.touch()

    runs = list_completed_runs(tmp_path)
    assert set(runs) == {older, newer}
