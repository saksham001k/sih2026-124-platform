import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from urban_intelligence.edge_runtime import (
    EvidenceOutbox,
    FrameEnvelope,
    GeoFence,
    InferenceScheduler,
    LatestFrameBuffer,
    ModelSchedule,
    active_geofence_keys,
    build_runtime_report,
    estimate_compute_utilization,
    sample_device_health,
)
from urban_intelligence.gps import GPSPoint


def schedule(
    name: str,
    fps: float,
    priority: int,
    activation: str = "always",
    context_keys: tuple[str, ...] = (),
) -> ModelSchedule:
    return ModelSchedule(
        name=name,
        target_fps=fps,
        priority=priority,
        activation=activation,  # type: ignore[arg-type]
        context_keys=context_keys,
    )


def finish(scheduler: InferenceScheduler, name: str, latency_ms: float = 10.0) -> None:
    scheduler.complete(name, latency_ms=latency_ms)


def test_latest_frame_buffer_never_builds_stale_backlog() -> None:
    buffer = LatestFrameBuffer(capacity=2)
    for index in range(4):
        buffer.push(FrameEnvelope(index, index / 10, f"frame-{index}"))

    frame = buffer.pop_latest()
    assert frame is not None
    assert frame.frame_index == 3
    assert frame.payload == "frame-3"
    assert buffer.snapshot() == {
        "capacity": 2,
        "queue_depth": 0,
        "captured_frames": 4,
        "dropped_analysis_frames": 3,
    }


def test_scheduler_runs_mixed_rates_without_starving_a_model() -> None:
    scheduler = InferenceScheduler(
        [
            schedule("traffic", 5, 100),
            schedule("road", 4, 90),
            schedule("assets", 1, 40),
        ]
    )

    selected = scheduler.acquire_next(now_s=0.0)
    assert selected and selected.name == "traffic"
    finish(scheduler, "traffic")
    selected = scheduler.acquire_next(now_s=0.01)
    assert selected and selected.name == "road"
    finish(scheduler, "road")
    selected = scheduler.acquire_next(now_s=0.02)
    assert selected and selected.name == "assets"
    finish(scheduler, "assets")
    assert scheduler.acquire_next(now_s=0.10) is None
    selected = scheduler.acquire_next(now_s=0.20)
    assert selected and selected.name == "traffic"


def test_triggered_work_preempts_due_background_work_and_expires() -> None:
    scheduler = InferenceScheduler(
        [
            schedule("traffic", 5, 100),
            schedule("anpr", 5, 200, activation="triggered"),
        ]
    )
    scheduler.trigger("anpr", now_s=4.0, duration_s=2.0)
    selected = scheduler.acquire_next(now_s=4.0)
    assert selected and selected.name == "anpr"
    finish(scheduler, "anpr")

    assert scheduler.snapshot(now_s=6.01)["schedules"]["anpr"]["trigger_active"] is False


def test_geofenced_schedule_requires_matching_live_context() -> None:
    scheduler = InferenceScheduler(
        [schedule("school_risk", 2, 150, "geofenced", ("school-zone-7",))]
    )
    assert scheduler.acquire_next(now_s=1.0) is None
    selected = scheduler.acquire_next(
        now_s=1.0,
        active_contexts={"school-zone-7"},
    )
    assert selected and selected.name == "school_risk"


def test_invalid_schedule_and_trigger_settings_are_rejected() -> None:
    with pytest.raises(ValueError, match="target_fps"):
        schedule("road", 0, 1)
    with pytest.raises(ValueError, match="context key"):
        schedule("assets", 1, 1, "geofenced")
    scheduler = InferenceScheduler([schedule("road", 1, 1)])
    with pytest.raises(ValueError, match="not a triggered"):
        scheduler.trigger("road", now_s=0, duration_s=1)
    with pytest.raises(ValueError, match="duration_s"):
        InferenceScheduler([schedule("anpr", 1, 1, "triggered")]).trigger(
            "anpr", now_s=0, duration_s=0
        )


def test_runtime_stats_record_success_and_failure() -> None:
    scheduler = InferenceScheduler([schedule("road", 2, 1)])
    assert scheduler.acquire_next(now_s=0)
    scheduler.complete("road", latency_ms=20)
    assert scheduler.acquire_next(now_s=0.5)
    scheduler.complete("road", latency_ms=40, success=False, error="runtime unavailable")

    stats = scheduler.snapshot(now_s=1)["schedules"]["road"]
    assert stats["attempted"] == 2
    assert stats["succeeded"] == 1
    assert stats["failed"] == 1
    assert stats["mean_latency_ms"] == 30
    assert stats["last_error"] == "runtime unavailable"


def test_compute_utilization_uses_measured_latency_and_flags_overload() -> None:
    schedules = [schedule("traffic", 5, 2), schedule("road", 4, 1)]
    report = estimate_compute_utilization(
        schedules,
        {"traffic": 100, "road": 150},
    )
    assert report["contributions"] == {"traffic": 0.5, "road": 0.6}
    assert report["estimated_utilization"] == 1.1
    assert report["overloaded"] is True


def test_geofence_uses_real_distance_not_coordinate_rounding() -> None:
    fences = [GeoFence("school-zone-7", 28.6139, 77.2090, 100)]
    inside = GPSPoint(0, 28.6141, 77.2091)
    outside = GPSPoint(0, 28.6300, 77.2300)
    assert active_geofence_keys(inside, fences) == {"school-zone-7"}
    assert active_geofence_keys(outside, fences) == set()


def test_evidence_outbox_is_atomic_retryable_and_acknowledgeable(tmp_path: Path) -> None:
    outbox = EvidenceOutbox(tmp_path / "outbox")
    queued = outbox.enqueue(
        "evt-0001",
        {"class": "pothole", "status": "pending_review"},
        queued_at=datetime(2026, 8, 23, tzinfo=UTC),
    )
    assert outbox.pending() == [queued]
    assert not list(queued.parent.glob("*.tmp"))

    outbox.record_failure("evt-0001", "4G unavailable")
    envelope = json.loads(queued.read_text(encoding="utf-8"))
    assert envelope["attempts"] == 1
    assert envelope["last_error"] == "4G unavailable"

    sent = outbox.acknowledge("evt-0001")
    assert sent.parent.name == "sent"
    assert not outbox.pending()


def test_evidence_outbox_rejects_path_traversal_and_symlink_root(tmp_path: Path) -> None:
    outbox = EvidenceOutbox(tmp_path / "outbox")
    with pytest.raises(ValueError, match="unsafe"):
        outbox.enqueue("../../private", {})

    real = tmp_path / "real"
    real.mkdir()
    link = tmp_path / "link"
    link.symlink_to(real, target_is_directory=True)
    with pytest.raises(ValueError, match="symbolic"):
        EvidenceOutbox(link)


def test_device_health_reads_pi_files_without_collecting_identity(tmp_path: Path) -> None:
    device = tmp_path / "model"
    thermal = tmp_path / "temp"
    meminfo = tmp_path / "meminfo"
    device.write_text("Raspberry Pi 5 Model B Rev 1.0\x00", encoding="utf-8")
    thermal.write_text("52345", encoding="utf-8")
    meminfo.write_text(
        "MemTotal:        8192000 kB\nMemAvailable:    4096000 kB\n",
        encoding="utf-8",
    )

    result = sample_device_health(
        device_tree_path=device,
        thermal_path=thermal,
        meminfo_path=meminfo,
    )
    assert result["raspberry_pi"] is True
    assert result["cpu_temperature_c"] == 52.34
    assert result["memory_total_mb"] == 8000
    assert result["memory_available_mb"] == 4000
    assert "hostname" not in result
    assert "username" not in result


def test_runtime_report_separates_capture_rate_from_analytics_rate() -> None:
    buffer = LatestFrameBuffer(2)
    for index in range(3):
        buffer.push(FrameEnvelope(index, index / 30, None))
    buffer.pop_latest()
    scheduler = InferenceScheduler([schedule("road", 2, 1)])
    assert scheduler.acquire_next(now_s=1)
    scheduler.complete("road", latency_ms=20)

    report = build_runtime_report(
        started_at_s=0,
        finished_at_s=2,
        frame_buffer=buffer,
        scheduler=scheduler,
        device_health={"raspberry_pi": False},
    )
    assert report["capture"]["captured_frames"] == 3
    assert report["capture"]["dropped_analysis_frames"] == 2
    assert report["analytics_attempts"] == 1
    assert report["analytics_attempts_per_second"] == 0.5
