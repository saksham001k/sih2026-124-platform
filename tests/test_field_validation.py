import pytest

from urban_intelligence.field_validation import evaluate_field_run


def valid_metrics() -> dict:
    return {
        "runtime_seconds": 120,
        "capture": {"captured_frames": 1800},
        "analytics_attempts": 500,
        "device": {"raspberry_pi": True, "device_model": "Raspberry Pi 5"},
        "gps_source_type": "real_telemetry",
        "gps_provider": "serial_nmea",
        "gps": {"has_fix": True},
        "model_errors": 0,
        "capture_error": "",
        "compute_budget": {"overloaded": False, "estimated_utilization": 0.62},
    }


def test_all_physical_gates_are_required_for_field_verification() -> None:
    report = evaluate_field_run(valid_metrics(), {"mission_status": "completed"})
    assert report["field_verified"] is True
    assert all(item["passed"] for item in report["checks"])


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("gps_source_type", "synthetic_demo"),
        ("gps_provider", "timestamped_csv"),
        ("runtime_seconds", 10),
        ("analytics_attempts", 0),
        ("model_errors", 1),
    ],
)
def test_any_failed_claim_gate_blocks_field_verification(key: str, value: object) -> None:
    metrics = valid_metrics()
    metrics[key] = value
    report = evaluate_field_run(metrics, {"mission_status": "completed"})
    assert report["field_verified"] is False
    assert report["status"] == "not_field_verified"


def test_non_pi_and_overloaded_run_cannot_be_mislabelled() -> None:
    metrics = valid_metrics()
    metrics["device"] = {"raspberry_pi": False}
    metrics["compute_budget"] = {"overloaded": True, "estimated_utilization": 1.2}
    report = evaluate_field_run(metrics, {"mission_status": "completed"})
    failed = {item["key"] for item in report["checks"] if not item["passed"]}
    assert failed == {"physical_raspberry_pi", "compute_headroom"}
