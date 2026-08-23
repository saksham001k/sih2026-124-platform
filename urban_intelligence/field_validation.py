"""Truthful field-readiness gates for live Raspberry Pi mission reports."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class ValidationCheck:
    key: str
    passed: bool
    detail: str

    def as_dict(self) -> dict[str, Any]:
        return {"key": self.key, "passed": self.passed, "detail": self.detail}


def evaluate_field_run(
    metrics: dict[str, Any],
    device_status: dict[str, Any],
    *,
    minimum_runtime_s: float = 60.0,
    minimum_captured_frames: int = 300,
) -> dict[str, Any]:
    """Return a claim gate; only all-pass physical runs are field-certified."""
    if minimum_runtime_s <= 0 or minimum_captured_frames < 1:
        raise ValueError("field validation thresholds must be positive")
    device = metrics.get("device", {})
    capture = metrics.get("capture", {})
    compute = metrics.get("compute_budget", {})
    gps = metrics.get("gps") or {}
    raspberry_pi = bool(device.get("raspberry_pi", device_status.get("raspberry_pi", False)))
    execution_environment = str(
        device.get(
            "execution_environment",
            device_status.get("execution_environment", "unknown"),
        )
    )
    runtime_s = float(metrics.get("runtime_seconds", 0) or 0)
    captured_frames = int(capture.get("captured_frames", 0) or 0)
    analytics_attempts = int(metrics.get("analytics_attempts", 0) or 0)
    gps_is_real = metrics.get("gps_source_type") == "real_telemetry"
    serial_nmea = metrics.get("gps_provider") == "serial_nmea"
    gps_has_fix = bool(gps.get("has_fix", False))
    mission_complete = device_status.get("mission_status") == "completed"
    model_errors = int(metrics.get("model_errors", 0) or 0)
    capture_error = str(metrics.get("capture_error", "")).strip()
    overloaded = bool(compute.get("overloaded", True))

    checks = [
        ValidationCheck(
            "physical_raspberry_pi",
            raspberry_pi,
            (
                "Raspberry Pi hardware detected"
                if raspberry_pi
                else "Run was not made on Raspberry Pi"
            ),
        ),
        ValidationCheck(
            "physical_execution_environment",
            execution_environment == "physical_device",
            (
                "Execution environment is a physical device"
                if execution_environment == "physical_device"
                else f"Execution environment is {execution_environment}; physical device required"
            ),
        ),
        ValidationCheck(
            "real_serial_gps",
            gps_is_real and serial_nmea and gps_has_fix,
            (
                "Live serial NMEA fix recorded"
                if gps_is_real and serial_nmea and gps_has_fix
                else "Requires real_telemetry from serial NMEA with a valid fix"
            ),
        ),
        ValidationCheck(
            "runtime_duration",
            runtime_s >= minimum_runtime_s,
            f"Measured {runtime_s:.1f}s; minimum {minimum_runtime_s:.1f}s",
        ),
        ValidationCheck(
            "continuous_capture",
            captured_frames >= minimum_captured_frames,
            f"Captured {captured_frames} frames; minimum {minimum_captured_frames}",
        ),
        ValidationCheck(
            "analytics_executed",
            analytics_attempts > 0,
            f"Analytics attempts: {analytics_attempts}",
        ),
        ValidationCheck(
            "error_free_completion",
            mission_complete and model_errors == 0 and not capture_error,
            (
                "Mission completed without capture/model errors"
                if mission_complete and model_errors == 0 and not capture_error
                else (
                    f"complete={mission_complete}, model_errors={model_errors}, "
                    f"capture_error={capture_error or 'none'}"
                )
            ),
        ),
        ValidationCheck(
            "compute_headroom",
            not overloaded,
            (
                f"Measured utilization {float(compute.get('estimated_utilization', 0)):.1%}"
                if compute
                else "Compute utilization was not measured"
            ),
        ),
    ]
    passed = all(item.passed for item in checks)
    return {
        "schema_version": 1,
        "status": "field_verified" if passed else "not_field_verified",
        "field_verified": passed,
        "checks": [item.as_dict() for item in checks],
        "claim": (
            "Verified on the named Raspberry Pi with live NMEA GPS for this mission only."
            if passed
            else "Do not claim Raspberry Pi field verification from this run."
        ),
    }
