"""Deterministic application-level Raspberry Pi mission emulation.

This harness exercises DrishtiPath's production adapters without claiming ARM CPU,
camera-driver, thermal, or physical Raspberry Pi validation. It intentionally reports
``execution_environment=emulated`` and verifies that the physical field gate remains shut.
"""

from __future__ import annotations

import argparse
import json
import platform
import time
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from edge_agent import RawDetection, RunnerSpec, run_agent
from urban_intelligence.delivery import DeliveryConfig, DeliveryResponse, EvidenceDeliveryClient
from urban_intelligence.edge_runtime import EvidenceOutbox, write_json_atomic
from urban_intelligence.field_validation import evaluate_field_run
from urban_intelligence.fleet import FleetIngestionService, FleetStore
from urban_intelligence.nmea import SerialNMEAGPS, nmea_checksum


def _rmc_sentence() -> bytes:
    payload = "GPRMC,123519,A,2836.8340,N,07712.5400,E,12.0,084.4,230826,,,A"
    return f"${payload}*{nmea_checksum(payload):02X}\r\n".encode("ascii")


class VirtualSerial:
    """Small serial-port double that continuously emits valid NMEA RMC fixes."""

    def __init__(self, sentence: bytes, *, delay_s: float = 0.01) -> None:
        self.sentence = sentence
        self.delay_s = delay_s
        self.closed = False

    def readline(self) -> bytes:
        if self.closed:
            return b""
        time.sleep(self.delay_s)
        return self.sentence

    def close(self) -> None:
        self.closed = True


class VirtualCapture:
    """Synthetic 720p dashcam source with real-time-like frame pacing."""

    def __init__(self, frame_count: int, frame_delay_s: float) -> None:
        self.frame_count = frame_count
        self.frame_delay_s = frame_delay_s
        self.index = 0
        self.released = False

    def isOpened(self) -> bool:
        return True

    def get(self, _: int) -> float:
        return 30.0

    def read(self) -> tuple[bool, VirtualFrame | None]:
        if self.index >= self.frame_count:
            return False, None
        time.sleep(self.frame_delay_s)
        frame = VirtualFrame()
        self.index += 1
        return True, frame

    def release(self) -> None:
        self.released = True


class VirtualCV2:
    CAP_PROP_FPS = 5

    def __init__(self, *, frame_count: int, frame_delay_s: float) -> None:
        self.capture = VirtualCapture(frame_count, frame_delay_s)

    def VideoCapture(self, _: object) -> VirtualCapture:  # noqa: N802
        return self.capture

    @staticmethod
    def imwrite(path: str, image: VirtualFrame) -> bool:
        if image.size == 0:
            return False
        Path(path).write_bytes(b"virtual-pi-jpeg-evidence\n")
        return True


class VirtualFrame:
    """Dependency-free ndarray surface used by crop/evidence code."""

    shape = (720, 1280, 3)
    size = 720 * 1280 * 3

    def __getitem__(self, _: object) -> VirtualFrame:
        return self


@dataclass(slots=True)
class VirtualRunner:
    spec: RunnerSpec

    def infer(self, _: VirtualFrame) -> list[RawDetection]:
        latency_s = {
            "traffic": 0.006,
            "road_damage": 0.009,
            "urban_assets": 0.012,
        }[self.spec.name]
        time.sleep(latency_s)
        if self.spec.name == "traffic":
            return [RawDetection("bus", 0.91, (360, 250, 780, 690), 42)]
        if self.spec.name == "road_damage":
            return [RawDetection("pothole", 0.88, (520, 510, 720, 650), 7)]
        return [RawDetection("waterlogging", 0.84, (180, 500, 1080, 700), 11)]


def _nmea_factory(device: str, *, baudrate: int) -> SerialNMEAGPS:
    serial = VirtualSerial(_rmc_sentence())
    return SerialNMEAGPS(
        device,
        baudrate=baudrate,
        serial_factory=lambda *_args, **_kwargs: serial,
    )


def _virtual_health() -> dict[str, Any]:
    return {
        "device_model": "Raspberry Pi 4 Model B (application emulator)",
        "raspberry_pi": True,
        "execution_environment": "emulated",
        "platform": platform.system() or "unknown",
        "machine": platform.machine() or "unknown",
        "logical_cpu_count": 4,
        "load_1m": 0.7,
        "cpu_temperature_c": 52.0,
        "memory_total_mb": 4096.0,
        "memory_available_mb": 2816.0,
    }


def _agent_args(output_dir: Path) -> argparse.Namespace:
    return argparse.Namespace(
        output_dir=str(output_dir),
        source="0",
        vehicle_id="virtual-bus-042",
        mission_id="virtual-pi-mission",
        route_id="route-blue",
        gps_csv=None,
        fixed_gps=None,
        gps_nmea_device="virtual://nmea-gps",
        gps_baud=9600,
        gps_fix_timeout_s=2.0,
        gps_max_age_s=2.0,
        gps_source_type="real_telemetry",
        geofences=None,
        traffic_model="virtual-traffic.ncnn",
        traffic_confidence=0.25,
        road_model="virtual-road-damage.ncnn",
        road_confidence=0.20,
        asset_model="virtual-urban-assets.ncnn",
        asset_confidence=0.25,
        image_size=416,
        device="cpu",
        profile="pi4",
        buffer_frames=3,
        duration_s=0,
        health_interval_s=0.5,
    )


def _deliver_to_virtual_centre(mission_dir: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    outbox = EvidenceOutbox(mission_dir / "outbox")
    store = FleetStore(mission_dir / "central" / "fleet.db")
    token = "virtual-pi-test-token"
    service = FleetIngestionService(store, token)

    def transport(
        _endpoint: str,
        body: bytes,
        headers: Mapping[str, str],
        _timeout_s: float,
    ) -> DeliveryResponse:
        status, payload = service.handle(
            body,
            authorization=str(headers.get("Authorization", "")),
            idempotency_key=str(headers.get("Idempotency-Key", "")),
        )
        return DeliveryResponse(status, json.dumps(payload))

    client = EvidenceDeliveryClient(
        DeliveryConfig(
            endpoint="http://virtual-command-centre/v1/evidence",
            max_attempts_per_run=1,
            base_backoff_s=0,
        ),
        transport=transport,
        environment={"DRISHTIPATH_INGEST_TOKEN": token},
        sleep=lambda _: None,
    )
    delivery = client.deliver(outbox)

    sent = sorted((mission_dir / "outbox" / "sent").glob("*.json"))
    duplicate_statuses: list[int] = []
    for path in sent:
        body = path.read_bytes()
        envelope = json.loads(body)
        status, _ = service.handle(
            body,
            authorization=f"Bearer {token}",
            idempotency_key=str(envelope["event_id"]),
        )
        duplicate_statuses.append(status)
    store.export(mission_dir / "central" / "export")
    fleet = store.summary()
    fleet["duplicate_replay_statuses"] = duplicate_statuses
    return delivery, fleet


def run_virtual_pi_test(
    output_dir: Path,
    *,
    frame_count: int = 240,
    frame_delay_s: float = 0.02,
) -> dict[str, Any]:
    """Run and grade one isolated virtual Pi mission."""
    if frame_count < 120:
        raise ValueError("frame_count must be at least 120 for mixed-rate confirmation")
    if frame_delay_s <= 0:
        raise ValueError("frame_delay_s must be positive")
    if output_dir.exists() and output_dir.is_symlink():
        raise ValueError("output directory must not be a symbolic link")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"output directory is not empty: {output_dir}")
    output_dir.parent.mkdir(parents=True, exist_ok=True)

    cv2 = VirtualCV2(frame_count=frame_count, frame_delay_s=frame_delay_s)
    metrics = run_agent(
        _agent_args(output_dir),
        cv2_module=cv2,
        runner_factory=lambda spec, _device: VirtualRunner(spec),
        health_sampler=_virtual_health,
        nmea_factory=_nmea_factory,
    )
    device_status = json.loads((output_dir / "device_status.json").read_text(encoding="utf-8"))
    delivery, fleet = _deliver_to_virtual_centre(output_dir)
    field_gate = evaluate_field_run(
        metrics,
        device_status,
        minimum_runtime_s=1.0,
        minimum_captured_frames=120,
    )
    write_json_atomic(output_dir / "field_validation.json", field_gate)

    failed_field_checks = {
        check["key"] for check in field_gate["checks"] if not check["passed"]
    }
    application_checks = {
        "capture_completed": metrics["capture"]["captured_frames"] == frame_count,
        "mixed_rate_models_executed": all(
            values["attempted"] > 0
            for values in metrics["scheduler"]["schedules"].values()
        ),
        "nmea_fix_received": bool((metrics.get("gps") or {}).get("has_fix")),
        "temporal_evidence_created": metrics["confirmed_events"] >= 1,
        "outbox_delivered": delivery["delivered"] == metrics["confirmed_events"],
        "central_store_received_all": fleet["event_count"] == metrics["confirmed_events"],
        "idempotency_replay_safe": bool(fleet["duplicate_replay_statuses"])
        and all(status == 200 for status in fleet["duplicate_replay_statuses"]),
        "physical_claim_blocked": field_gate["field_verified"] is False
        and "physical_execution_environment" in failed_field_checks,
        "no_runtime_errors": metrics["model_errors"] == 0 and not metrics["capture_error"],
    }
    report = {
        "schema_version": 1,
        "test_mode": "application_level_raspberry_pi_emulation",
        "arm_instruction_emulation": False,
        "physical_hardware_tested": False,
        "host_machine": platform.machine() or "unknown",
        "virtual_profile": "Raspberry Pi 4 / 4 GB / CPU-only / mixed-rate",
        "passed": all(application_checks.values()),
        "application_checks": application_checks,
        "edge_metrics": metrics,
        "delivery": delivery,
        "fleet": fleet,
        "physical_field_gate": field_gate,
        "limitations": [
            "No ARM instruction-set emulation was available in this environment.",
            (
                "Camera drivers, GPIO, serial voltage levels, thermals and power stability "
                "were not tested."
            ),
            (
                "Synthetic model runners validate orchestration and measured scheduling, "
                "not model accuracy."
            ),
            "A physical Raspberry Pi mission is still required for field verification.",
        ],
    }
    write_json_atomic(output_dir / "virtual_pi_report.json", report)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="artifacts/virtual_pi/latest")
    parser.add_argument("--frames", type=int, default=240)
    parser.add_argument("--frame-delay-ms", type=float, default=20.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        report = run_virtual_pi_test(
            Path(args.output_dir),
            frame_count=args.frames,
            frame_delay_s=args.frame_delay_ms / 1000.0,
        )
    except (OSError, RuntimeError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc
    print(json.dumps(report, indent=2))
    print(f"Virtual Pi report written to: {Path(args.output_dir) / 'virtual_pi_report.json'}")
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
