"""Live dashcam edge agent with mixed-rate inference and durable evidence output."""

from __future__ import annotations

import argparse
import json
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from urban_intelligence.classes import normalize_class_name
from urban_intelligence.edge_runtime import (
    SAFE_EVENT_ID,
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
    schedule_dict,
    write_json_atomic,
)
from urban_intelligence.gps import GPSPoint, GPSTrack, load_gps_csv
from urban_intelligence.models import Detection
from urban_intelligence.nmea import SerialNMEAGPS
from urban_intelligence.temporal import TemporalEventFilter
from urban_intelligence.traffic import PERSON_CLASS, VEHICLE_CLASSES

PROFILE_RATES: dict[str, dict[str, float]] = {
    # Conservative starting points only. The generated report must be used to tune
    # rates on each physical device; these are not benchmark claims.
    "pi4": {"traffic": 3.0, "road_damage": 2.0, "urban_assets": 0.5},
    "pi5": {"traffic": 5.0, "road_damage": 4.0, "urban_assets": 1.0},
    "desktop": {"traffic": 8.0, "road_damage": 8.0, "urban_assets": 2.0},
}

TASK_PRIORITIES = {
    "traffic": 100,
    "road_damage": 90,
    "urban_assets": 40,
}

TRAFFIC_CLASSES = set(VEHICLE_CLASSES) | {PERSON_CLASS}


@dataclass(frozen=True, slots=True)
class RawDetection:
    class_name: str
    confidence: float
    bbox: tuple[int, int, int, int]
    track_id: int | None


@dataclass(frozen=True, slots=True)
class RunnerSpec:
    name: str
    model_path: str
    confidence: float
    image_size: int
    tracking: bool
    allowed_classes: frozenset[str] = frozenset()


class UltralyticsRunner:
    """Thin adapter that keeps one exported model resident for the whole mission."""

    def __init__(self, spec: RunnerSpec, *, device: str) -> None:
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise RuntimeError("Install edge dependencies from requirements.txt") from exc
        self.spec = spec
        self.model = YOLO(spec.model_path)
        self.device = device

    def infer(self, frame: Any) -> list[RawDetection]:
        arguments = {
            "conf": self.spec.confidence,
            "imgsz": self.spec.image_size,
            "device": self.device,
            "verbose": False,
        }
        if self.spec.tracking:
            results = self.model.track(
                frame,
                persist=True,
                tracker="bytetrack.yaml",
                **arguments,
            )
        else:
            results = self.model.predict(frame, **arguments)
        result = results[0]
        boxes = result.boxes
        if boxes is None or len(boxes) == 0:
            return []

        coordinates = boxes.xyxy.cpu().tolist()
        confidences = boxes.conf.cpu().tolist()
        classes = boxes.cls.cpu().tolist()
        track_ids = (
            boxes.id.cpu().tolist() if boxes.id is not None else [None] * len(coordinates)
        )
        detections: list[RawDetection] = []
        for coords, confidence, class_id, track_id in zip(
            coordinates,
            confidences,
            classes,
            track_ids,
            strict=True,
        ):
            class_name = normalize_class_name(str(self.model.names[int(class_id)]))
            if self.spec.allowed_classes and class_name not in self.spec.allowed_classes:
                continue
            detections.append(
                RawDetection(
                    class_name=class_name,
                    confidence=float(confidence),
                    bbox=tuple(int(value) for value in coords),  # type: ignore[arg-type]
                    track_id=None if track_id is None else int(track_id),
                )
            )
        return detections


class CaptureWorker(threading.Thread):
    """Continuously capture frames while inference consumes only the newest one."""

    def __init__(
        self,
        *,
        capture: Any,
        frame_buffer: LatestFrameBuffer,
        source_fps: float,
        pace_file: bool,
    ) -> None:
        super().__init__(name="drishtipath-capture", daemon=True)
        self.capture = capture
        self.frame_buffer = frame_buffer
        self.source_fps = max(source_fps, 1.0)
        self.pace_file = pace_file
        self.done = threading.Event()
        self.stop_requested = threading.Event()
        self.error = ""

    def run(self) -> None:
        started_at = time.monotonic()
        frame_index = 0
        try:
            while not self.stop_requested.is_set():
                if self.pace_file:
                    target_time = started_at + (frame_index / self.source_fps)
                    delay = target_time - time.monotonic()
                    if delay > 0:
                        time.sleep(min(delay, 0.1))
                        continue
                ok, frame = self.capture.read()
                if not ok:
                    break
                self.frame_buffer.push(
                    FrameEnvelope(
                        frame_index=frame_index,
                        captured_at_s=time.monotonic(),
                        payload=frame,
                    )
                )
                frame_index += 1
        except Exception as exc:  # capture backends raise implementation-specific errors
            self.error = f"{type(exc).__name__}: {exc}"
        finally:
            self.capture.release()
            self.done.set()

    def stop(self) -> None:
        self.stop_requested.set()


def parse_source(value: str) -> int | str:
    source = value.strip()
    if not source:
        raise ValueError("source must not be empty")
    if source.isdigit():
        return int(source)
    return source


def parse_fixed_gps(value: str) -> GPSPoint:
    try:
        latitude_raw, longitude_raw = value.split(",", 1)
        latitude = float(latitude_raw)
        longitude = float(longitude_raw)
    except (TypeError, ValueError) as exc:
        raise ValueError("fixed GPS must be LAT,LON") from exc
    if not -90 <= latitude <= 90 or not -180 <= longitude <= 180:
        raise ValueError("fixed GPS coordinates are invalid")
    return GPSPoint(0.0, latitude, longitude)


def load_geofences(path: Path | None) -> list[GeoFence]:
    if path is None:
        return []
    if not path.is_file():
        raise ValueError(f"geofence file not found: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, list):
        raise ValueError("geofence JSON must contain a list")
    geofences: list[GeoFence] = []
    for item in value:
        if not isinstance(item, dict):
            raise ValueError("each geofence must be an object")
        geofences.append(
            GeoFence(
                key=str(item["key"]),
                latitude=float(item["latitude"]),
                longitude=float(item["longitude"]),
                radius_m=float(item["radius_m"]),
            )
        )
    return geofences


def model_schedules(profile: str, configured_tasks: set[str]) -> list[ModelSchedule]:
    if profile not in PROFILE_RATES:
        raise ValueError(f"profile must be one of: {', '.join(PROFILE_RATES)}")
    unknown = configured_tasks - set(TASK_PRIORITIES)
    if unknown:
        raise ValueError(f"unknown configured tasks: {', '.join(sorted(unknown))}")
    return [
        ModelSchedule(
            name=name,
            target_fps=PROFILE_RATES[profile][name],
            priority=TASK_PRIORITIES[name],
        )
        for name in ("traffic", "road_damage", "urban_assets")
        if name in configured_tasks
    ]


def prepare_output_dir(path: Path) -> None:
    if path.exists() and path.is_symlink():
        raise ValueError("output directory must not be a symbolic link")
    if path.exists() and any(path.iterdir()):
        raise FileExistsError(f"output directory is not empty: {path}")
    path.mkdir(parents=True, exist_ok=True)


def validate_gps_configuration(
    *,
    gps_csv: str | None,
    fixed_gps: str | None,
    gps_source_type: str,
    gps_nmea_device: str | None = None,
) -> None:
    configured = [bool(gps_csv), bool(fixed_gps), bool(gps_nmea_device)]
    if sum(configured) > 1:
        raise ValueError("use only one of --gps-csv, --fixed-gps, or --gps-nmea-device")
    if not any(configured):
        raise ValueError("provide --gps-csv, --fixed-gps, or --gps-nmea-device")
    if fixed_gps and gps_source_type == "real_telemetry":
        raise ValueError("a fixed demo coordinate cannot be labelled real_telemetry")
    if gps_nmea_device and gps_source_type != "real_telemetry":
        raise ValueError("a live NMEA device must be labelled real_telemetry")


def validate_mission_identifiers(vehicle_id: str, mission_id: str, route_id: str) -> None:
    for name, value in (
        ("vehicle_id", vehicle_id),
        ("mission_id", mission_id),
        ("route_id", route_id),
    ):
        if not SAFE_EVENT_ID.fullmatch(value):
            raise ValueError(f"{name} must be a safe non-empty identifier")


def _gps_at(
    *,
    elapsed_s: float,
    gps_track: GPSTrack | None,
    fixed_gps: GPSPoint | None,
    nmea_gps: SerialNMEAGPS | None = None,
    nmea_max_age_s: float = 3.0,
) -> GPSPoint:
    if gps_track is not None:
        return gps_track.at(elapsed_s)
    if fixed_gps is not None:
        return GPSPoint(elapsed_s, fixed_gps.latitude, fixed_gps.longitude)
    if nmea_gps is not None:
        fix = nmea_gps.latest(max_age_s=nmea_max_age_s)
        if fix is None:
            raise RuntimeError("Live NMEA GPS fix is unavailable or stale")
        return GPSPoint(elapsed_s, fix.point.latitude, fix.point.longitude)
    raise RuntimeError("GPS provider is not configured")


def _detection_record(
    *,
    task_name: str,
    frame: FrameEnvelope,
    elapsed_s: float,
    location: GPSPoint,
    detection: RawDetection,
) -> dict[str, Any]:
    return {
        "task": task_name,
        "frame_index": frame.frame_index,
        "video_time_s": round(elapsed_s, 3),
        "class": detection.class_name,
        "confidence": round(detection.confidence, 4),
        "track_id": detection.track_id,
        "bbox": list(detection.bbox),
        "lat": round(location.latitude, 7),
        "lon": round(location.longitude, 7),
    }


def _save_confirmation(
    *,
    cv2: Any,
    task_name: str,
    frame: FrameEnvelope,
    confirmation: Any,
    output_dir: Path,
    outbox: EvidenceOutbox,
    gps_source_type: str,
    vehicle_id: str,
    mission_id: str,
    route_id: str,
) -> dict[str, Any]:
    detection = confirmation.detection
    class_key = "".join(
        character if character.isalnum() else "-"
        for character in confirmation.class_name
    ).strip("-")
    track_key = "untracked" if detection.track_id is None else str(detection.track_id)
    event_id = (
        f"{task_name}-{frame.frame_index:08d}-{class_key[:32]}-{track_key}-"
        f"{confirmation.hit_count}h"
    )
    evidence_dir = output_dir / "evidence"
    evidence_dir.mkdir(exist_ok=True)
    context_path = evidence_dir / f"{event_id}-frame.jpg"
    crop_path = evidence_dir / f"{event_id}-crop.jpg"
    x1, y1, x2, y2 = detection.bbox
    height, width = frame.payload.shape[:2]
    crop = frame.payload[max(0, y1) : min(height, y2), max(0, x1) : min(width, x2)]
    context_written = bool(cv2.imwrite(str(context_path), frame.payload))
    crop_written = bool(crop.size and cv2.imwrite(str(crop_path), crop))
    event = {
        "event_id": event_id,
        "event_type": "confirmed_edge_detection",
        "module": task_name,
        "class": confirmation.class_name,
        "confidence": round(confirmation.average_confidence, 4),
        "frame_index": frame.frame_index,
        "video_time_s": round(detection.video_time_s, 3),
        "lat": round(detection.latitude, 7),
        "lon": round(detection.longitude, 7),
        "temporal_hits": confirmation.hit_count,
        "gps_source_type": gps_source_type,
        "vehicle_id": vehicle_id,
        "mission_id": mission_id,
        "route_id": route_id,
        "status": "pending_review",
        "requires_human_review": True,
        "evidence_frame": str(context_path) if context_written else "",
        "evidence_crop": str(crop_path) if crop_written else "",
    }
    outbox.enqueue(event_id, event)
    return event


def build_runner_specs(args: argparse.Namespace) -> list[RunnerSpec]:
    specs = [
        RunnerSpec(
            name="traffic",
            model_path=args.traffic_model,
            confidence=args.traffic_confidence,
            image_size=args.image_size,
            tracking=True,
            allowed_classes=frozenset(TRAFFIC_CLASSES),
        ),
        RunnerSpec(
            name="road_damage",
            model_path=args.road_model,
            confidence=args.road_confidence,
            image_size=args.image_size,
            tracking=True,
        ),
    ]
    if args.asset_model:
        specs.append(
            RunnerSpec(
                name="urban_assets",
                model_path=args.asset_model,
                confidence=args.asset_confidence,
                image_size=args.image_size,
                tracking=True,
            )
        )
    return specs


def run_agent(
    args: argparse.Namespace,
    *,
    cv2_module: Any | None = None,
    runner_factory: Callable[[RunnerSpec, str], Any] | None = None,
    health_sampler: Callable[[], dict[str, Any]] = sample_device_health,
) -> dict[str, Any]:
    if cv2_module is None:
        try:
            import cv2 as cv2_module
        except ImportError as exc:
            raise RuntimeError("Install edge dependencies from requirements.txt") from exc
    cv2 = cv2_module
    create_runner = runner_factory or (lambda spec, device: UltralyticsRunner(spec, device=device))

    output_dir = Path(args.output_dir)
    prepare_output_dir(output_dir)
    source = parse_source(args.source)
    validate_gps_configuration(
        gps_csv=args.gps_csv,
        fixed_gps=args.fixed_gps,
        gps_source_type=args.gps_source_type,
        gps_nmea_device=getattr(args, "gps_nmea_device", None),
    )
    vehicle_id = getattr(args, "vehicle_id", "bus-demo-01")
    mission_id = getattr(args, "mission_id", "mission-demo")
    route_id = getattr(args, "route_id", "route-unassigned")
    validate_mission_identifiers(vehicle_id, mission_id, route_id)
    capture = cv2.VideoCapture(source)
    if not capture.isOpened():
        raise RuntimeError(f"could not open dashcam source: {args.source}")
    source_fps = float(capture.get(cv2.CAP_PROP_FPS) or 30.0)

    nmea_gps: SerialNMEAGPS | None = None
    try:
        gps_track = load_gps_csv(args.gps_csv) if args.gps_csv else None
        fixed_gps = parse_fixed_gps(args.fixed_gps) if args.fixed_gps else None
        if getattr(args, "gps_nmea_device", None):
            nmea_gps = SerialNMEAGPS(
                args.gps_nmea_device,
                baudrate=getattr(args, "gps_baud", 9600),
            )
            nmea_gps.start()
            nmea_gps.wait_for_fix(
                timeout_s=getattr(args, "gps_fix_timeout_s", 15.0)
            )
        geofences = load_geofences(Path(args.geofences) if args.geofences else None)

        specs = build_runner_specs(args)
        runners = {
            spec.name: create_runner(spec, args.device)
            for spec in specs
        }
    except BaseException:
        capture.release()
        if nmea_gps is not None:
            nmea_gps.close()
        raise
    schedules = model_schedules(args.profile, set(runners))
    scheduler = InferenceScheduler(schedules)
    temporal_filters = {
        name: TemporalEventFilter(window_size=5, min_hits=3)
        for name in ("road_damage", "urban_assets")
        if name in runners
    }
    frame_buffer = LatestFrameBuffer(capacity=args.buffer_frames)
    outbox = EvidenceOutbox(output_dir / "outbox")
    capture_worker = CaptureWorker(
        capture=capture,
        frame_buffer=frame_buffer,
        source_fps=source_fps,
        pace_file=isinstance(source, str) and Path(source).is_file(),
    )

    detections_path = output_dir / "detections.ndjson"
    events_path = output_dir / "events.ndjson"
    started_at = time.monotonic()
    last_health_sample = -float("inf")
    confirmed_events = 0
    model_errors = 0
    capture_worker.start()
    try:
        with detections_path.open("w", encoding="utf-8") as detections_handle, events_path.open(
            "w", encoding="utf-8"
        ) as events_handle:
            while True:
                now = time.monotonic()
                if args.duration_s > 0 and now - started_at >= args.duration_s:
                    break
                frame = frame_buffer.pop_latest()
                if frame is None:
                    if capture_worker.done.is_set():
                        break
                    time.sleep(0.005)
                    continue

                elapsed_s = max(0.0, frame.captured_at_s - started_at)
                location = _gps_at(
                    elapsed_s=elapsed_s,
                    gps_track=gps_track,
                    fixed_gps=fixed_gps,
                    nmea_gps=nmea_gps,
                    nmea_max_age_s=getattr(args, "gps_max_age_s", 3.0),
                )
                contexts = active_geofence_keys(location, geofences)
                task = scheduler.acquire_next(now_s=frame.captured_at_s, active_contexts=contexts)
                if task is None:
                    continue

                infer_started = time.perf_counter()
                try:
                    raw_detections = runners[task.name].infer(frame.payload)
                except Exception as exc:
                    latency_ms = (time.perf_counter() - infer_started) * 1000.0
                    scheduler.complete(
                        task.name,
                        latency_ms=latency_ms,
                        success=False,
                        error=f"{type(exc).__name__}: {exc}",
                    )
                    model_errors += 1
                    continue
                latency_ms = (time.perf_counter() - infer_started) * 1000.0
                scheduler.complete(task.name, latency_ms=latency_ms)

                typed_detections: list[Detection] = []
                for detection in raw_detections:
                    record = _detection_record(
                        task_name=task.name,
                        frame=frame,
                        elapsed_s=elapsed_s,
                        location=location,
                        detection=detection,
                    )
                    detections_handle.write(json.dumps(record) + "\n")
                    typed_detections.append(
                        Detection(
                            frame_index=frame.frame_index,
                            video_time_s=elapsed_s,
                            class_name=detection.class_name,
                            confidence=detection.confidence,
                            bbox=detection.bbox,
                            latitude=location.latitude,
                            longitude=location.longitude,
                            track_id=detection.track_id,
                        )
                    )
                detections_handle.flush()

                temporal_filter = temporal_filters.get(task.name)
                if temporal_filter is not None:
                    for confirmation in temporal_filter.update(typed_detections):
                        event = _save_confirmation(
                            cv2=cv2,
                            task_name=task.name,
                            frame=frame,
                            confirmation=confirmation,
                            output_dir=output_dir,
                            outbox=outbox,
                            gps_source_type=args.gps_source_type,
                            vehicle_id=vehicle_id,
                            mission_id=mission_id,
                            route_id=route_id,
                        )
                        events_handle.write(json.dumps(event) + "\n")
                        events_handle.flush()
                        confirmed_events += 1

                if now - last_health_sample >= args.health_interval_s:
                    health = health_sampler()
                    health["capture"] = frame_buffer.snapshot()
                    health["scheduler"] = scheduler.snapshot(now_s=now)
                    health["active_geofences"] = sorted(contexts)
                    if nmea_gps is not None:
                        health["gps"] = nmea_gps.snapshot()
                    write_json_atomic(output_dir / "device_status.json", health)
                    last_health_sample = now
    finally:
        capture_worker.stop()
        capture_worker.join(timeout=3.0)
        if nmea_gps is not None:
            nmea_gps.close()

    finished_at = time.monotonic()
    device_health = health_sampler()
    report = build_runtime_report(
        started_at_s=started_at,
        finished_at_s=finished_at,
        frame_buffer=frame_buffer,
        scheduler=scheduler,
        device_health=device_health,
    )
    measured_latencies = {
        name: float(values["mean_latency_ms"])
        for name, values in report["scheduler"]["schedules"].items()
    }
    report.update(
        {
            "source": args.source,
            "source_fps": round(source_fps, 3),
            "gps_source_type": args.gps_source_type,
            "vehicle_id": vehicle_id,
            "mission_id": mission_id,
            "route_id": route_id,
            "gps_provider": (
                "serial_nmea"
                if nmea_gps is not None
                else "timestamped_csv"
                if gps_track is not None
                else "fixed_demo_coordinate"
            ),
            "gps": None if nmea_gps is None else nmea_gps.snapshot(),
            "profile": args.profile,
            "configured_schedules": [schedule_dict(item) for item in schedules],
            "confirmed_events": confirmed_events,
            "pending_evidence_packets": len(outbox.pending()),
            "model_errors": model_errors,
            "capture_error": capture_worker.error,
            "compute_budget": estimate_compute_utilization(schedules, measured_latencies),
            "claim_policy": (
                "Profile rates are configuration targets. Device performance is represented "
                "only by measured runtime fields in this report."
            ),
        }
    )
    write_json_atomic(output_dir / "metrics.json", report)
    write_json_atomic(
        output_dir / "device_status.json",
        {
            **device_health,
            "capture": frame_buffer.snapshot(),
            "scheduler": scheduler.snapshot(now_s=finished_at),
            "mission_status": "completed",
        },
    )
    return report


def default_output_dir() -> str:
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    return f"artifacts/edge_live/run-{timestamp}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default="0", help="Camera index, video path, or RTSP URL")
    parser.add_argument("--vehicle-id", default="bus-demo-01")
    parser.add_argument("--mission-id", default="mission-demo")
    parser.add_argument("--route-id", default="route-unassigned")
    parser.add_argument("--gps-csv", help="Timestamped GPS CSV for route replay or telemetry")
    parser.add_argument("--fixed-gps", help="Fixed demo coordinate as LAT,LON")
    parser.add_argument("--gps-nmea-device", help="Live serial NMEA source, e.g. /dev/ttyUSB0")
    parser.add_argument("--gps-baud", type=int, default=9600)
    parser.add_argument("--gps-fix-timeout-s", type=float, default=15.0)
    parser.add_argument("--gps-max-age-s", type=float, default=3.0)
    parser.add_argument(
        "--gps-source-type",
        choices=("real_telemetry", "synthetic_demo", "unknown"),
        required=True,
    )
    parser.add_argument("--geofences", help="Optional geofence JSON")
    parser.add_argument("--profile", choices=tuple(PROFILE_RATES), default="pi4")
    parser.add_argument("--device", default="cpu", help="Ultralytics inference device")
    parser.add_argument("--road-model", default="models/road_hazards.pt")
    parser.add_argument("--traffic-model", default="yolov8n.pt")
    parser.add_argument("--asset-model", default="")
    parser.add_argument("--road-confidence", type=float, default=0.20)
    parser.add_argument("--traffic-confidence", type=float, default=0.25)
    parser.add_argument("--asset-confidence", type=float, default=0.25)
    parser.add_argument("--image-size", type=int, default=416)
    parser.add_argument("--buffer-frames", type=int, default=3)
    parser.add_argument("--duration-s", type=float, default=0.0)
    parser.add_argument("--health-interval-s", type=float, default=1.0)
    parser.add_argument("--output-dir", default=default_output_dir())
    args = parser.parse_args()
    try:
        validate_gps_configuration(
            gps_csv=args.gps_csv,
            fixed_gps=args.fixed_gps,
            gps_source_type=args.gps_source_type,
            gps_nmea_device=args.gps_nmea_device,
        )
        validate_mission_identifiers(args.vehicle_id, args.mission_id, args.route_id)
    except ValueError as exc:
        parser.error(str(exc))
    if args.image_size < 160:
        parser.error("--image-size must be at least 160")
    if args.duration_s < 0:
        parser.error("--duration-s must not be negative")
    if args.health_interval_s <= 0:
        parser.error("--health-interval-s must be positive")
    if args.gps_baud <= 0 or args.gps_fix_timeout_s <= 0 or args.gps_max_age_s <= 0:
        parser.error("NMEA GPS baud and timing values must be positive")
    return args


def main() -> None:
    args = parse_args()
    try:
        report = run_agent(args)
    except (FileExistsError, RuntimeError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc
    print(json.dumps(report, indent=2))
    print(f"Live edge artifacts written to: {args.output_dir}")


if __name__ == "__main__":
    main()
