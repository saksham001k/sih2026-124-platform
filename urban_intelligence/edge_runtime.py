"""Deterministic scheduling and device telemetry for the live edge runtime.

This module deliberately contains no OpenCV, model-runtime, or network imports.  The
Raspberry Pi entry point supplies those adapters while these pure-data components keep
load shedding, activation rules, evidence durability, and metrics independently tested.
"""

from __future__ import annotations

import json
import math
import os
import platform
import re
import threading
from collections import deque
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from urban_intelligence.gps import GPSPoint, haversine_m

ActivationMode = Literal["always", "geofenced", "triggered"]
ACTIVATION_MODES = {"always", "geofenced", "triggered"}
SAFE_EVENT_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")


@dataclass(frozen=True, slots=True)
class FrameEnvelope:
    """One captured frame and its monotonic acquisition time."""

    frame_index: int
    captured_at_s: float
    payload: Any = field(repr=False, compare=False)

    def __post_init__(self) -> None:
        if self.frame_index < 0:
            raise ValueError("frame_index must not be negative")
        if not math.isfinite(self.captured_at_s) or self.captured_at_s < 0:
            raise ValueError("captured_at_s must be a finite non-negative value")


class LatestFrameBuffer:
    """Small thread-safe buffer that favours fresh evidence over stale backlog."""

    def __init__(self, capacity: int = 3) -> None:
        if capacity < 1:
            raise ValueError("capacity must be at least 1")
        self.capacity = capacity
        self._frames: deque[FrameEnvelope] = deque()
        self._lock = threading.Lock()
        self._captured_frames = 0
        self._dropped_frames = 0

    def push(self, frame: FrameEnvelope) -> None:
        with self._lock:
            self._captured_frames += 1
            if len(self._frames) >= self.capacity:
                self._frames.popleft()
                self._dropped_frames += 1
            self._frames.append(frame)

    def pop_latest(self) -> FrameEnvelope | None:
        """Return the newest frame and account for stale queued frames as dropped."""
        with self._lock:
            if not self._frames:
                return None
            stale = len(self._frames) - 1
            self._dropped_frames += stale
            newest = self._frames[-1]
            self._frames.clear()
            return newest

    def snapshot(self) -> dict[str, int]:
        with self._lock:
            return {
                "capacity": self.capacity,
                "queue_depth": len(self._frames),
                "captured_frames": self._captured_frames,
                "dropped_analysis_frames": self._dropped_frames,
            }


@dataclass(frozen=True, slots=True)
class ModelSchedule:
    """Execution policy for one model-backed or rule-backed capability."""

    name: str
    target_fps: float
    priority: int
    activation: ActivationMode = "always"
    context_keys: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.name or not SAFE_EVENT_ID.fullmatch(self.name):
            raise ValueError("schedule name must be a safe non-empty identifier")
        if not math.isfinite(self.target_fps) or self.target_fps <= 0:
            raise ValueError("target_fps must be greater than zero")
        if self.priority < 0:
            raise ValueError("priority must not be negative")
        if self.activation not in ACTIVATION_MODES:
            raise ValueError(f"activation must be one of: {', '.join(sorted(ACTIVATION_MODES))}")
        if self.activation == "geofenced" and not self.context_keys:
            raise ValueError("geofenced schedules require at least one context key")

    @property
    def interval_s(self) -> float:
        return 1.0 / self.target_fps


@dataclass(slots=True)
class ModelRuntimeStats:
    attempted: int = 0
    succeeded: int = 0
    failed: int = 0
    total_latency_ms: float = 0.0
    max_latency_ms: float = 0.0
    last_latency_ms: float = 0.0
    last_error: str = ""

    def record(self, latency_ms: float, *, success: bool, error: str = "") -> None:
        self.attempted += 1
        self.total_latency_ms += latency_ms
        self.max_latency_ms = max(self.max_latency_ms, latency_ms)
        self.last_latency_ms = latency_ms
        if success:
            self.succeeded += 1
            self.last_error = ""
        else:
            self.failed += 1
            self.last_error = error

    def snapshot(self) -> dict[str, Any]:
        mean = self.total_latency_ms / self.attempted if self.attempted else 0.0
        return {
            "attempted": self.attempted,
            "succeeded": self.succeeded,
            "failed": self.failed,
            "mean_latency_ms": round(mean, 3),
            "max_latency_ms": round(self.max_latency_ms, 3),
            "last_latency_ms": round(self.last_latency_ms, 3),
            "last_error": self.last_error,
        }


class InferenceScheduler:
    """Single-worker, deadline-aware scheduler for mixed-rate edge workloads."""

    def __init__(self, schedules: Sequence[ModelSchedule]) -> None:
        if not schedules:
            raise ValueError("at least one model schedule is required")
        self._schedules: dict[str, ModelSchedule] = {}
        for schedule in schedules:
            if schedule.name in self._schedules:
                raise ValueError(f"duplicate schedule: {schedule.name}")
            self._schedules[schedule.name] = schedule
        self._last_started_s: dict[str, float] = {}
        self._running: set[str] = set()
        self._trigger_until_s: dict[str, float] = {}
        self._stats = {name: ModelRuntimeStats() for name in self._schedules}

    def trigger(self, name: str, *, now_s: float, duration_s: float) -> None:
        schedule = self._get(name)
        if schedule.activation != "triggered":
            raise ValueError(f"{name} is not a triggered schedule")
        if duration_s <= 0 or not math.isfinite(duration_s):
            raise ValueError("duration_s must be a finite positive value")
        self._trigger_until_s[name] = max(
            self._trigger_until_s.get(name, -math.inf),
            now_s + duration_s,
        )

    def clear_trigger(self, name: str) -> None:
        self._get(name)
        self._trigger_until_s.pop(name, None)

    def _get(self, name: str) -> ModelSchedule:
        try:
            return self._schedules[name]
        except KeyError as exc:
            raise KeyError(f"unknown schedule: {name}") from exc

    def _is_active(
        self,
        schedule: ModelSchedule,
        *,
        now_s: float,
        active_contexts: set[str],
    ) -> bool:
        if schedule.activation == "always":
            return True
        if schedule.activation == "triggered":
            return now_s <= self._trigger_until_s.get(schedule.name, -math.inf)
        return bool(set(schedule.context_keys) & active_contexts)

    def acquire_next(
        self,
        *,
        now_s: float,
        active_contexts: set[str] | None = None,
    ) -> ModelSchedule | None:
        """Reserve the most overdue active workload for the single inference worker."""
        contexts = active_contexts or set()
        candidates: list[tuple[int, float, int, str, ModelSchedule]] = []
        for schedule in self._schedules.values():
            if schedule.name in self._running:
                continue
            if not self._is_active(schedule, now_s=now_s, active_contexts=contexts):
                continue
            last_started = self._last_started_s.get(schedule.name)
            due_at = -math.inf if last_started is None else last_started + schedule.interval_s
            if now_s + 1e-9 < due_at:
                continue
            trigger_rank = 1 if schedule.activation == "triggered" else 0
            candidates.append(
                (-trigger_rank, due_at, -schedule.priority, schedule.name, schedule)
            )
        if not candidates:
            return None
        selected = min(candidates)[-1]
        self._running.add(selected.name)
        self._last_started_s[selected.name] = now_s
        return selected

    def complete(
        self,
        name: str,
        *,
        latency_ms: float,
        success: bool = True,
        error: str = "",
    ) -> None:
        self._get(name)
        if name not in self._running:
            raise RuntimeError(f"schedule was not acquired: {name}")
        if not math.isfinite(latency_ms) or latency_ms < 0:
            raise ValueError("latency_ms must be a finite non-negative value")
        self._running.remove(name)
        self._stats[name].record(latency_ms, success=success, error=error)

    def snapshot(self, *, now_s: float) -> dict[str, Any]:
        return {
            "schedules": {
                name: {
                    "target_fps": schedule.target_fps,
                    "priority": schedule.priority,
                    "activation": schedule.activation,
                    "context_keys": list(schedule.context_keys),
                    "trigger_active": now_s
                    <= self._trigger_until_s.get(name, -math.inf),
                    **self._stats[name].snapshot(),
                }
                for name, schedule in self._schedules.items()
            },
            "running": sorted(self._running),
        }


@dataclass(frozen=True, slots=True)
class GeoFence:
    """One context boundary that activates selected workloads."""

    key: str
    latitude: float
    longitude: float
    radius_m: float

    def __post_init__(self) -> None:
        if not self.key or not SAFE_EVENT_ID.fullmatch(self.key):
            raise ValueError("geofence key must be a safe identifier")
        if not -90 <= self.latitude <= 90 or not -180 <= self.longitude <= 180:
            raise ValueError("geofence coordinates are invalid")
        if self.radius_m <= 0 or not math.isfinite(self.radius_m):
            raise ValueError("geofence radius_m must be a finite positive value")

    def contains(self, point: GPSPoint) -> bool:
        return (
            haversine_m(
                self.latitude,
                self.longitude,
                point.latitude,
                point.longitude,
            )
            <= self.radius_m
        )


def active_geofence_keys(point: GPSPoint, geofences: Sequence[GeoFence]) -> set[str]:
    return {geofence.key for geofence in geofences if geofence.contains(point)}


def estimate_compute_utilization(
    schedules: Sequence[ModelSchedule],
    latency_ms_by_name: Mapping[str, float],
) -> dict[str, Any]:
    """Estimate single-worker demand from measured model latency and target rates."""
    contributions: dict[str, float] = {}
    for schedule in schedules:
        latency_ms = float(latency_ms_by_name.get(schedule.name, 0.0))
        if latency_ms < 0 or not math.isfinite(latency_ms):
            raise ValueError("latency measurements must be finite and non-negative")
        contributions[schedule.name] = schedule.target_fps * latency_ms / 1000.0
    utilization = sum(contributions.values())
    return {
        "estimated_utilization": round(utilization, 4),
        "headroom": round(1.0 - utilization, 4),
        "overloaded": utilization > 1.0,
        "contributions": {name: round(value, 4) for name, value in contributions.items()},
    }


class EvidenceOutbox:
    """Atomic disk-backed evidence queue for intermittent mobile connectivity."""

    def __init__(self, root: Path) -> None:
        if root.exists() and root.is_symlink():
            raise ValueError("outbox root must not be a symbolic link")
        self.root = root
        self.pending_dir = root / "pending"
        self.sent_dir = root / "sent"
        self.pending_dir.mkdir(parents=True, exist_ok=True)
        self.sent_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _validate_event_id(event_id: str) -> str:
        if not SAFE_EVENT_ID.fullmatch(event_id):
            raise ValueError("event_id contains unsafe characters")
        return event_id

    def enqueue(
        self,
        event_id: str,
        payload: Mapping[str, Any],
        *,
        queued_at: datetime | None = None,
    ) -> Path:
        safe_id = self._validate_event_id(event_id)
        path = self.pending_dir / f"{safe_id}.json"
        temporary = self.pending_dir / f".{safe_id}.json.tmp"
        envelope = {
            "schema_version": 1,
            "event_id": safe_id,
            "queued_at": (queued_at or datetime.now(UTC)).isoformat(),
            "attempts": 0,
            "payload": dict(payload),
        }
        temporary.write_text(json.dumps(envelope, indent=2), encoding="utf-8")
        temporary.replace(path)
        return path

    def pending(self) -> list[Path]:
        return sorted(
            path
            for path in self.pending_dir.glob("*.json")
            if path.is_file() and not path.is_symlink()
        )

    def record_failure(self, event_id: str, error: str) -> Path:
        safe_id = self._validate_event_id(event_id)
        path = self.pending_dir / f"{safe_id}.json"
        if not path.is_file():
            raise FileNotFoundError(path)
        envelope = json.loads(path.read_text(encoding="utf-8"))
        envelope["attempts"] = int(envelope.get("attempts", 0)) + 1
        envelope["last_error"] = str(error)[:500]
        envelope["last_attempt_at"] = datetime.now(UTC).isoformat()
        temporary = self.pending_dir / f".{safe_id}.json.tmp"
        temporary.write_text(json.dumps(envelope, indent=2), encoding="utf-8")
        temporary.replace(path)
        return path

    def acknowledge(self, event_id: str) -> Path:
        safe_id = self._validate_event_id(event_id)
        source = self.pending_dir / f"{safe_id}.json"
        if not source.is_file():
            raise FileNotFoundError(source)
        destination = self.sent_dir / source.name
        source.replace(destination)
        return destination


def _read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8").replace("\x00", "").strip()
    except (OSError, UnicodeError):
        return ""


def _memory_info(path: Path) -> tuple[float | None, float | None]:
    values: dict[str, float] = {}
    for line in _read_text(path).splitlines():
        if ":" not in line:
            continue
        key, raw = line.split(":", 1)
        token = raw.strip().split()[0] if raw.strip() else ""
        try:
            values[key] = float(token) / 1024.0
        except ValueError:
            continue
    return values.get("MemTotal"), values.get("MemAvailable")


def sample_device_health(
    *,
    device_tree_path: Path = Path("/proc/device-tree/model"),
    thermal_path: Path = Path("/sys/class/thermal/thermal_zone0/temp"),
    meminfo_path: Path = Path("/proc/meminfo"),
) -> dict[str, Any]:
    """Return privacy-safe hardware telemetry without hostname or user identifiers."""
    device_model = _read_text(device_tree_path) or None
    thermal_raw = _read_text(thermal_path)
    temperature_c: float | None = None
    try:
        temperature_value = float(thermal_raw)
        temperature_c = temperature_value / 1000.0 if temperature_value > 200 else temperature_value
    except ValueError:
        pass
    total_mb, available_mb = _memory_info(meminfo_path)
    try:
        load_1m = float(os.getloadavg()[0])
    except (AttributeError, OSError):
        load_1m = None
    return {
        "sampled_at": datetime.now(UTC).isoformat(),
        "device_model": device_model,
        "raspberry_pi": bool(device_model and "raspberry pi" in device_model.lower()),
        "platform": platform.system() or "unknown",
        "machine": platform.machine() or "unknown",
        "logical_cpu_count": os.cpu_count(),
        "load_1m": None if load_1m is None else round(load_1m, 3),
        "cpu_temperature_c": None if temperature_c is None else round(temperature_c, 2),
        "memory_total_mb": None if total_mb is None else round(total_mb, 1),
        "memory_available_mb": None if available_mb is None else round(available_mb, 1),
    }


def write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(dict(payload), indent=2), encoding="utf-8")
    temporary.replace(path)


def build_runtime_report(
    *,
    started_at_s: float,
    finished_at_s: float,
    frame_buffer: LatestFrameBuffer,
    scheduler: InferenceScheduler,
    device_health: Mapping[str, Any],
) -> dict[str, Any]:
    if finished_at_s < started_at_s:
        raise ValueError("finished_at_s must not precede started_at_s")
    duration_s = finished_at_s - started_at_s
    buffer_stats = frame_buffer.snapshot()
    analyzed = sum(
        item["attempted"]
        for item in scheduler.snapshot(now_s=finished_at_s)["schedules"].values()
    )
    return {
        "schema_version": 1,
        "runtime_seconds": round(duration_s, 3),
        "capture": buffer_stats,
        "analytics_attempts": analyzed,
        "analytics_attempts_per_second": round(analyzed / duration_s, 3)
        if duration_s
        else 0.0,
        "scheduler": scheduler.snapshot(now_s=finished_at_s),
        "device": dict(device_health),
    }


def schedule_dict(schedule: ModelSchedule) -> dict[str, Any]:
    """Stable serializable representation used by run manifests."""
    return asdict(schedule)
