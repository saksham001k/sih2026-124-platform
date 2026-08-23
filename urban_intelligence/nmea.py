"""NMEA RMC parsing and a small serial GPS adapter for Raspberry Pi deployments."""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from urban_intelligence.gps import GPSPoint

KNOT_TO_MPS = 0.514444


@dataclass(frozen=True, slots=True)
class NMEAFix:
    point: GPSPoint
    speed_mps: float
    captured_at: datetime | None
    talker: str


def nmea_checksum(payload: str) -> int:
    checksum = 0
    for character in payload:
        checksum ^= ord(character)
    return checksum


def _coordinate(value: str, hemisphere: str, *, latitude: bool) -> float:
    if not value:
        raise ValueError("NMEA coordinate is empty")
    degrees_width = 2 if latitude else 3
    if len(value) <= degrees_width:
        raise ValueError("NMEA coordinate is malformed")
    degrees = float(value[:degrees_width])
    minutes = float(value[degrees_width:])
    if not 0 <= minutes < 60:
        raise ValueError("NMEA coordinate minutes are invalid")
    coordinate = degrees + minutes / 60.0
    if hemisphere in {"S", "W"}:
        coordinate *= -1
    expected = {"N", "S"} if latitude else {"E", "W"}
    if hemisphere not in expected:
        raise ValueError("NMEA hemisphere is invalid")
    return coordinate


def _captured_at(time_value: str, date_value: str) -> datetime | None:
    if len(time_value) < 6 or len(date_value) != 6:
        return None
    try:
        base = datetime.strptime(
            f"{date_value}{time_value[:6]}",
            "%d%m%y%H%M%S",
        ).replace(tzinfo=UTC)
        fraction = float(f"0.{time_value.split('.', 1)[1]}") if "." in time_value else 0.0
    except ValueError:
        return None
    return base.replace(microsecond=int(fraction * 1_000_000))


def parse_rmc_sentence(sentence: str, *, timestamp_s: float) -> NMEAFix:
    """Parse a checksum-validated GPRMC/GNRMC sentence into a WGS84 fix."""
    text = sentence.strip()
    if not text.startswith("$") or "*" not in text:
        raise ValueError("NMEA sentence must include '$' and checksum")
    payload, checksum_text = text[1:].rsplit("*", 1)
    try:
        expected_checksum = int(checksum_text[:2], 16)
    except ValueError as exc:
        raise ValueError("NMEA checksum is malformed") from exc
    if nmea_checksum(payload) != expected_checksum:
        raise ValueError("NMEA checksum mismatch")
    fields = payload.split(",")
    if len(fields) < 10 or fields[0] not in {"GPRMC", "GNRMC", "GLRMC", "GARMC"}:
        raise ValueError("NMEA sentence is not a supported RMC fix")
    if fields[2] != "A":
        raise ValueError("NMEA RMC fix is not active")
    latitude = _coordinate(fields[3], fields[4], latitude=True)
    longitude = _coordinate(fields[5], fields[6], latitude=False)
    speed_knots = float(fields[7] or 0.0)
    return NMEAFix(
        point=GPSPoint(timestamp_s, latitude, longitude),
        speed_mps=max(0.0, speed_knots * KNOT_TO_MPS),
        captured_at=_captured_at(fields[1], fields[9]),
        talker=fields[0][:2],
    )


class SerialNMEAGPS:
    """Background reader retaining only the latest valid serial RMC fix."""

    def __init__(
        self,
        device: str,
        *,
        baudrate: int = 9600,
        serial_factory: Callable[..., Any] | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if not device.strip():
            raise ValueError("NMEA serial device must not be empty")
        if baudrate <= 0:
            raise ValueError("NMEA baudrate must be positive")
        self.device = device
        self.baudrate = baudrate
        self._serial_factory = serial_factory
        self._clock = clock
        self._started_at = 0.0
        self._latest: NMEAFix | None = None
        self._latest_monotonic = 0.0
        self._invalid_sentences = 0
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._serial: Any | None = None

    def start(self) -> None:
        if self._thread is not None:
            raise RuntimeError("NMEA reader is already started")
        factory = self._serial_factory
        if factory is None:
            try:
                import serial
            except ImportError as exc:
                raise RuntimeError("Install pyserial to use live NMEA GPS") from exc
            factory = serial.Serial
        self._serial = factory(self.device, self.baudrate, timeout=1.0)
        self._started_at = self._clock()
        self._thread = threading.Thread(
            target=self._run,
            name="drishtipath-nmea-gps",
            daemon=True,
        )
        self._thread.start()

    def _run(self) -> None:
        assert self._serial is not None
        while not self._stop.is_set():
            raw = self._serial.readline()
            if not raw:
                continue
            try:
                text = raw.decode("ascii", errors="strict") if isinstance(raw, bytes) else str(raw)
                now = self._clock()
                fix = parse_rmc_sentence(text, timestamp_s=max(0.0, now - self._started_at))
            except (UnicodeError, ValueError):
                with self._lock:
                    self._invalid_sentences += 1
                continue
            with self._lock:
                self._latest = fix
                self._latest_monotonic = now

    def wait_for_fix(self, *, timeout_s: float = 15.0) -> NMEAFix:
        if timeout_s <= 0:
            raise ValueError("GPS fix timeout must be positive")
        deadline = self._clock() + timeout_s
        while self._clock() < deadline:
            fix = self.latest(max_age_s=timeout_s)
            if fix is not None:
                return fix
            time.sleep(0.05)
        raise RuntimeError("No valid NMEA GPS fix received before timeout")

    def latest(self, *, max_age_s: float = 3.0) -> NMEAFix | None:
        if max_age_s <= 0:
            raise ValueError("GPS maximum age must be positive")
        with self._lock:
            if self._latest is None:
                return None
            if self._clock() - self._latest_monotonic > max_age_s:
                return None
            return self._latest

    def close(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        if self._serial is not None:
            self._serial.close()

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            latest = self._latest
            return {
                "device": self.device,
                "baudrate": self.baudrate,
                "has_fix": latest is not None,
                "invalid_sentences": self._invalid_sentences,
                "latest_timestamp_s": None if latest is None else latest.point.timestamp_s,
                "latest_speed_mps": None if latest is None else round(latest.speed_mps, 3),
            }
