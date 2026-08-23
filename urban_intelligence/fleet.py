"""Central fleet evidence store, geospatial deduplication, and OD aggregation."""

from __future__ import annotations

import csv
import hmac
import json
import math
import re
import sqlite3
from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from urban_intelligence.gps import haversine_m

IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
DEFICIENCY_CLASSES = frozenset(
    {
        "pothole",
        "longitudinal_crack",
        "transverse_crack",
        "alligator_crack",
        "waterlogging",
        "missing_divider",
        "missing_zebra_crossing",
        "missing_traffic_signboard",
        "damaged_divider",
        "damaged_zebra_crossing",
        "damaged_traffic_signboard",
    }
)


def _number(value: object, default: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if math.isfinite(number) else default


def _coordinate(payload: Mapping[str, Any]) -> tuple[float, float]:
    latitude = _number(payload.get("lat", payload.get("latitude")), float("nan"))
    longitude = _number(payload.get("lon", payload.get("longitude")), float("nan"))
    if not math.isfinite(latitude) or not math.isfinite(longitude):
        raise ValueError("fleet event requires latitude and longitude")
    if not -90 <= latitude <= 90 or not -180 <= longitude <= 180:
        raise ValueError("fleet event coordinates are invalid")
    return latitude, longitude


@dataclass(frozen=True, slots=True)
class IngestResult:
    event_id: str
    vehicle_id: str
    mission_id: str
    accepted: bool
    duplicate: bool


class FleetStore:
    """SQLite-backed fleet store with idempotent event ingestion."""

    def __init__(self, path: Path) -> None:
        if path.exists() and path.is_symlink():
            raise ValueError("fleet database must not be a symbolic link")
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA foreign_keys=ON")
        return connection

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS events (
                    vehicle_id TEXT NOT NULL,
                    event_id TEXT NOT NULL,
                    mission_id TEXT NOT NULL,
                    route_id TEXT NOT NULL,
                    received_at TEXT NOT NULL,
                    event_type TEXT NOT NULL,
                    class_name TEXT NOT NULL,
                    video_time_s REAL NOT NULL,
                    latitude REAL NOT NULL,
                    longitude REAL NOT NULL,
                    severity TEXT NOT NULL,
                    status TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    PRIMARY KEY (vehicle_id, event_id)
                );
                CREATE INDEX IF NOT EXISTS events_geo ON events(class_name, latitude, longitude);
                CREATE INDEX IF NOT EXISTS events_mission ON events(vehicle_id, mission_id);
                CREATE TABLE IF NOT EXISTS missions (
                    vehicle_id TEXT NOT NULL,
                    mission_id TEXT NOT NULL,
                    route_id TEXT NOT NULL,
                    gps_source_type TEXT NOT NULL,
                    start_time_s REAL NOT NULL,
                    end_time_s REAL NOT NULL,
                    start_latitude REAL NOT NULL,
                    start_longitude REAL NOT NULL,
                    end_latitude REAL NOT NULL,
                    end_longitude REAL NOT NULL,
                    event_count INTEGER NOT NULL,
                    PRIMARY KEY (vehicle_id, mission_id)
                );
                """
            )

    def ingest_envelope(self, envelope: Mapping[str, Any]) -> IngestResult:
        payload_value = envelope.get("payload")
        if not isinstance(payload_value, Mapping):
            raise ValueError("evidence envelope payload must be an object")
        payload = dict(payload_value)
        event_id = str(envelope.get("event_id", payload.get("event_id", ""))).strip()
        vehicle_id = str(payload.get("vehicle_id", "")).strip()
        mission_id = str(payload.get("mission_id", "")).strip()
        route_id = str(payload.get("route_id", "unassigned")).strip() or "unassigned"
        for name, value in (
            ("event_id", event_id),
            ("vehicle_id", vehicle_id),
            ("mission_id", mission_id),
            ("route_id", route_id),
        ):
            if not IDENTIFIER.fullmatch(value):
                raise ValueError(f"{name} must be a safe non-empty identifier")
        latitude, longitude = _coordinate(payload)
        video_time_s = _number(payload.get("video_time_s"))
        if video_time_s < 0:
            raise ValueError("event video_time_s must be non-negative")
        event_type = str(payload.get("event_type", "edge_event")).strip() or "edge_event"
        class_name = str(payload.get("class", payload.get("class_name", "unknown"))).strip()
        severity = str(payload.get("severity", "review")).strip() or "review"
        status = str(payload.get("status", "pending_review")).strip() or "pending_review"
        received_at = datetime.now(UTC).isoformat()
        payload_json = json.dumps(payload, sort_keys=True, separators=(",", ":"))

        with self._connect() as connection:
            cursor = connection.execute(
                """
                INSERT OR IGNORE INTO events (
                    vehicle_id, event_id, mission_id, route_id, received_at,
                    event_type, class_name, video_time_s, latitude, longitude,
                    severity, status, payload_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    vehicle_id,
                    event_id,
                    mission_id,
                    route_id,
                    received_at,
                    event_type,
                    class_name,
                    video_time_s,
                    latitude,
                    longitude,
                    severity,
                    status,
                    payload_json,
                ),
            )
            accepted = cursor.rowcount == 1
            if accepted:
                self._update_mission(
                    connection,
                    vehicle_id=vehicle_id,
                    mission_id=mission_id,
                    route_id=route_id,
                    gps_source_type=str(payload.get("gps_source_type", "unknown")),
                    video_time_s=video_time_s,
                    latitude=latitude,
                    longitude=longitude,
                )
        return IngestResult(
            event_id=event_id,
            vehicle_id=vehicle_id,
            mission_id=mission_id,
            accepted=accepted,
            duplicate=not accepted,
        )

    @staticmethod
    def _update_mission(
        connection: sqlite3.Connection,
        *,
        vehicle_id: str,
        mission_id: str,
        route_id: str,
        gps_source_type: str,
        video_time_s: float,
        latitude: float,
        longitude: float,
    ) -> None:
        row = connection.execute(
            "SELECT * FROM missions WHERE vehicle_id = ? AND mission_id = ?",
            (vehicle_id, mission_id),
        ).fetchone()
        if row is None:
            connection.execute(
                """
                INSERT INTO missions VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    vehicle_id,
                    mission_id,
                    route_id,
                    gps_source_type,
                    video_time_s,
                    video_time_s,
                    latitude,
                    longitude,
                    latitude,
                    longitude,
                    1,
                ),
            )
            return
        start_time = float(row["start_time_s"])
        end_time = float(row["end_time_s"])
        start_lat = float(row["start_latitude"])
        start_lon = float(row["start_longitude"])
        end_lat = float(row["end_latitude"])
        end_lon = float(row["end_longitude"])
        if video_time_s < start_time:
            start_time, start_lat, start_lon = video_time_s, latitude, longitude
        if video_time_s >= end_time:
            end_time, end_lat, end_lon = video_time_s, latitude, longitude
        connection.execute(
            """
            UPDATE missions SET route_id = ?, gps_source_type = ?,
                start_time_s = ?, end_time_s = ?,
                start_latitude = ?, start_longitude = ?,
                end_latitude = ?, end_longitude = ?, event_count = event_count + 1
            WHERE vehicle_id = ? AND mission_id = ?
            """,
            (
                route_id,
                gps_source_type,
                start_time,
                end_time,
                start_lat,
                start_lon,
                end_lat,
                end_lon,
                vehicle_id,
                mission_id,
            ),
        )

    def events(self) -> list[dict[str, Any]]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT * FROM events ORDER BY received_at, vehicle_id, event_id"
            ).fetchall()
        return [dict(row) for row in rows]

    def missions(self) -> list[dict[str, Any]]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT * FROM missions ORDER BY vehicle_id, mission_id"
            ).fetchall()
        return [dict(row) for row in rows]

    def od_matrix(self, *, precision: int = 3) -> list[dict[str, Any]]:
        if not 1 <= precision <= 6:
            raise ValueError("OD precision must be between 1 and 6")
        groups: dict[tuple[str, str, str], dict[str, Any]] = {}
        for mission in self.missions():
            origin = self._zone(
                mission["start_latitude"], mission["start_longitude"], precision
            )
            destination = self._zone(
                mission["end_latitude"], mission["end_longitude"], precision
            )
            key = (str(mission["route_id"]), origin, destination)
            group = groups.setdefault(
                key,
                {
                    "route_id": key[0],
                    "origin_zone": origin,
                    "destination_zone": destination,
                    "trip_count": 0,
                    "vehicles": set(),
                    "duration_total_s": 0.0,
                },
            )
            group["trip_count"] += 1
            group["vehicles"].add(mission["vehicle_id"])
            group["duration_total_s"] += max(
                0.0,
                float(mission["end_time_s"]) - float(mission["start_time_s"]),
            )
        result: list[dict[str, Any]] = []
        for key in sorted(groups):
            group = groups[key]
            trips = int(group["trip_count"])
            result.append(
                {
                    "route_id": group["route_id"],
                    "origin_zone": group["origin_zone"],
                    "destination_zone": group["destination_zone"],
                    "trip_count": trips,
                    "unique_vehicles": len(group["vehicles"]),
                    "mean_observed_duration_s": round(
                        group["duration_total_s"] / trips,
                        3,
                    ),
                }
            )
        return result

    @staticmethod
    def _zone(latitude: float, longitude: float, precision: int) -> str:
        return f"{float(latitude):.{precision}f},{float(longitude):.{precision}f}"

    def deficiency_clusters(self, *, radius_m: float = 20.0) -> list[dict[str, Any]]:
        if radius_m <= 0:
            raise ValueError("cluster radius must be positive")
        clusters: list[dict[str, Any]] = []
        for event in self.events():
            if event["class_name"] not in DEFICIENCY_CLASSES:
                continue
            match = None
            for cluster in clusters:
                if cluster["class"] != event["class_name"]:
                    continue
                if (
                    haversine_m(
                        cluster["latitude"],
                        cluster["longitude"],
                        event["latitude"],
                        event["longitude"],
                    )
                    <= radius_m
                ):
                    match = cluster
                    break
            if match is None:
                clusters.append(
                    {
                        "cluster_id": f"deficiency-{len(clusters) + 1:04d}",
                        "class": event["class_name"],
                        "latitude": event["latitude"],
                        "longitude": event["longitude"],
                        "sightings": 1,
                        "vehicles": {event["vehicle_id"]},
                        "event_ids": [event["event_id"]],
                    }
                )
                continue
            sightings = int(match["sightings"])
            match["latitude"] = (
                match["latitude"] * sightings + event["latitude"]
            ) / (sightings + 1)
            match["longitude"] = (
                match["longitude"] * sightings + event["longitude"]
            ) / (sightings + 1)
            match["sightings"] = sightings + 1
            match["vehicles"].add(event["vehicle_id"])
            match["event_ids"].append(event["event_id"])
        return [
            {
                **cluster,
                "latitude": round(cluster["latitude"], 7),
                "longitude": round(cluster["longitude"], 7),
                "vehicles": sorted(cluster["vehicles"]),
                "unique_vehicles": len(cluster["vehicles"]),
            }
            for cluster in clusters
        ]

    def summary(self) -> dict[str, Any]:
        events = self.events()
        missions = self.missions()
        return {
            "schema_version": 1,
            "generated_at": datetime.now(UTC).isoformat(),
            "vehicle_count": len({row["vehicle_id"] for row in missions}),
            "mission_count": len(missions),
            "event_count": len(events),
            "events_by_type": dict(sorted(Counter(row["event_type"] for row in events).items())),
            "events_by_class": dict(sorted(Counter(row["class_name"] for row in events).items())),
            "pending_review": sum(row["status"] == "pending_review" for row in events),
            "deficiency_cluster_count": len(self.deficiency_clusters()),
            "od_pair_count": len(self.od_matrix()),
        }

    def export(self, output_dir: Path) -> dict[str, Path]:
        output_dir.mkdir(parents=True, exist_ok=True)
        summary_path = output_dir / "fleet_summary.json"
        clusters_path = output_dir / "deficiency_clusters.json"
        geojson_path = output_dir / "fleet_events.geojson"
        od_path = output_dir / "od_matrix.csv"
        summary_path.write_text(json.dumps(self.summary(), indent=2), encoding="utf-8")
        clusters_path.write_text(
            json.dumps(self.deficiency_clusters(), indent=2),
            encoding="utf-8",
        )
        features = [
            {
                "type": "Feature",
                "geometry": {
                    "type": "Point",
                    "coordinates": [event["longitude"], event["latitude"]],
                },
                "properties": {
                    key: event[key]
                    for key in (
                        "vehicle_id",
                        "event_id",
                        "mission_id",
                        "route_id",
                        "event_type",
                        "class_name",
                        "severity",
                        "status",
                    )
                },
            }
            for event in self.events()
        ]
        geojson_path.write_text(
            json.dumps({"type": "FeatureCollection", "features": features}, indent=2),
            encoding="utf-8",
        )
        od_rows = self.od_matrix()
        with od_path.open("w", newline="", encoding="utf-8") as handle:
            columns = [
                "route_id",
                "origin_zone",
                "destination_zone",
                "trip_count",
                "unique_vehicles",
                "mean_observed_duration_s",
            ]
            writer = csv.DictWriter(handle, fieldnames=columns)
            writer.writeheader()
            writer.writerows(od_rows)
        return {
            "summary": summary_path,
            "clusters": clusters_path,
            "geojson": geojson_path,
            "od_matrix": od_path,
        }


class FleetIngestionService:
    """Pure HTTP-boundary logic used by the central server and tests."""

    def __init__(self, store: FleetStore, token: str) -> None:
        if not token:
            raise ValueError("fleet ingestion token must not be empty")
        self.store = store
        self.token = token

    def authorized(self, authorization: str) -> bool:
        return hmac.compare_digest(authorization, f"Bearer {self.token}")

    def handle(
        self,
        body: bytes,
        *,
        authorization: str,
        idempotency_key: str,
    ) -> tuple[int, dict[str, Any]]:
        if not self.authorized(authorization):
            return 401, {"error": "unauthorized"}
        if len(body) > 1024 * 1024:
            return 413, {"error": "payload_too_large"}
        try:
            envelope = json.loads(body.decode("utf-8"))
        except (UnicodeError, json.JSONDecodeError):
            return 400, {"error": "invalid_json"}
        if not isinstance(envelope, dict):
            return 400, {"error": "invalid_envelope"}
        if str(envelope.get("event_id", "")) != idempotency_key:
            return 409, {"error": "idempotency_key_mismatch"}
        try:
            result = self.store.ingest_envelope(envelope)
        except ValueError as exc:
            return 422, {"error": str(exc)}
        return (
            200 if result.duplicate else 202,
            {
                "event_id": result.event_id,
                "accepted": result.accepted,
                "duplicate": result.duplicate,
            },
        )
