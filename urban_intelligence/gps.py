"""Timestamp-based video/GPS alignment and spatial helpers."""

from __future__ import annotations

import csv
import math
from bisect import bisect_right
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class GPSPoint:
    timestamp_s: float
    latitude: float
    longitude: float


class GPSTrack:
    """A sorted GPS trajectory supporting linear interpolation."""

    def __init__(self, points: list[GPSPoint]):
        if not points:
            raise ValueError("GPS track must contain at least one point")
        self.points = sorted(points, key=lambda point: point.timestamp_s)
        self._timestamps = [point.timestamp_s for point in self.points]

    def at(self, timestamp_s: float) -> GPSPoint:
        """Interpolate the location at a video-relative timestamp."""
        if timestamp_s <= self.points[0].timestamp_s:
            point = self.points[0]
            return GPSPoint(timestamp_s, point.latitude, point.longitude)
        if timestamp_s >= self.points[-1].timestamp_s:
            point = self.points[-1]
            return GPSPoint(timestamp_s, point.latitude, point.longitude)

        right_index = bisect_right(self._timestamps, timestamp_s)
        left = self.points[right_index - 1]
        right = self.points[right_index]
        span = right.timestamp_s - left.timestamp_s
        ratio = 0.0 if span == 0 else (timestamp_s - left.timestamp_s) / span
        return GPSPoint(
            timestamp_s=timestamp_s,
            latitude=left.latitude + ((right.latitude - left.latitude) * ratio),
            longitude=left.longitude + ((right.longitude - left.longitude) * ratio),
        )

    def for_frame(self, frame_index: int, fps: float) -> GPSPoint:
        if fps <= 0:
            raise ValueError("FPS must be greater than zero")
        return self.at(frame_index / fps)


def load_gps_csv(path: str | Path) -> GPSTrack:
    """Load timestamp/latitude/longitude data from a CSV file."""
    csv_path = Path(path)
    with csv_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        columns = set(reader.fieldnames or [])
        latitude_column = "lat" if "lat" in columns else "latitude"
        longitude_column = "lon" if "lon" in columns else "longitude"
        if latitude_column not in columns or longitude_column not in columns:
            raise ValueError("GPS CSV requires lat/lon or latitude/longitude columns")

        rows = list(reader)
        if not rows:
            raise ValueError("GPS CSV contains no data rows")

    points: list[GPSPoint] = []
    for index, row in enumerate(rows):
        raw_timestamp = row.get("timestamp", row.get("timestamp_s", index))
        points.append(
            GPSPoint(
                timestamp_s=float(raw_timestamp),
                latitude=float(row[latitude_column]),
                longitude=float(row[longitude_column]),
            )
        )
    return GPSTrack(points)


def haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Return great-circle distance between two WGS84 coordinates in metres."""
    earth_radius_m = 6_371_000.0
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)
    a = (
        math.sin(delta_phi / 2) ** 2
        + math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda / 2) ** 2
    )
    return 2 * earth_radius_m * math.asin(math.sqrt(a))
