"""Pure-data scene builder for the DrishtiPath operational command centre."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

from urban_intelligence.classes import normalize_class_name

DEFAULT_CENTER = {"latitude": 28.6139, "longitude": 77.2090, "zoom": 14.0}

HAZARD_COLORS = {
    "pothole": [255, 78, 89, 230],
    "longitudinal_crack": [255, 166, 43, 225],
    "transverse_crack": [255, 207, 64, 225],
    "alligator_crack": [225, 92, 255, 225],
}


def _number(value: object, default: float = 0.0) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if math.isfinite(result) else default


def _text(value: object, default: str = "") -> str:
    if value is None:
        return default
    result = str(value).strip()
    return result or default


def _boolean(value: object) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y"}
    return bool(value)


def _coordinate(record: Mapping[str, Any]) -> tuple[float, float] | None:
    latitude = _number(record.get("lat", record.get("latitude")), float("nan"))
    longitude = _number(record.get("lon", record.get("longitude")), float("nan"))
    if not math.isfinite(latitude) or not math.isfinite(longitude):
        return None
    if not -90 <= latitude <= 90 or not -180 <= longitude <= 180:
        return None
    return latitude, longitude


def _estimate_zoom(coordinates: Sequence[tuple[float, float]]) -> float:
    if len(coordinates) < 2:
        return 15.5
    latitudes = [item[0] for item in coordinates]
    longitudes = [item[1] for item in coordinates]
    span = max(max(latitudes) - min(latitudes), max(longitudes) - min(longitudes))
    if span <= 0.002:
        return 16.0
    if span <= 0.005:
        return 15.0
    if span <= 0.02:
        return 13.5
    if span <= 0.08:
        return 11.5
    return 9.5


def _scene_center(coordinates: Sequence[tuple[float, float]]) -> dict[str, float]:
    if not coordinates:
        return dict(DEFAULT_CENTER)
    latitudes = [item[0] for item in coordinates]
    longitudes = [item[1] for item in coordinates]
    return {
        "latitude": (min(latitudes) + max(latitudes)) / 2,
        "longitude": (min(longitudes) + max(longitudes)) / 2,
        "zoom": _estimate_zoom(coordinates),
    }


def _timed_route_points(records: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    timed_points: list[tuple[float, list[float]]] = []
    seen: set[tuple[float, float]] = set()
    for index, record in enumerate(records):
        coordinate = _coordinate(record)
        if coordinate is None:
            continue
        latitude, longitude = coordinate
        key = (round(latitude, 7), round(longitude, 7))
        if key in seen:
            continue
        seen.add(key)
        timestamp = _number(
            record.get("timestamp_s", record.get("timestamp", index)),
            float(index),
        )
        timed_points.append((timestamp, [longitude, latitude]))
    timed_points.sort(key=lambda item: item[0])
    return [
        {"time_s": timestamp, "position": position}
        for timestamp, position in timed_points
    ]


def _route_points(records: Sequence[Mapping[str, Any]]) -> list[list[float]]:
    return [item["position"] for item in _timed_route_points(records)]


def _hazard_points(records: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    points: list[dict[str, Any]] = []
    for index, record in enumerate(records):
        coordinate = _coordinate(record)
        if coordinate is None:
            continue
        latitude, longitude = coordinate
        class_name = normalize_class_name(_text(record.get("class"), "road_hazard"))
        confidence = min(1.0, max(0.0, _number(record.get("confidence"))))
        observations = max(1, int(_number(record.get("observation_count"), 1)))
        points.append(
            {
                "id": _text(record.get("event_id"), f"hazard-{index + 1}"),
                "kind": "Road hazard",
                "label": class_name.replace("_", " ").title(),
                "position": [longitude, latitude],
                "time_s": _number(record.get("video_time_s")),
                "confidence": confidence,
                "confidence_label": f"{confidence * 100:.1f}%",
                "observations": observations,
                "elevation": 35 + (confidence * 115) + min(observations, 5) * 8,
                "radius": 9,
                "color": HAZARD_COLORS.get(class_name, [255, 116, 64, 225]),
                "status": _text(record.get("status"), "pending_review"),
            }
        )
    return points


def _traffic_points(records: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    points: list[dict[str, Any]] = []
    for index, record in enumerate(records):
        coordinate = _coordinate(record)
        if coordinate is None:
            continue
        latitude, longitude = coordinate
        occupancy = min(1.0, max(0.0, _number(record.get("mean_occupancy"))))
        vehicles = max(0.0, _number(record.get("mean_vehicle_count")))
        congested = _boolean(record.get("congested", False))
        if congested:
            color = [255, 71, 87, 180]
        else:
            color = [26, 220, 198, 80 + int(occupancy * 130)]
        points.append(
            {
                "id": _text(record.get("window_index"), f"traffic-{index + 1}"),
                "kind": "Traffic window",
                "label": f"{vehicles:.1f} vehicles · {occupancy * 100:.1f}% occupancy",
                "position": [longitude, latitude],
                "time_s": _number(
                    record.get("midpoint_time_s", record.get("start_time_s"))
                ),
                "occupancy": occupancy,
                "vehicles": vehicles,
                "elevation": 15 + (occupancy * 260) + min(vehicles, 20) * 6,
                "radius": 18,
                "color": color,
                "status": "congested" if congested else "observed",
            }
        )
    return points


def _bottleneck_points(records: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    points: list[dict[str, Any]] = []
    for index, record in enumerate(records):
        coordinate = _coordinate(record)
        if coordinate is None:
            continue
        latitude, longitude = coordinate
        vehicles = max(0.0, _number(record.get("mean_vehicle_count")))
        occupancy = min(1.0, max(0.0, _number(record.get("mean_occupancy"))))
        points.append(
            {
                "id": _text(record.get("event_id"), f"bottleneck-{index + 1}"),
                "kind": "Bottleneck",
                "label": f"{vehicles:.1f} vehicles · {occupancy * 100:.1f}% occupancy",
                "position": [longitude, latitude],
                "time_s": _number(record.get("start_time_s")),
                "elevation": 220,
                "radius": 26,
                "color": [255, 49, 72, 235],
                "status": _text(record.get("status"), "pending_review"),
            }
        )
    return points


def _anpr_points(records: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Build privacy-safe points; full plate text is deliberately never copied."""
    points: list[dict[str, Any]] = []
    for index, record in enumerate(records):
        coordinate = _coordinate(record)
        if coordinate is None:
            continue
        latitude, longitude = coordinate
        masked_plate = _text(record.get("masked_plate"), "Masked plate")
        points.append(
            {
                "id": _text(record.get("event_id"), f"anpr-{index + 1}"),
                "kind": "ANPR evidence",
                "label": masked_plate,
                "position": [longitude, latitude],
                "time_s": _number(record.get("first_video_time_s")),
                "elevation": 105,
                "radius": 19,
                "color": [70, 155, 255, 235],
                "status": _text(record.get("status"), "pending_review"),
            }
        )
    return points


def _timeline(
    road_events: Sequence[Mapping[str, Any]],
    bottleneck_events: Sequence[Mapping[str, Any]],
    anpr_events: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index, record in enumerate(road_events):
        class_name = normalize_class_name(_text(record.get("class"), "road_hazard"))
        rows.append(
            {
                "id": _text(record.get("event_id"), f"hazard-{index + 1}"),
                "kind": "road",
                "time_s": _number(record.get("video_time_s")),
                "title": class_name.replace("_", " ").title(),
                "detail": f"Confidence {_number(record.get('confidence')) * 100:.1f}%",
                "status": _text(record.get("status"), "pending_review"),
                "evidence_frame": _text(record.get("evidence_frame")),
                "evidence_crop": _text(record.get("evidence_crop")),
            }
        )
    for index, record in enumerate(bottleneck_events):
        rows.append(
            {
                "id": _text(record.get("event_id"), f"bottleneck-{index + 1}"),
                "kind": "traffic",
                "time_s": _number(record.get("start_time_s")),
                "title": "Traffic bottleneck",
                "detail": f"{_number(record.get('mean_vehicle_count')):.1f} vehicles",
                "status": _text(record.get("status"), "pending_review"),
                "evidence_frame": "",
                "evidence_crop": "",
            }
        )
    for index, record in enumerate(anpr_events):
        rows.append(
            {
                "id": _text(record.get("event_id"), f"anpr-{index + 1}"),
                "kind": "anpr",
                "time_s": _number(record.get("first_video_time_s")),
                "title": "ANPR evidence",
                "detail": _text(record.get("masked_plate"), "Masked plate"),
                "status": _text(record.get("status"), "pending_review"),
                "evidence_frame": _text(record.get("evidence_frame")),
                "evidence_crop": _text(record.get("evidence_crop")),
            }
        )
    rows.sort(key=lambda item: (item["time_s"], item["kind"], item["id"]))
    return rows


def build_operational_scene(
    *,
    route_records: Sequence[Mapping[str, Any]] = (),
    road_events: Sequence[Mapping[str, Any]] = (),
    traffic_windows: Sequence[Mapping[str, Any]] = (),
    bottleneck_events: Sequence[Mapping[str, Any]] = (),
    anpr_events: Sequence[Mapping[str, Any]] = (),
) -> dict[str, Any]:
    """Normalize module outputs into one privacy-safe 3D operational scene."""
    route_timed = _timed_route_points(route_records)
    route = [item["position"] for item in route_timed]
    hazards = _hazard_points(road_events)
    traffic = _traffic_points(traffic_windows)
    bottlenecks = _bottleneck_points(bottleneck_events)
    anpr = _anpr_points(anpr_events)

    coordinates = [
        *[(position[1], position[0]) for position in route],
        *[
            (point["position"][1], point["position"][0])
            for point in [*hazards, *traffic, *bottlenecks, *anpr]
        ],
    ]

    return {
        "center": _scene_center(coordinates),
        "route": route,
        "route_timed": route_timed,
        "hazards": hazards,
        "traffic": traffic,
        "bottlenecks": bottlenecks,
        "anpr": anpr,
        "timeline": _timeline(road_events, bottleneck_events, anpr_events),
        "counts": {
            "hazards": len(hazards),
            "traffic_windows": len(traffic),
            "bottlenecks": len(bottlenecks),
            "anpr": len(anpr),
        },
    }


def filter_scene_at(scene: Mapping[str, Any], cutoff_s: float) -> dict[str, Any]:
    """Return an operational scene containing observations up to ``cutoff_s``."""
    if not math.isfinite(cutoff_s) or cutoff_s < 0:
        raise ValueError("scene replay cutoff must be finite and non-negative")
    route_timed = [
        dict(item)
        for item in scene.get("route_timed", [])
        if _number(item.get("time_s")) <= cutoff_s
    ]
    filtered: dict[str, Any] = {
        "route_timed": route_timed,
        "route": [item["position"] for item in route_timed],
    }
    for layer in ("hazards", "traffic", "bottlenecks", "anpr"):
        filtered[layer] = [
            dict(item)
            for item in scene.get(layer, [])
            if _number(item.get("time_s")) <= cutoff_s
        ]
    filtered["timeline"] = [
        dict(item)
        for item in scene.get("timeline", [])
        if _number(item.get("time_s")) <= cutoff_s
    ]
    coordinates = [
        *[(position[1], position[0]) for position in filtered["route"]],
        *[
            (point["position"][1], point["position"][0])
            for point in [
                *filtered["hazards"],
                *filtered["traffic"],
                *filtered["bottlenecks"],
                *filtered["anpr"],
            ]
        ],
    ]
    filtered["center"] = _scene_center(coordinates)
    filtered["counts"] = {
        "hazards": len(filtered["hazards"]),
        "traffic_windows": len(filtered["traffic"]),
        "bottlenecks": len(filtered["bottlenecks"]),
        "anpr": len(filtered["anpr"]),
    }
    return filtered
