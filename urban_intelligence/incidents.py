"""Privacy-safe correlation of safety candidates with reviewed ANPR evidence."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

from urban_intelligence.gps import haversine_m

LINKABLE_EVENT_TYPES = frozenset({"suspected_hit_and_run", "rash_driving_candidate"})
FORBIDDEN_PLATE_FIELDS = frozenset(
    {"plate", "plate_text", "normalized_plate", "full_plate", "ocr_text"}
)


def _number(value: object, default: float = float("nan")) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if math.isfinite(number) else default


def _time(record: Mapping[str, Any]) -> float:
    for key in ("video_time_s", "first_video_time_s", "best_observation_time_s"):
        value = _number(record.get(key))
        if math.isfinite(value):
            return value
    return float("nan")


def _coordinate(record: Mapping[str, Any]) -> tuple[float, float] | None:
    latitude = _number(record.get("lat", record.get("latitude")))
    longitude = _number(record.get("lon", record.get("longitude")))
    if not math.isfinite(latitude) or not math.isfinite(longitude):
        return None
    if not -90 <= latitude <= 90 or not -180 <= longitude <= 180:
        return None
    return latitude, longitude


def link_anpr_evidence(
    safety_events: Sequence[Mapping[str, Any]],
    anpr_events: Sequence[Mapping[str, Any]],
    *,
    max_time_delta_s: float = 8.0,
    max_distance_m: float = 75.0,
) -> list[dict[str, Any]]:
    """Attach the nearest masked plate event without copying complete OCR text."""
    if max_time_delta_s <= 0 or max_distance_m <= 0:
        raise ValueError("ANPR correlation thresholds must be positive")
    linked: list[dict[str, Any]] = []
    for safety in safety_events:
        event = {
            key: value
            for key, value in safety.items()
            if key not in FORBIDDEN_PLATE_FIELDS
        }
        event_type = str(event.get("event_type", ""))
        event["anpr_link_status"] = "not_applicable"
        if event_type not in LINKABLE_EVENT_TYPES:
            linked.append(event)
            continue

        event_time = _time(event)
        event_coordinate = _coordinate(event)
        candidates: list[tuple[float, float, str, Mapping[str, Any]]] = []
        for plate in anpr_events:
            plate_time = _time(plate)
            plate_coordinate = _coordinate(plate)
            if not math.isfinite(event_time) or not math.isfinite(plate_time):
                continue
            time_delta = abs(plate_time - event_time)
            if time_delta > max_time_delta_s:
                continue
            distance = 0.0
            if event_coordinate is not None and plate_coordinate is not None:
                distance = haversine_m(*event_coordinate, *plate_coordinate)
                if distance > max_distance_m:
                    continue
            candidates.append(
                (time_delta, distance, str(plate.get("event_id", "")), plate)
            )

        if not candidates:
            event["anpr_link_status"] = "no_candidate"
            linked.append(event)
            continue
        time_delta, distance, plate_event_id, plate = min(candidates)
        masked = str(plate.get("masked_plate", "")).strip()
        event.update(
            {
                "anpr_link_status": "candidate_pending_review",
                "anpr_event_id": plate_event_id,
                "masked_plate": masked or "Masked plate unavailable",
                "anpr_time_delta_s": round(time_delta, 3),
                "anpr_distance_m": round(distance, 2),
                "anpr_requires_human_review": True,
            }
        )
        linked.append(event)
    return linked
