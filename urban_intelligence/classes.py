"""Friendly class-name normalization for road-hazard and COCO labels."""

from __future__ import annotations

# Official RDD2022 codes and friendly aliases (case-insensitive keys).
ROAD_HAZARD_ALIASES: dict[str, str] = {
    "d00": "longitudinal_crack",
    "d10": "transverse_crack",
    "d20": "alligator_crack",
    "d40": "pothole",
    "longitudinal_crack": "longitudinal_crack",
    "transverse_crack": "transverse_crack",
    "alligator_crack": "alligator_crack",
    "pothole": "pothole",
}


def normalize_class_name(class_name: str) -> str:
    """Return a friendly hazard name; leave unknown classes unchanged."""
    key = class_name.strip().lower().replace(" ", "_").replace("-", "_")
    return ROAD_HAZARD_ALIASES.get(key, class_name)


def class_matches_filter(class_name: str, allowed_classes: set[str]) -> bool:
    """Check whether a detection class passes a normalized filter set."""
    if not allowed_classes:
        return True
    normalized = normalize_class_name(class_name).lower()
    return normalized in allowed_classes or class_name.strip().lower() in allowed_classes
