"""Urban-asset inspection and inventory-aware missing-infrastructure logic.

Missing infrastructure is an absence claim: a detector cannot learn a useful
``missing_zebra_crossing`` box.  DrishtiPath therefore compares temporally stable visual
observations with a geospatial inventory.  Every absence result remains a review candidate
and is only emitted when the inventory explicitly says the asset should be camera-visible.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from urban_intelligence.gps import GPSPoint, haversine_m

ASSET_TYPES = frozenset({"divider", "zebra_crossing", "traffic_signboard"})
DIRECT_HAZARD_CLASSES = frozenset(
    {
        "waterlogging",
        "damaged_divider",
        "damaged_zebra_crossing",
        "damaged_traffic_signboard",
    }
)
PRESENT_CLASS_BY_ASSET = {
    "divider": "road_divider",
    "zebra_crossing": "zebra_crossing",
    "traffic_signboard": "traffic_signboard",
}
DAMAGED_CLASS_BY_ASSET = {
    "divider": "damaged_divider",
    "zebra_crossing": "damaged_zebra_crossing",
    "traffic_signboard": "damaged_traffic_signboard",
}
ASSET_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")

_CLASS_ALIASES = {
    "divider": "road_divider",
    "median": "road_divider",
    "road_median": "road_divider",
    "zebra": "zebra_crossing",
    "crosswalk": "zebra_crossing",
    "pedestrian_crossing": "zebra_crossing",
    "traffic_sign": "traffic_signboard",
    "road_sign": "traffic_signboard",
    "signboard": "traffic_signboard",
    "flooded_road": "waterlogging",
    "standing_water": "waterlogging",
    "water_log": "waterlogging",
}

_ASSET_TYPE_ALIASES = {
    "divider": "divider",
    "road_divider": "divider",
    "median": "divider",
    "road_median": "divider",
    "zebra": "zebra_crossing",
    "crosswalk": "zebra_crossing",
    "pedestrian_crossing": "zebra_crossing",
    "zebra_crossing": "zebra_crossing",
    "traffic_sign": "traffic_signboard",
    "road_sign": "traffic_signboard",
    "signboard": "traffic_signboard",
    "traffic_signboard": "traffic_signboard",
}


def normalize_asset_class(value: str) -> str:
    normalized = re.sub(r"[^a-z0-9]+", "_", value.strip().lower()).strip("_")
    return _CLASS_ALIASES.get(normalized, normalized)


def normalize_asset_type(value: str) -> str:
    normalized = re.sub(r"[^a-z0-9]+", "_", value.strip().lower()).strip("_")
    return _ASSET_TYPE_ALIASES.get(normalized, normalized)


@dataclass(frozen=True, slots=True)
class AssetObservation:
    frame_index: int
    video_time_s: float
    class_name: str
    confidence: float
    latitude: float
    longitude: float
    bbox: tuple[float, float, float, float] | None = None

    def __post_init__(self) -> None:
        if self.frame_index < 0 or self.video_time_s < 0:
            raise ValueError("asset observation frame and time must be non-negative")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("asset observation confidence must be between 0 and 1")
        if not -90 <= self.latitude <= 90 or not -180 <= self.longitude <= 180:
            raise ValueError("asset observation coordinates are invalid")

    @property
    def normalized_class(self) -> str:
        return normalize_asset_class(self.class_name)


@dataclass(frozen=True, slots=True)
class ExpectedAsset:
    asset_id: str
    asset_type: str
    latitude: float
    longitude: float
    inspection_radius_m: float = 25.0
    camera_visible: bool = False
    min_sampled_frames: int = 5
    min_visual_hits: int = 2

    def __post_init__(self) -> None:
        if not ASSET_ID_PATTERN.fullmatch(self.asset_id):
            raise ValueError("asset_id must be a safe identifier")
        if self.asset_type not in ASSET_TYPES:
            raise ValueError(f"asset_type must be one of: {', '.join(sorted(ASSET_TYPES))}")
        if not -90 <= self.latitude <= 90 or not -180 <= self.longitude <= 180:
            raise ValueError("expected asset coordinates are invalid")
        if not math.isfinite(self.inspection_radius_m) or self.inspection_radius_m <= 0:
            raise ValueError("inspection_radius_m must be positive")
        if self.min_sampled_frames < 1 or self.min_visual_hits < 1:
            raise ValueError("inspection sample thresholds must be positive")


@dataclass(frozen=True, slots=True)
class AssetInspectionEvent:
    event_id: str
    asset_id: str
    event_type: str
    class_name: str
    first_frame: int
    last_frame: int
    start_time_s: float
    end_time_s: float
    sampled_frames: int
    visual_hits: int
    latitude: float
    longitude: float
    evidence_strength: float
    status: str = "pending_review"
    requires_human_review: bool = True
    method: str = "inventory_visual_confirmation"

    def as_dict(self) -> dict[str, Any]:
        return {
            "event_id": self.event_id,
            "asset_id": self.asset_id,
            "event_type": self.event_type,
            "class": self.class_name,
            "first_frame": self.first_frame,
            "last_frame": self.last_frame,
            "start_time_s": round(self.start_time_s, 3),
            "end_time_s": round(self.end_time_s, 3),
            "sampled_frames": self.sampled_frames,
            "visual_hits": self.visual_hits,
            "lat": round(self.latitude, 7),
            "lon": round(self.longitude, 7),
            "evidence_strength": round(self.evidence_strength, 4),
            "status": self.status,
            "requires_human_review": self.requires_human_review,
            "method": self.method,
        }


@dataclass(slots=True)
class _AssetVisit:
    visit_number: int
    first_frame: int
    last_frame: int
    start_time_s: float
    end_time_s: float
    sampled_frames: int = 0
    present_hits: int = 0
    damaged_hits: int = 0


@dataclass(slots=True)
class AssetInventoryInspector:
    inventory: list[ExpectedAsset]
    _active: dict[str, _AssetVisit] = field(default_factory=dict)
    _visit_counts: dict[str, int] = field(default_factory=dict)
    events: list[AssetInspectionEvent] = field(default_factory=list)

    def __post_init__(self) -> None:
        ids = [item.asset_id for item in self.inventory]
        if len(ids) != len(set(ids)):
            raise ValueError("asset inventory contains duplicate asset_id values")

    def observe(
        self,
        *,
        frame_index: int,
        video_time_s: float,
        location: GPSPoint,
        observations: list[AssetObservation],
    ) -> list[AssetInspectionEvent]:
        emitted: list[AssetInspectionEvent] = []
        normalized = [item.normalized_class for item in observations]
        for asset in self.inventory:
            inside = (
                haversine_m(
                    location.latitude,
                    location.longitude,
                    asset.latitude,
                    asset.longitude,
                )
                <= asset.inspection_radius_m
            )
            visit = self._active.get(asset.asset_id)
            if not inside:
                if visit is not None:
                    event = self._close_visit(asset, visit)
                    if event is not None:
                        emitted.append(event)
                    self._active.pop(asset.asset_id, None)
                continue

            if visit is None:
                number = self._visit_counts.get(asset.asset_id, 0) + 1
                self._visit_counts[asset.asset_id] = number
                visit = _AssetVisit(
                    visit_number=number,
                    first_frame=frame_index,
                    last_frame=frame_index,
                    start_time_s=video_time_s,
                    end_time_s=video_time_s,
                )
                self._active[asset.asset_id] = visit

            visit.sampled_frames += 1
            visit.last_frame = frame_index
            visit.end_time_s = video_time_s
            present_class = PRESENT_CLASS_BY_ASSET[asset.asset_type]
            damaged_class = DAMAGED_CLASS_BY_ASSET[asset.asset_type]
            if present_class in normalized:
                visit.present_hits += 1
            if damaged_class in normalized:
                visit.damaged_hits += 1
        return emitted

    def finalize(self) -> list[AssetInspectionEvent]:
        emitted: list[AssetInspectionEvent] = []
        by_id = {item.asset_id: item for item in self.inventory}
        for asset_id, visit in list(self._active.items()):
            event = self._close_visit(by_id[asset_id], visit)
            if event is not None:
                emitted.append(event)
        self._active.clear()
        return emitted

    def _close_visit(
        self,
        asset: ExpectedAsset,
        visit: _AssetVisit,
    ) -> AssetInspectionEvent | None:
        if not asset.camera_visible or visit.sampled_frames < asset.min_sampled_frames:
            return None
        if visit.damaged_hits >= asset.min_visual_hits:
            event_type = "damaged_asset"
            class_name = DAMAGED_CLASS_BY_ASSET[asset.asset_type]
            visual_hits = visit.damaged_hits
            evidence_strength = min(1.0, visit.damaged_hits / visit.sampled_frames)
        elif visit.present_hits >= asset.min_visual_hits:
            return None
        else:
            event_type = "missing_asset_candidate"
            class_name = f"missing_{asset.asset_type}"
            visual_hits = visit.present_hits
            evidence_strength = min(
                1.0,
                max(0.0, (visit.sampled_frames - visual_hits) / visit.sampled_frames),
            )
        event = AssetInspectionEvent(
            event_id=f"asset-{asset.asset_id}-visit-{visit.visit_number:03d}",
            asset_id=asset.asset_id,
            event_type=event_type,
            class_name=class_name,
            first_frame=visit.first_frame,
            last_frame=visit.last_frame,
            start_time_s=visit.start_time_s,
            end_time_s=visit.end_time_s,
            sampled_frames=visit.sampled_frames,
            visual_hits=visual_hits,
            latitude=asset.latitude,
            longitude=asset.longitude,
            evidence_strength=evidence_strength,
            method=(
                "inventory_absence_heuristic"
                if event_type == "missing_asset_candidate"
                else "inventory_damage_confirmation"
            ),
        )
        self.events.append(event)
        return event


def load_asset_inventory(path: str | Path) -> list[ExpectedAsset]:
    inventory_path = Path(path)
    value = json.loads(inventory_path.read_text(encoding="utf-8"))
    if not isinstance(value, list):
        raise ValueError("asset inventory JSON must contain a list")
    inventory: list[ExpectedAsset] = []
    for item in value:
        if not isinstance(item, dict):
            raise ValueError("each asset inventory entry must be an object")
        inventory.append(
            ExpectedAsset(
                asset_id=str(item["asset_id"]),
                asset_type=normalize_asset_type(str(item["asset_type"])),
                latitude=float(item["latitude"]),
                longitude=float(item["longitude"]),
                inspection_radius_m=float(item.get("inspection_radius_m", 25.0)),
                camera_visible=bool(item.get("camera_visible", False)),
                min_sampled_frames=int(item.get("min_sampled_frames", 5)),
                min_visual_hits=int(item.get("min_visual_hits", 2)),
            )
        )
    return inventory
