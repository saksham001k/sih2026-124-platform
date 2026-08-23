"""Rule-based vulnerable-road-user and vehicle-behaviour candidates.

The rules consume ByteTrack detections already produced by the traffic model.  They do not
claim legal speed, intent, collision certainty, or a person's age.  Every event is a
reviewable candidate with its method and limitations attached.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from typing import Any

from urban_intelligence.traffic import (
    PERSON_CLASS,
    VEHICLE_CLASSES,
    NormalizedROI,
    bbox_area,
    bbox_center,
    center_in_roi,
)


@dataclass(frozen=True, slots=True)
class TrackedRoadUser:
    frame_index: int
    video_time_s: float
    track_id: int
    class_name: str
    confidence: float
    bbox: tuple[float, float, float, float]
    latitude: float
    longitude: float

    def __post_init__(self) -> None:
        if self.frame_index < 0 or self.video_time_s < 0 or self.track_id < 0:
            raise ValueError("tracked road-user frame, time, and track_id must be non-negative")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("tracked road-user confidence must be between 0 and 1")


@dataclass(frozen=True, slots=True)
class SafetyEvent:
    event_id: str
    event_type: str
    subtype: str
    video_time_s: float
    frame_index: int
    latitude: float
    longitude: float
    severity: str
    vehicle_track_id: int | None
    person_track_id: int | None
    evidence_score: float
    school_zone_context: bool = False
    child_identity_inferred: bool = False
    status: str = "pending_review"
    requires_human_review: bool = True
    method: str = "image_space_trajectory_heuristic"

    def as_dict(self) -> dict[str, Any]:
        return {
            "event_id": self.event_id,
            "event_type": self.event_type,
            "subtype": self.subtype,
            "video_time_s": round(self.video_time_s, 3),
            "frame_index": self.frame_index,
            "lat": round(self.latitude, 7),
            "lon": round(self.longitude, 7),
            "severity": self.severity,
            "vehicle_track_id": self.vehicle_track_id,
            "person_track_id": self.person_track_id,
            "evidence_score": round(self.evidence_score, 4),
            "school_zone_context": self.school_zone_context,
            "child_identity_inferred": self.child_identity_inferred,
            "status": self.status,
            "requires_human_review": self.requires_human_review,
            "method": self.method,
        }


@dataclass(frozen=True, slots=True)
class _CollisionCandidate:
    vehicle_track_id: int
    person_track_id: int
    frame_index: int
    video_time_s: float
    latitude: float
    longitude: float
    evidence_score: float


def bbox_iou(
    left: tuple[float, float, float, float],
    right: tuple[float, float, float, float],
) -> float:
    lx1, ly1, lx2, ly2 = left
    rx1, ry1, rx2, ry2 = right
    ix1, iy1 = max(lx1, rx1), max(ly1, ry1)
    ix2, iy2 = min(lx2, rx2), min(ly2, ry2)
    intersection = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    union = bbox_area(left) + bbox_area(right) - intersection
    return intersection / union if union > 0 else 0.0


def normalized_center_distance(
    left: tuple[float, float, float, float],
    right: tuple[float, float, float, float],
    *,
    frame_width: int,
    frame_height: int,
) -> float:
    lx, ly = bbox_center(left)
    rx, ry = bbox_center(right)
    diagonal = math.hypot(frame_width, frame_height)
    return math.hypot(lx - rx, ly - ry) / diagonal if diagonal else float("inf")


@dataclass(slots=True)
class RoadSafetyAnalyzer:
    """Generate explainable safety candidates from tracked traffic detections."""

    crossing_roi: NormalizedROI
    conflict_distance: float = 0.16
    collision_distance: float = 0.065
    lateral_speed_threshold: float = 0.45
    rapid_approach_threshold: float = 0.90
    hit_and_run_gap_s: float = 1.5
    history_size: int = 8
    _histories: dict[tuple[str, int], deque[TrackedRoadUser]] = field(default_factory=dict)
    _emitted_keys: set[str] = field(default_factory=set)
    _collision_candidates: dict[tuple[int, int], _CollisionCandidate] = field(
        default_factory=dict
    )
    events: list[SafetyEvent] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not 0 < self.collision_distance <= self.conflict_distance <= 1:
            raise ValueError("safety distance thresholds are invalid")
        if self.lateral_speed_threshold <= 0 or self.rapid_approach_threshold <= 0:
            raise ValueError("motion thresholds must be positive")
        if self.hit_and_run_gap_s <= 0 or self.history_size < 2:
            raise ValueError("hit-and-run gap and history size must be positive")

    def observe(
        self,
        detections: list[TrackedRoadUser],
        *,
        frame_width: int,
        frame_height: int,
        school_zone_active: bool = False,
    ) -> list[SafetyEvent]:
        if frame_width <= 0 or frame_height <= 0:
            raise ValueError("frame dimensions must be positive")
        if not detections:
            return []

        emitted: list[SafetyEvent] = []
        for detection in detections:
            key = (detection.class_name, detection.track_id)
            history = self._histories.setdefault(key, deque(maxlen=self.history_size))
            if history and detection.video_time_s < history[-1].video_time_s:
                raise ValueError("tracked detections must be observed in time order")
            history.append(detection)

        people = [item for item in detections if item.class_name == PERSON_CLASS]
        vehicles = [item for item in detections if item.class_name in VEHICLE_CLASSES]
        for vehicle in vehicles:
            emitted.extend(
                self._motion_candidates(
                    vehicle,
                    frame_width=frame_width,
                )
            )

        for person in people:
            if not center_in_roi(person.bbox, self.crossing_roi, frame_width, frame_height):
                continue
            for vehicle in vehicles:
                current_distance = normalized_center_distance(
                    person.bbox,
                    vehicle.bbox,
                    frame_width=frame_width,
                    frame_height=frame_height,
                )
                overlap = bbox_iou(person.bbox, vehicle.bbox)
                closing = self._distance_is_closing(
                    person,
                    vehicle,
                    current_distance=current_distance,
                    frame_width=frame_width,
                    frame_height=frame_height,
                )
                if current_distance <= self.conflict_distance and (closing or overlap > 0):
                    key = f"conflict:{person.track_id}:{vehicle.track_id}"
                    if key not in self._emitted_keys:
                        self._emitted_keys.add(key)
                        event = SafetyEvent(
                            event_id=(
                                f"safety-conflict-p{person.track_id}-v{vehicle.track_id}"
                            ),
                            event_type="vulnerable_pedestrian_conflict",
                            subtype=(
                                "school_zone_crossing_conflict"
                                if school_zone_active
                                else "pedestrian_crossing_conflict"
                            ),
                            video_time_s=person.video_time_s,
                            frame_index=person.frame_index,
                            latitude=person.latitude,
                            longitude=person.longitude,
                            severity="high" if school_zone_active else "medium",
                            vehicle_track_id=vehicle.track_id,
                            person_track_id=person.track_id,
                            evidence_score=min(
                                1.0,
                                max(overlap, 1.0 - current_distance / self.conflict_distance),
                            ),
                            school_zone_context=school_zone_active,
                        )
                        self.events.append(event)
                        emitted.append(event)

                if current_distance <= self.collision_distance or overlap >= 0.01:
                    pair = (vehicle.track_id, person.track_id)
                    score = min(
                        1.0,
                        max(overlap, 1.0 - current_distance / self.collision_distance),
                    )
                    previous = self._collision_candidates.get(pair)
                    if previous is None or score > previous.evidence_score:
                        self._collision_candidates[pair] = _CollisionCandidate(
                            vehicle_track_id=vehicle.track_id,
                            person_track_id=person.track_id,
                            frame_index=person.frame_index,
                            video_time_s=person.video_time_s,
                            latitude=person.latitude,
                            longitude=person.longitude,
                            evidence_score=score,
                        )

        now_s = max(item.video_time_s for item in detections)
        current_vehicle_ids = {item.track_id for item in vehicles}
        current_person_ids = {item.track_id for item in people}
        emitted.extend(
            self._departed_vehicle_candidates(
                now_s=now_s,
                current_vehicle_ids=current_vehicle_ids,
                current_person_ids=current_person_ids,
            )
        )
        return emitted

    def flush(
        self,
        *,
        now_s: float,
        visible_person_ids: set[int] | None = None,
    ) -> list[SafetyEvent]:
        return self._departed_vehicle_candidates(
            now_s=now_s,
            current_vehicle_ids=set(),
            current_person_ids=visible_person_ids or set(),
        )

    def _motion_candidates(
        self,
        vehicle: TrackedRoadUser,
        *,
        frame_width: int,
    ) -> list[SafetyEvent]:
        history = self._histories[(vehicle.class_name, vehicle.track_id)]
        if len(history) < 2:
            return []
        previous = history[-2]
        delta_s = vehicle.video_time_s - previous.video_time_s
        if delta_s <= 0:
            return []
        emitted: list[SafetyEvent] = []
        previous_x, _ = bbox_center(previous.bbox)
        current_x, _ = bbox_center(vehicle.bbox)
        lateral_rate = abs(current_x - previous_x) / max(frame_width, 1) / delta_s
        previous_area = bbox_area(previous.bbox)
        current_area = bbox_area(vehicle.bbox)
        area_growth = (
            math.log(current_area / previous_area) / delta_s
            if previous_area > 0 and current_area > 0
            else 0.0
        )
        candidates = [
            ("abrupt_lateral_motion", lateral_rate, self.lateral_speed_threshold),
            ("rapid_camera_approach", area_growth, self.rapid_approach_threshold),
        ]
        for subtype, measurement, threshold in candidates:
            if measurement < threshold:
                continue
            key = f"motion:{vehicle.track_id}:{subtype}"
            if key in self._emitted_keys:
                continue
            self._emitted_keys.add(key)
            score = min(1.0, measurement / (threshold * 2.0))
            event = SafetyEvent(
                event_id=f"safety-motion-v{vehicle.track_id}-{subtype}",
                event_type="rash_driving_candidate",
                subtype=subtype,
                video_time_s=vehicle.video_time_s,
                frame_index=vehicle.frame_index,
                latitude=vehicle.latitude,
                longitude=vehicle.longitude,
                severity="medium",
                vehicle_track_id=vehicle.track_id,
                person_track_id=None,
                evidence_score=score,
            )
            self.events.append(event)
            emitted.append(event)
        return emitted

    def _distance_is_closing(
        self,
        person: TrackedRoadUser,
        vehicle: TrackedRoadUser,
        *,
        current_distance: float,
        frame_width: int,
        frame_height: int,
    ) -> bool:
        person_history = self._histories[(person.class_name, person.track_id)]
        vehicle_history = self._histories[(vehicle.class_name, vehicle.track_id)]
        if len(person_history) < 2 or len(vehicle_history) < 2:
            return False
        previous_distance = normalized_center_distance(
            person_history[-2].bbox,
            vehicle_history[-2].bbox,
            frame_width=frame_width,
            frame_height=frame_height,
        )
        return previous_distance - current_distance >= 0.005

    def _departed_vehicle_candidates(
        self,
        *,
        now_s: float,
        current_vehicle_ids: set[int],
        current_person_ids: set[int],
    ) -> list[SafetyEvent]:
        emitted: list[SafetyEvent] = []
        for pair, candidate in list(self._collision_candidates.items()):
            if now_s - candidate.video_time_s < self.hit_and_run_gap_s:
                continue
            vehicle_id, person_id = pair
            if vehicle_id in current_vehicle_ids or person_id not in current_person_ids:
                continue
            key = f"hitrun:{vehicle_id}:{person_id}"
            if key not in self._emitted_keys:
                self._emitted_keys.add(key)
                event = SafetyEvent(
                    event_id=f"safety-hitrun-v{vehicle_id}-p{person_id}",
                    event_type="suspected_hit_and_run",
                    subtype="vehicle_departed_after_person_proximity",
                    video_time_s=candidate.video_time_s,
                    frame_index=candidate.frame_index,
                    latitude=candidate.latitude,
                    longitude=candidate.longitude,
                    severity="critical",
                    vehicle_track_id=vehicle_id,
                    person_track_id=person_id,
                    evidence_score=candidate.evidence_score,
                    method="proximity_then_departure_heuristic",
                )
                self.events.append(event)
                emitted.append(event)
            self._collision_candidates.pop(pair, None)
        return emitted


SAFETY_LIMITATIONS = [
    "School-zone context does not identify a pedestrian as a child.",
    "Image-space motion is not legal speed in km/h.",
    "Rash-driving and hit-and-run outputs are review candidates, not legal conclusions.",
    "Camera calibration, pose estimation, and field evaluation are required for deployment.",
]
