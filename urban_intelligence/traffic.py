"""Pure traffic-analytics helpers for ROI density and bottleneck heuristics."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass, field, replace

VEHICLE_CLASSES = frozenset({"car", "motorcycle", "bus", "truck", "bicycle"})
PERSON_CLASS = "person"
GPS_SOURCE_TYPES = ("synthetic_demo", "real_telemetry", "unknown")
DEFAULT_ROI = (0.0, 0.30, 1.0, 1.0)


@dataclass(frozen=True, slots=True)
class NormalizedROI:
    """Axis-aligned ROI in normalized image coordinates (x1,y1,x2,y2)."""

    x1: float
    y1: float
    x2: float
    y2: float

    def pixel_bounds(self, frame_width: int, frame_height: int) -> tuple[int, int, int, int]:
        return (
            int(round(self.x1 * frame_width)),
            int(round(self.y1 * frame_height)),
            int(round(self.x2 * frame_width)),
            int(round(self.y2 * frame_height)),
        )

    def pixel_area(self, frame_width: int, frame_height: int) -> float:
        x1, y1, x2, y2 = self.pixel_bounds(frame_width, frame_height)
        return max(0, x2 - x1) * max(0, y2 - y1)


@dataclass(frozen=True, slots=True)
class TrafficDetection:
    """One model detection used by the traffic analytics pipeline."""

    frame_index: int
    video_time_s: float
    track_id: int | None
    class_name: str
    confidence: float
    bbox: tuple[float, float, float, float]


def bbox_center(bbox: tuple[float, float, float, float]) -> tuple[float, float]:
    x1, y1, x2, y2 = bbox
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


def bbox_area(bbox: tuple[float, float, float, float]) -> float:
    x1, y1, x2, y2 = bbox
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def parse_roi(value: str | tuple[float, float, float, float] | NormalizedROI) -> NormalizedROI:
    if isinstance(value, NormalizedROI):
        roi = value
    elif isinstance(value, tuple):
        if len(value) != 4:
            raise ValueError("ROI tuple must contain exactly four values: x1,y1,x2,y2")
        roi = NormalizedROI(*map(float, value))
    else:
        parts = [part.strip() for part in str(value).split(",")]
        if len(parts) != 4:
            raise ValueError("ROI must be four comma-separated values: x1,y1,x2,y2")
        try:
            roi = NormalizedROI(*(float(part) for part in parts))
        except ValueError as exc:
            raise ValueError("ROI values must be numeric") from exc
    validate_roi(roi)
    return roi


def validate_roi(roi: NormalizedROI) -> None:
    for name, value in (
        ("x1", roi.x1),
        ("y1", roi.y1),
        ("x2", roi.x2),
        ("y2", roi.y2),
    ):
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"ROI {name} must be between 0 and 1 inclusive")
    if roi.x2 <= roi.x1:
        raise ValueError("ROI x2 must be greater than x1")
    if roi.y2 <= roi.y1:
        raise ValueError("ROI y2 must be greater than y1")


def center_in_roi(
    bbox: tuple[float, float, float, float],
    roi: NormalizedROI,
    frame_width: int,
    frame_height: int,
) -> bool:
    cx, cy = bbox_center(bbox)
    x1, y1, x2, y2 = roi.pixel_bounds(frame_width, frame_height)
    return x1 <= cx <= x2 and y1 <= cy <= y2


def clipped_bbox_area_to_roi(
    bbox: tuple[float, float, float, float],
    roi: NormalizedROI,
    frame_width: int,
    frame_height: int,
) -> float:
    bx1, by1, bx2, by2 = bbox
    rx1, ry1, rx2, ry2 = roi.pixel_bounds(frame_width, frame_height)
    inter_x1 = max(bx1, rx1)
    inter_y1 = max(by1, ry1)
    inter_x2 = min(bx2, rx2)
    inter_y2 = min(by2, ry2)
    return max(0.0, inter_x2 - inter_x1) * max(0.0, inter_y2 - inter_y1)


def occupancy_ratio(
    detections: Iterable[TrafficDetection],
    roi: NormalizedROI,
    frame_width: int,
    frame_height: int,
) -> float:
    roi_area = roi.pixel_area(frame_width, frame_height)
    if roi_area <= 0:
        return 0.0
    occupied = 0.0
    for detection in detections:
        if detection.class_name not in VEHICLE_CLASSES:
            continue
        if not center_in_roi(detection.bbox, roi, frame_width, frame_height):
            continue
        occupied += clipped_bbox_area_to_roi(
            detection.bbox, roi, frame_width, frame_height
        )
    return max(0.0, min(1.0, occupied / roi_area))


@dataclass(frozen=True, slots=True)
class FrameTrafficSnapshot:
    frame_index: int
    video_time_s: float
    vehicle_count: int
    occupancy: float
    class_counts: dict[str, int]
    person_count: int
    roi_detections: list[TrafficDetection]


def summarize_frame(
    detections: list[TrafficDetection],
    *,
    roi: NormalizedROI,
    frame_width: int,
    frame_height: int,
    frame_index: int,
    video_time_s: float,
) -> FrameTrafficSnapshot:
    roi_detections = [
        detection
        for detection in detections
        if center_in_roi(detection.bbox, roi, frame_width, frame_height)
    ]
    class_counts: dict[str, int] = {name: 0 for name in sorted(VEHICLE_CLASSES)}
    person_count = 0
    vehicle_count = 0
    for detection in roi_detections:
        if detection.class_name in VEHICLE_CLASSES:
            class_counts[detection.class_name] = class_counts.get(detection.class_name, 0) + 1
            vehicle_count += 1
        elif detection.class_name == PERSON_CLASS:
            person_count += 1
    return FrameTrafficSnapshot(
        frame_index=frame_index,
        video_time_s=video_time_s,
        vehicle_count=vehicle_count,
        occupancy=occupancy_ratio(roi_detections, roi, frame_width, frame_height),
        class_counts=class_counts,
        person_count=person_count,
        roi_detections=roi_detections,
    )


@dataclass(slots=True)
class UniqueVehicleCounter:
    """Count each valid ByteTrack ID once when it first enters the ROI."""

    seen_track_ids: set[int] = field(default_factory=set)
    totals_by_class: Counter[str] = field(default_factory=Counter)
    total_unique: int = 0

    def observe(self, detections: Iterable[TrafficDetection]) -> list[TrafficDetection]:
        """Return detections whose track IDs were counted for the first time."""
        newly_counted: list[TrafficDetection] = []
        for detection in detections:
            if detection.class_name not in VEHICLE_CLASSES:
                continue
            if detection.track_id is None or detection.track_id < 0:
                continue
            if detection.track_id in self.seen_track_ids:
                continue
            self.seen_track_ids.add(detection.track_id)
            self.totals_by_class[detection.class_name] += 1
            self.total_unique += 1
            newly_counted.append(detection)
        return newly_counted


@dataclass(frozen=True, slots=True)
class TrafficWindow:
    window_index: int
    start_time_s: float
    end_time_s: float
    midpoint_time_s: float
    processed_frames: int
    mean_vehicle_count: float
    peak_vehicle_count: int
    mean_occupancy: float
    peak_occupancy: float
    unique_entries: int
    class_counts: dict[str, int]
    person_count: int
    latitude: float
    longitude: float
    congested: bool
    is_partial: bool = False


@dataclass(slots=True)
class TrafficWindowAggregator:
    """Aggregate fixed-duration traffic windows and attach GPS at midpoint."""

    window_seconds: float
    _window_index: int = 0
    _start_time_s: float | None = None
    _frames: list[FrameTrafficSnapshot] = field(default_factory=list)
    _unique_in_window: set[int] = field(default_factory=set)
    _class_counts: Counter[str] = field(default_factory=Counter)
    _person_count: int = 0
    windows: list[TrafficWindow] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.window_seconds <= 0:
            raise ValueError("window_seconds must be greater than 0")

    def _ensure_start(self, video_time_s: float) -> None:
        if self._start_time_s is None:
            self._start_time_s = video_time_s

    def add_frame(
        self,
        snapshot: FrameTrafficSnapshot,
        *,
        newly_counted: Iterable[TrafficDetection],
        gps_lookup,
    ) -> list[TrafficWindow]:
        """Add a processed frame. gps_lookup(timestamp_s) -> object with lat/lon."""
        closed: list[TrafficWindow] = []
        self._ensure_start(snapshot.video_time_s)
        assert self._start_time_s is not None

        while snapshot.video_time_s >= self._start_time_s + self.window_seconds:
            if self._frames:
                closed.append(self._close_current(gps_lookup))
            else:
                self._start_time_s += self.window_seconds
                self._window_index += 1

        self._frames.append(snapshot)
        for detection in newly_counted:
            if detection.track_id is None:
                continue
            if detection.track_id in self._unique_in_window:
                continue
            self._unique_in_window.add(detection.track_id)
            self._class_counts[detection.class_name] += 1
        self._person_count += snapshot.person_count
        return closed

    def flush(self, gps_lookup) -> list[TrafficWindow]:
        if not self._frames:
            return []
        return [self._close_current(gps_lookup, partial=True)]

    def _close_current(self, gps_lookup, *, partial: bool = False) -> TrafficWindow:
        assert self._start_time_s is not None
        assert self._frames
        start = self._start_time_s
        end = start + self.window_seconds
        if partial:
            last_time = self._frames[-1].video_time_s
            end = max(last_time, start)
        midpoint = (start + end) / 2.0
        location = gps_lookup(midpoint)
        vehicle_counts = [frame.vehicle_count for frame in self._frames]
        occupancies = [frame.occupancy for frame in self._frames]
        window = TrafficWindow(
            window_index=self._window_index,
            start_time_s=round(start, 3),
            end_time_s=round(end, 3),
            midpoint_time_s=round(midpoint, 3),
            processed_frames=len(self._frames),
            mean_vehicle_count=round(sum(vehicle_counts) / len(vehicle_counts), 4),
            peak_vehicle_count=max(vehicle_counts),
            mean_occupancy=round(sum(occupancies) / len(occupancies), 4),
            peak_occupancy=round(max(occupancies), 4),
            unique_entries=len(self._unique_in_window),
            class_counts={
                name: int(self._class_counts.get(name, 0)) for name in sorted(VEHICLE_CLASSES)
            },
            person_count=self._person_count,
            latitude=float(location.latitude),
            longitude=float(location.longitude),
            congested=False,
            is_partial=partial,
        )
        self.windows.append(window)
        self._window_index += 1
        self._start_time_s = start + self.window_seconds
        self._frames = []
        self._unique_in_window = set()
        self._class_counts = Counter()
        self._person_count = 0
        return window


@dataclass(frozen=True, slots=True)
class BottleneckEvent:
    event_id: str
    start_time_s: float
    end_time_s: float
    triggering_window_index: int
    mean_vehicle_count: float
    mean_occupancy: float
    latitude: float
    longitude: float
    consecutive_windows: int
    status: str = "pending_review"
    method: str = "configurable_roi_heuristic"


@dataclass(slots=True)
class CongestionDetector:
    """Emit one bottleneck event per congestion episode using consecutive windows."""

    min_mean_vehicles: float
    min_mean_occupancy: float
    consecutive_windows: int
    _pending_streak: list[TrafficWindow] = field(default_factory=list)
    _episode_windows: list[TrafficWindow] = field(default_factory=list)
    _episode_active: bool = False
    _event_counter: int = 0
    events: list[BottleneckEvent] = field(default_factory=list)
    congested_window_indices: set[int] = field(default_factory=set)

    def __post_init__(self) -> None:
        if self.min_mean_vehicles < 0:
            raise ValueError("min_mean_vehicles must be >= 0")
        if not 0.0 <= self.min_mean_occupancy <= 1.0:
            raise ValueError("min_mean_occupancy must be between 0 and 1")
        if self.consecutive_windows < 1:
            raise ValueError("consecutive_windows must be at least 1")

    def qualifies(self, window: TrafficWindow) -> bool:
        return (
            not window.is_partial
            and window.mean_vehicle_count >= self.min_mean_vehicles
            and window.mean_occupancy >= self.min_mean_occupancy
        )

    def observe(self, window: TrafficWindow) -> BottleneckEvent | None:
        """Process one closed window. Partial windows never affect episodes."""
        if window.is_partial:
            return None

        if self.qualifies(window):
            emitted: BottleneckEvent | None = None
            if not self._episode_active:
                self._pending_streak.append(window)
                if len(self._pending_streak) >= self.consecutive_windows:
                    self._episode_active = True
                    self._episode_windows = list(self._pending_streak)
                    self._pending_streak = []
                    for item in self._episode_windows:
                        self.congested_window_indices.add(item.window_index)
                    emitted = self._emit_or_update_event(is_new=True)
            else:
                self._episode_windows.append(window)
                self.congested_window_indices.add(window.window_index)
                self._emit_or_update_event(is_new=False)
            return emitted

        self._pending_streak = []
        if self._episode_active:
            self._episode_active = False
            self._episode_windows = []
        return None

    def finalize(self) -> None:
        """Close an active episode at EOF using the last qualifying full window."""
        if self._episode_active and self._episode_windows:
            self._emit_or_update_event(is_new=False)
        self._episode_active = False
        self._pending_streak = []
        self._episode_windows = []

    def _emit_or_update_event(self, *, is_new: bool) -> BottleneckEvent:
        assert self._episode_windows
        windows = self._episode_windows
        mean_vehicles = sum(item.mean_vehicle_count for item in windows) / len(windows)
        mean_occ = sum(item.mean_occupancy for item in windows) / len(windows)
        latest = windows[-1]
        if is_new:
            self._event_counter += 1
            trigger = windows[self.consecutive_windows - 1]
            event = BottleneckEvent(
                event_id=f"bottleneck-{self._event_counter:04d}",
                start_time_s=windows[0].start_time_s,
                end_time_s=latest.end_time_s,
                triggering_window_index=trigger.window_index,
                mean_vehicle_count=round(mean_vehicles, 4),
                mean_occupancy=round(mean_occ, 4),
                latitude=latest.latitude,
                longitude=latest.longitude,
                consecutive_windows=self.consecutive_windows,
            )
            self.events.append(event)
            return event

        previous = self.events[-1]
        updated = BottleneckEvent(
            event_id=previous.event_id,
            start_time_s=windows[0].start_time_s,
            end_time_s=latest.end_time_s,
            triggering_window_index=previous.triggering_window_index,
            mean_vehicle_count=round(mean_vehicles, 4),
            mean_occupancy=round(mean_occ, 4),
            latitude=latest.latitude,
            longitude=latest.longitude,
            consecutive_windows=previous.consecutive_windows,
        )
        self.events[-1] = updated
        return updated

    @property
    def episode_active(self) -> bool:
        return self._episode_active


def apply_congestion_flags(
    windows: list[TrafficWindow],
    congested_indices: set[int],
) -> list[TrafficWindow]:
    """Mark windows that belonged to a confirmed congestion episode."""
    return [
        replace(window, congested=window.window_index in congested_indices)
        for window in windows
    ]


def mark_windows_congested(
    windows: list[TrafficWindow],
    events: list[BottleneckEvent],
) -> list[TrafficWindow]:
    """Backward-compatible helper for event-index marking (prefer apply_congestion_flags)."""
    congested_indexes: set[int] = set()
    for event in events:
        start_index = max(0, event.triggering_window_index - event.consecutive_windows + 1)
        for index in range(start_index, event.triggering_window_index + 1):
            congested_indexes.add(index)
    return apply_congestion_flags(windows, congested_indexes)


def validate_traffic_settings(
    *,
    confidence: float,
    frame_skip: int,
    window_seconds: float,
    congestion_min_vehicles: float,
    congestion_min_occupancy: float,
    congestion_consecutive_windows: int,
    gps_source_type: str,
    roi: NormalizedROI | None = None,
) -> None:
    if not 0.0 <= confidence <= 1.0:
        raise ValueError("confidence must be between 0 and 1")
    if frame_skip < 1:
        raise ValueError("frame_skip must be at least 1")
    if window_seconds <= 0:
        raise ValueError("window_seconds must be greater than 0")
    if congestion_min_vehicles < 0:
        raise ValueError("congestion_min_vehicles must be >= 0")
    if not 0.0 <= congestion_min_occupancy <= 1.0:
        raise ValueError("congestion_min_occupancy must be between 0 and 1")
    if congestion_consecutive_windows < 1:
        raise ValueError("congestion_consecutive_windows must be at least 1")
    if gps_source_type not in GPS_SOURCE_TYPES:
        raise ValueError(
            f"gps_source_type must be one of: {', '.join(GPS_SOURCE_TYPES)}"
        )
    if roi is not None:
        validate_roi(roi)


TRAFFIC_SUMMARY_LIMITATIONS = [
    "COCO provides generic vehicle and person classes only.",
    "COCO does not identify school children.",
    "ROI occupancy is an image-space proxy, not calibrated road occupancy.",
    "Unique vehicle counts depend on ByteTrack continuity; IDs may fragment.",
    "Bottleneck detection is a configurable prototype heuristic, not a municipal standard.",
    "Speed in km/h is not estimated without camera calibration and reliable GPS speed.",
    "Real deployment requires camera calibration, real bus GPS, and field validation.",
]
