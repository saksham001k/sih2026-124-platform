"""Track-aware temporal confirmation for suppressing transient detections."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

from .classes import normalize_class_name
from .models import ConfirmedTrack, Detection


def bbox_iou(
    left: tuple[int, int, int, int],
    right: tuple[int, int, int, int],
) -> float:
    """Return intersection-over-union for two axis-aligned boxes."""
    left_x1, left_y1, left_x2, left_y2 = left
    right_x1, right_y1, right_x2, right_y2 = right

    inter_x1 = max(left_x1, right_x1)
    inter_y1 = max(left_y1, right_y1)
    inter_x2 = min(left_x2, right_x2)
    inter_y2 = min(left_y2, right_y2)

    inter_width = max(0, inter_x2 - inter_x1)
    inter_height = max(0, inter_y2 - inter_y1)
    intersection = inter_width * inter_height
    if intersection == 0:
        return 0.0

    left_area = max(0, left_x2 - left_x1) * max(0, left_y2 - left_y1)
    right_area = max(0, right_x2 - right_x1) * max(0, right_y2 - right_y1)
    union = left_area + right_area - intersection
    if union <= 0:
        return 0.0
    return intersection / union


@dataclass(slots=True)
class _TrackState:
    class_name: str
    first_frame: int
    last_seen_tick: int
    history: deque[bool]
    confidences: deque[float]
    latest: Detection
    emitted: bool = False


@dataclass(slots=True)
class TemporalEventFilter:
    """Confirm a hazard track after ``min_hits`` appearances in a sliding window.

    Tracked detections match by ``track_id``. Untracked detections associate to recent
    same-class states by highest IoU above ``iou_threshold``. Each state accepts at
    most one detection per processed frame.
    """

    window_size: int = 5
    min_hits: int = 3
    max_missed_windows: int = 2
    iou_threshold: float = 0.30
    _states: dict[str, _TrackState] = field(default_factory=dict, init=False)
    _tick: int = field(default=0, init=False)
    _anon_counter: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        if self.window_size < 1:
            raise ValueError("window_size must be positive")
        if not 1 <= self.min_hits <= self.window_size:
            raise ValueError("min_hits must be between 1 and window_size")
        if not 0.0 <= self.iou_threshold <= 1.0:
            raise ValueError("iou_threshold must be between 0.0 and 1.0")

    def _tracked_key(self, class_name: str, track_id: int | str) -> str:
        return f"{class_name}:{track_id}"

    def _is_recent(self, state: _TrackState) -> bool:
        return self._tick - state.last_seen_tick <= self.window_size

    def _detection_sort_key(self, detection: Detection) -> tuple:
        return (
            -detection.confidence,
            detection.bbox[0],
            detection.bbox[1],
            detection.bbox[2],
            detection.bbox[3],
            detection.frame_index,
        )

    def _is_duplicate_same_frame(
        self,
        detection: Detection,
        matched_detections: list[Detection],
    ) -> bool:
        """Ignore duplicate same-frame boxes already matched to a temporal state."""
        class_name = normalize_class_name(detection.class_name)
        for matched in matched_detections:
            if normalize_class_name(matched.class_name) != class_name:
                continue
            if bbox_iou(detection.bbox, matched.bbox) >= self.iou_threshold:
                return True
        return False

    def _match_state_key(
        self,
        detection: Detection,
        matched_state_keys: set[str],
    ) -> str | None:
        class_name = normalize_class_name(detection.class_name)

        if detection.track_id is not None:
            tracked_key = self._tracked_key(class_name, detection.track_id)
            if tracked_key in self._states and tracked_key not in matched_state_keys:
                return tracked_key

        best_key: str | None = None
        best_iou = -1.0
        for key, state in sorted(self._states.items()):
            if key in matched_state_keys:
                continue
            if normalize_class_name(state.class_name) != class_name:
                continue
            if not self._is_recent(state):
                continue
            overlap = bbox_iou(detection.bbox, state.latest.bbox)
            if overlap < self.iou_threshold:
                continue
            if best_key is None or overlap > best_iou or (overlap == best_iou and key < best_key):
                best_iou = overlap
                best_key = key

        return best_key

    def _create_state_key(self, detection: Detection) -> str:
        class_name = normalize_class_name(detection.class_name)
        if detection.track_id is not None:
            return self._tracked_key(class_name, detection.track_id)
        self._anon_counter += 1
        return f"{class_name}:anon-{self._anon_counter}"

    def _maybe_rekey_for_track_id(self, state_key: str, detection: Detection) -> str:
        if detection.track_id is None:
            return state_key
        class_name = normalize_class_name(detection.class_name)
        tracked_key = self._tracked_key(class_name, detection.track_id)
        if state_key == tracked_key or tracked_key in self._states:
            return state_key
        state = self._states.pop(state_key)
        self._states[tracked_key] = state
        return tracked_key

    def _activate_state(self, state_key: str, detection: Detection) -> None:
        state = self._states[state_key]
        state.history[-1] = True
        state.last_seen_tick = self._tick
        state.latest = detection
        state.confidences.append(detection.confidence)

    def update(self, detections: list[Detection]) -> list[ConfirmedTrack]:
        """Consume one processed frame and return newly confirmed tracks."""
        self._tick += 1
        for state in self._states.values():
            state.history.append(False)

        matched_state_keys: set[str] = set()
        matched_detections: list[Detection] = []

        for detection in sorted(detections, key=self._detection_sort_key):
            if self._is_duplicate_same_frame(detection, matched_detections):
                continue

            state_key = self._match_state_key(detection, matched_state_keys)
            if state_key is None:
                state_key = self._create_state_key(detection)
                self._states[state_key] = _TrackState(
                    class_name=normalize_class_name(detection.class_name),
                    first_frame=detection.frame_index,
                    last_seen_tick=self._tick,
                    history=deque(maxlen=self.window_size),
                    confidences=deque(maxlen=self.window_size),
                    latest=detection,
                )
                self._states[state_key].history.append(True)
                self._states[state_key].confidences.append(detection.confidence)
            else:
                state_key = self._maybe_rekey_for_track_id(state_key, detection)
                self._activate_state(state_key, detection)

            matched_state_keys.add(state_key)
            matched_detections.append(detection)

        confirmations: list[ConfirmedTrack] = []
        for key, state in self._states.items():
            hit_count = sum(state.history)
            if hit_count >= self.min_hits and not state.emitted:
                state.emitted = True
                confirmations.append(
                    ConfirmedTrack(
                        key=key,
                        class_name=state.class_name,
                        average_confidence=sum(state.confidences) / len(state.confidences),
                        first_frame=state.first_frame,
                        confirmed_frame=state.latest.frame_index,
                        hit_count=hit_count,
                        detection=state.latest,
                    )
                )

        expiry_ticks = self.window_size * self.max_missed_windows
        expired = [
            key
            for key, state in self._states.items()
            if self._tick - state.last_seen_tick > expiry_ticks
        ]
        for key in expired:
            del self._states[key]

        return confirmations

    @property
    def active_tracks(self) -> int:
        return len(self._states)
