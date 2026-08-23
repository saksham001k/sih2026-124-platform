"""Track-aware temporal confirmation for suppressing transient detections."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

from .models import ConfirmedTrack, Detection


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
    """Confirm a track after ``min_hits`` appearances in a sliding window."""

    window_size: int = 5
    min_hits: int = 3
    max_missed_windows: int = 2
    _states: dict[str, _TrackState] = field(default_factory=dict, init=False)
    _tick: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        if self.window_size < 1:
            raise ValueError("window_size must be positive")
        if not 1 <= self.min_hits <= self.window_size:
            raise ValueError("min_hits must be between 1 and window_size")

    def update(self, detections: list[Detection]) -> list[ConfirmedTrack]:
        """Consume one processed frame and return newly confirmed tracks."""
        self._tick += 1
        for state in self._states.values():
            state.history.append(False)

        for detection in detections:
            key = detection.temporal_key
            state = self._states.get(key)
            if state is None:
                state = _TrackState(
                    class_name=detection.class_name,
                    first_frame=detection.frame_index,
                    last_seen_tick=self._tick,
                    history=deque(maxlen=self.window_size),
                    confidences=deque(maxlen=self.window_size),
                    latest=detection,
                )
                state.history.append(True)
                self._states[key] = state
            else:
                state.history[-1] = True
                state.last_seen_tick = self._tick
                state.latest = detection
            state.confidences.append(detection.confidence)

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
