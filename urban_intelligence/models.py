"""Typed records shared by edge-processing components."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Detection:
    """One model detection aligned to video time and GPS."""

    frame_index: int
    video_time_s: float
    class_name: str
    confidence: float
    bbox: tuple[int, int, int, int]
    latitude: float
    longitude: float
    track_id: int | str | None = None

    @property
    def temporal_key(self) -> str:
        """Return a stable tracker key, with a coarse spatial fallback."""
        if self.track_id is not None:
            return f"{self.class_name}:{self.track_id}"

        x1, y1, x2, y2 = self.bbox
        center_x = ((x1 + x2) // 2) // 64
        center_y = ((y1 + y2) // 2) // 64
        return f"{self.class_name}:grid-{center_x}-{center_y}"


@dataclass(frozen=True, slots=True)
class ConfirmedTrack:
    """A detection track that passed the temporal persistence policy."""

    key: str
    class_name: str
    average_confidence: float
    first_frame: int
    confirmed_frame: int
    hit_count: int
    detection: Detection
