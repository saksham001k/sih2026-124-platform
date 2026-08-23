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
        """Return the tracker key when available; otherwise mark as untracked."""
        if self.track_id is not None:
            return f"{self.class_name}:{self.track_id}"
        return f"{self.class_name}:untracked"


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
