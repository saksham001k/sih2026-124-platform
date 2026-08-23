import pytest

from urban_intelligence.models import Detection
from urban_intelligence.temporal import TemporalEventFilter


def detection(frame: int, track_id: int = 7, confidence: float = 0.8) -> Detection:
    return Detection(
        frame_index=frame,
        video_time_s=frame / 30,
        class_name="pothole",
        confidence=confidence,
        bbox=(10, 10, 30, 30),
        latitude=28.6,
        longitude=77.2,
        track_id=track_id,
    )


def test_confirms_three_hits_in_five_processed_frames() -> None:
    event_filter = TemporalEventFilter(window_size=5, min_hits=3)
    assert event_filter.update([detection(0)]) == []
    assert event_filter.update([detection(3)]) == []
    assert event_filter.update([]) == []
    confirmed = event_filter.update([detection(9)])
    assert len(confirmed) == 1
    assert confirmed[0].hit_count == 3
    assert confirmed[0].average_confidence == pytest.approx(0.8)


def test_transient_detection_is_not_confirmed() -> None:
    event_filter = TemporalEventFilter(window_size=5, min_hits=3)
    assert event_filter.update([detection(0)]) == []
    for _ in range(5):
        assert event_filter.update([]) == []


def test_confirmation_emits_only_once_per_track() -> None:
    event_filter = TemporalEventFilter(window_size=3, min_hits=2)
    event_filter.update([detection(0)])
    assert len(event_filter.update([detection(1)])) == 1
    assert event_filter.update([detection(2)]) == []
