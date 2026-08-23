import pytest

from urban_intelligence.models import Detection
from urban_intelligence.temporal import TemporalEventFilter, bbox_iou


def detection(
    frame: int,
    track_id: int | None = 7,
    confidence: float = 0.8,
    bbox: tuple[int, int, int, int] = (10, 10, 30, 30),
    class_name: str = "pothole",
) -> Detection:
    return Detection(
        frame_index=frame,
        video_time_s=frame / 30,
        class_name=class_name,
        confidence=confidence,
        bbox=bbox,
        latitude=28.6,
        longitude=77.2,
        track_id=track_id,
    )


def crack_detection(
    frame: int,
    bbox: tuple[int, int, int, int],
    confidence: float,
    track_id: int | None = None,
) -> Detection:
    return detection(
        frame=frame,
        track_id=track_id,
        confidence=confidence,
        bbox=bbox,
        class_name="longitudinal_crack",
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


def test_real_longitudinal_crack_sequence_confirms_one_event() -> None:
    event_filter = TemporalEventFilter(window_size=5, min_hits=3)
    assert event_filter.update([crack_detection(18, (1013, 651, 1112, 822), 0.2895)]) == []
    assert event_filter.update([crack_detection(19, (1015, 649, 1112, 877), 0.1900)]) == []
    assert event_filter.update([]) == []
    confirmed = event_filter.update([crack_detection(21, (1026, 675, 1130, 911), 0.1247)])

    assert len(confirmed) == 1
    assert confirmed[0].class_name == "longitudinal_crack"
    assert confirmed[0].first_frame == 18
    assert confirmed[0].confirmed_frame == 21
    assert confirmed[0].hit_count == 3
    assert confirmed[0].detection.track_id is None


def test_untracked_boxes_crossing_grid_boundary_still_associate() -> None:
    event_filter = TemporalEventFilter(window_size=5, min_hits=2, iou_threshold=0.30)
    first = detection(
        frame=0,
        track_id=None,
        bbox=(60, 60, 90, 90),
        class_name="pothole",
    )
    second = detection(
        frame=1,
        track_id=None,
        bbox=(62, 62, 92, 92),
        class_name="pothole",
    )

    assert bbox_iou(first.bbox, second.bbox) > 0.30
    event_filter.update([first])
    confirmed = event_filter.update([second])
    assert len(confirmed) == 1
    assert confirmed[0].hit_count == 2


def test_spatially_separate_same_class_boxes_do_not_associate() -> None:
    event_filter = TemporalEventFilter(window_size=5, min_hits=2, iou_threshold=0.30)
    left = detection(frame=0, track_id=None, bbox=(10, 10, 40, 40), class_name="pothole")
    right = detection(frame=1, track_id=None, bbox=(200, 200, 240, 240), class_name="pothole")

    event_filter.update([left])
    confirmed = event_filter.update([right])
    assert confirmed == []
    assert event_filter.active_tracks == 2


def test_two_overlapping_detections_in_one_frame_count_once() -> None:
    event_filter = TemporalEventFilter(window_size=3, min_hits=2, iou_threshold=0.30)
    first = detection(frame=0, track_id=None, bbox=(10, 10, 50, 50), class_name="pothole")
    duplicate = detection(frame=0, track_id=None, bbox=(12, 12, 48, 48), class_name="pothole")
    follow_up = detection(frame=1, track_id=None, bbox=(11, 11, 49, 49), class_name="pothole")

    event_filter.update([first, duplicate])
    confirmed = event_filter.update([follow_up])

    assert len(confirmed) == 1
    assert confirmed[0].hit_count == 2


def test_track_id_transition_remains_associated_by_overlap() -> None:
    event_filter = TemporalEventFilter(window_size=5, min_hits=2, iou_threshold=0.30)
    untracked = detection(frame=0, track_id=None, bbox=(30, 30, 70, 70), class_name="pothole")
    tracked = detection(frame=1, track_id=42, bbox=(31, 31, 71, 71), class_name="pothole")

    event_filter.update([untracked])
    confirmed = event_filter.update([tracked])

    assert len(confirmed) == 1
    assert confirmed[0].key == "pothole:42"
    assert confirmed[0].hit_count == 2


def test_stable_tracker_id_behavior_is_unchanged() -> None:
    event_filter = TemporalEventFilter(window_size=3, min_hits=2)
    event_filter.update([detection(0, track_id=9)])
    confirmed = event_filter.update([detection(1, track_id=9)])

    assert len(confirmed) == 1
    assert confirmed[0].key == "pothole:9"
    assert confirmed[0].hit_count == 2


def test_iou_threshold_validation() -> None:
    with pytest.raises(ValueError, match="iou_threshold"):
        TemporalEventFilter(iou_threshold=1.5)
