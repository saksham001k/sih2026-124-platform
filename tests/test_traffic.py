"""Tests for ROI traffic analytics and bottleneck heuristics."""

from __future__ import annotations

import pytest

from urban_intelligence.gps import GPSPoint, GPSTrack
from urban_intelligence.traffic import (
    CongestionDetector,
    NormalizedROI,
    TrafficDetection,
    TrafficWindowAggregator,
    UniqueVehicleCounter,
    apply_congestion_flags,
    bbox_area,
    bbox_center,
    center_in_roi,
    clipped_bbox_area_to_roi,
    occupancy_ratio,
    parse_roi,
    summarize_frame,
    validate_traffic_settings,
)


def detection(
    *,
    frame: int = 0,
    time_s: float = 0.0,
    track_id: int | None = 1,
    class_name: str = "car",
    confidence: float = 0.9,
    bbox: tuple[float, float, float, float] = (10, 40, 40, 70),
) -> TrafficDetection:
    return TrafficDetection(
        frame_index=frame,
        video_time_s=time_s,
        track_id=track_id,
        class_name=class_name,
        confidence=confidence,
        bbox=bbox,
    )


def test_bbox_center_and_area() -> None:
    assert bbox_center((10, 20, 30, 40)) == (20.0, 30.0)
    assert bbox_area((10, 20, 30, 40)) == 400.0


def test_parse_roi_valid_and_invalid() -> None:
    roi = parse_roi("0.0,0.30,1.0,1.0")
    assert roi == NormalizedROI(0.0, 0.30, 1.0, 1.0)
    with pytest.raises(ValueError, match="four"):
        parse_roi("0.0,0.3,1.0")
    with pytest.raises(ValueError, match="between 0 and 1"):
        parse_roi("0.0,-0.1,1.0,1.0")
    with pytest.raises(ValueError, match="x2"):
        parse_roi("0.8,0.3,0.2,1.0")


def test_center_inside_and_outside_roi() -> None:
    roi = parse_roi("0.0,0.30,1.0,1.0")
    # Frame 100x100 → ROI y from 30..100
    assert center_in_roi((40, 50, 60, 70), roi, 100, 100) is True
    assert center_in_roi((40, 0, 60, 20), roi, 100, 100) is False


def test_clipped_area_and_occupancy_clamped() -> None:
    roi = parse_roi("0.0,0.0,1.0,1.0")
    clipped = clipped_bbox_area_to_roi((0, 0, 50, 50), roi, 100, 100)
    assert clipped == 2500.0
    huge = [
        detection(bbox=(0, 0, 200, 200), track_id=1),
        detection(bbox=(0, 0, 200, 200), track_id=2, class_name="bus"),
    ]
    assert occupancy_ratio(huge, roi, 100, 100) == 1.0


def test_summarize_frame_per_class_counts() -> None:
    roi = parse_roi("0.0,0.0,1.0,1.0")
    snapshot = summarize_frame(
        [
            detection(class_name="car", track_id=1, bbox=(10, 10, 30, 30)),
            detection(class_name="truck", track_id=2, bbox=(40, 40, 60, 60)),
            detection(class_name="person", track_id=3, bbox=(70, 70, 80, 90)),
            detection(class_name="car", track_id=None, bbox=(15, 15, 25, 25)),
        ],
        roi=roi,
        frame_width=100,
        frame_height=100,
        frame_index=0,
        video_time_s=0.0,
    )
    assert snapshot.vehicle_count == 3
    assert snapshot.class_counts["car"] == 2
    assert snapshot.class_counts["truck"] == 1
    assert snapshot.person_count == 1


def test_unique_tracked_vehicle_counted_once_and_reentry_ignored() -> None:
    counter = UniqueVehicleCounter()
    first = counter.observe(
        [detection(track_id=7, class_name="car"), detection(track_id=None, class_name="car")]
    )
    assert len(first) == 1
    assert counter.total_unique == 1
    second = counter.observe([detection(track_id=7, class_name="car")])
    assert second == []
    assert counter.total_unique == 1
    assert counter.totals_by_class["car"] == 1


def test_untracked_detection_excluded_from_unique_totals() -> None:
    counter = UniqueVehicleCounter()
    counter.observe([detection(track_id=None), detection(track_id=-1)])
    assert counter.total_unique == 0


def test_traffic_window_aggregation_and_partial_flush() -> None:
    gps = GPSTrack(
        [
            GPSPoint(0.0, 28.0, 77.0),
            GPSPoint(20.0, 28.1, 77.1),
        ]
    )
    aggregator = TrafficWindowAggregator(window_seconds=5.0)
    roi = parse_roi("0.0,0.0,1.0,1.0")
    counter = UniqueVehicleCounter()

    closed_all = []
    for index, time_s in enumerate([0.0, 1.0, 2.0, 5.0, 6.0]):
        snap = summarize_frame(
            [detection(frame=index, time_s=time_s, track_id=index + 1)],
            roi=roi,
            frame_width=100,
            frame_height=100,
            frame_index=index,
            video_time_s=time_s,
        )
        newly = counter.observe(snap.roi_detections)
        closed_all.extend(
            aggregator.add_frame(snap, newly_counted=newly, gps_lookup=gps.at)
        )

    assert len(closed_all) == 1
    assert closed_all[0].window_index == 0
    assert closed_all[0].processed_frames == 3
    assert closed_all[0].unique_entries == 3
    assert closed_all[0].midpoint_time_s == pytest.approx(2.5)

    flushed = aggregator.flush(gps.at)
    assert len(flushed) == 1
    assert flushed[0].window_index == 1
    assert flushed[0].processed_frames == 2
    assert flushed[0].is_partial is True
    assert flushed[0].end_time_s == pytest.approx(6.0)
    assert flushed[0].latitude == pytest.approx(28.0275, rel=1e-4)


def test_gps_midpoint_attachment() -> None:
    gps = GPSTrack([GPSPoint(0.0, 10.0, 20.0), GPSPoint(10.0, 12.0, 22.0)])
    aggregator = TrafficWindowAggregator(window_seconds=4.0)
    roi = parse_roi("0.0,0.0,1.0,1.0")
    snap = summarize_frame(
        [detection(time_s=0.0)],
        roi=roi,
        frame_width=100,
        frame_height=100,
        frame_index=0,
        video_time_s=0.0,
    )
    aggregator.add_frame(snap, newly_counted=[], gps_lookup=gps.at)
    snap2 = summarize_frame(
        [detection(time_s=3.0, track_id=2)],
        roi=roi,
        frame_width=100,
        frame_height=100,
        frame_index=1,
        video_time_s=3.0,
    )
    aggregator.add_frame(snap2, newly_counted=[], gps_lookup=gps.at)
    windows = aggregator.flush(gps.at)
    assert windows[0].midpoint_time_s == pytest.approx(1.5)
    assert windows[0].latitude == pytest.approx(10.3, rel=1e-4)
    assert windows[0].longitude == pytest.approx(20.3, rel=1e-4)


def _qualifying_window(
    index: int,
    vehicles: float = 8.0,
    occupancy: float = 0.25,
    *,
    is_partial: bool = False,
):
    from urban_intelligence.traffic import TrafficWindow

    return TrafficWindow(
        window_index=index,
        start_time_s=float(index * 5),
        end_time_s=float(index * 5 + 5),
        midpoint_time_s=float(index * 5 + 2.5),
        processed_frames=3,
        mean_vehicle_count=vehicles,
        peak_vehicle_count=int(vehicles),
        mean_occupancy=occupancy,
        peak_occupancy=occupancy,
        unique_entries=2,
        class_counts={"bicycle": 0, "bus": 0, "car": 2, "motorcycle": 0, "truck": 0},
        person_count=0,
        latitude=28.6,
        longitude=77.2,
        congested=False,
        is_partial=is_partial,
    )


def test_congestion_episode_marks_all_qualifying_windows_and_extends_end_time() -> None:
    detector = CongestionDetector(
        min_mean_vehicles=6,
        min_mean_occupancy=0.18,
        consecutive_windows=3,
    )
    windows = [_qualifying_window(i) for i in range(4)]
    events = []
    for window in windows:
        event = detector.observe(window)
        if event is not None:
            events.append(event)

    assert len(events) == 1
    assert detector.observe(_qualifying_window(99, vehicles=1.0, occupancy=0.01)) is None
    assert len(detector.events) == 1

    marked = apply_congestion_flags(windows, detector.congested_window_indices)
    assert all(window.congested for window in marked)
    assert detector.events[0].end_time_s == pytest.approx(20.0)
    assert detector.events[0].start_time_s == pytest.approx(0.0)
    assert detector.events[0].mean_vehicle_count == pytest.approx(8.0)
    assert detector.events[0].mean_occupancy == pytest.approx(0.25)


def test_congestion_requires_consecutive_windows_and_emits_once() -> None:
    detector = CongestionDetector(
        min_mean_vehicles=6,
        min_mean_occupancy=0.18,
        consecutive_windows=3,
    )
    assert detector.observe(_qualifying_window(0)) is None
    assert detector.observe(_qualifying_window(1)) is None
    event = detector.observe(_qualifying_window(2))
    assert event is not None
    assert event.event_id == "bottleneck-0001"
    assert event.status == "pending_review"
    assert event.method == "configurable_roi_heuristic"
    assert event.consecutive_windows == 3
    # Still in episode — no second emission.
    assert detector.observe(_qualifying_window(3)) is None
    assert len(detector.events) == 1


def test_congestion_resets_after_recovery_and_second_episode_gets_new_id() -> None:
    detector = CongestionDetector(
        min_mean_vehicles=6,
        min_mean_occupancy=0.18,
        consecutive_windows=2,
    )
    assert detector.observe(_qualifying_window(0)) is None
    assert detector.observe(_qualifying_window(1)) is not None
    assert detector.observe(_qualifying_window(2, vehicles=1.0, occupancy=0.01)) is None
    assert detector.episode_active is False
    assert detector.observe(_qualifying_window(3)) is None
    second = detector.observe(_qualifying_window(4))
    assert second is not None
    assert second.event_id == "bottleneck-0002"


def test_partial_final_window_cannot_trigger_or_extend_congestion() -> None:
    detector = CongestionDetector(
        min_mean_vehicles=6,
        min_mean_occupancy=0.18,
        consecutive_windows=3,
    )
    assert detector.observe(_qualifying_window(0)) is None
    assert detector.observe(_qualifying_window(1)) is None
    partial = _qualifying_window(2, is_partial=True)
    assert detector.observe(partial) is None
    assert len(detector.events) == 0
    assert detector.congested_window_indices == set()

    # Two full qualifying windows plus one partial cannot satisfy threshold of three.
    detector = CongestionDetector(6, 0.18, 3)
    detector.observe(_qualifying_window(0))
    detector.observe(_qualifying_window(1))
    detector.observe(_qualifying_window(2, is_partial=True))
    assert len(detector.events) == 0


def test_partial_window_does_not_extend_active_episode() -> None:
    detector = CongestionDetector(6, 0.18, 2)
    detector.observe(_qualifying_window(0))
    detector.observe(_qualifying_window(1))
    assert len(detector.events) == 1
    end_before = detector.events[0].end_time_s
    detector.observe(_qualifying_window(2, is_partial=True))
    assert detector.events[0].end_time_s == end_before
    assert 2 not in detector.congested_window_indices


def test_finalize_closes_active_episode_at_eof() -> None:
    detector = CongestionDetector(6, 0.18, 2)
    detector.observe(_qualifying_window(0))
    detector.observe(_qualifying_window(1))
    detector.observe(_qualifying_window(2))
    assert detector.episode_active is True
    detector.finalize()
    assert detector.episode_active is False
    assert detector.events[0].end_time_s == pytest.approx(15.0)


def test_mark_windows_congested_and_deterministic_ids() -> None:
    detector = CongestionDetector(6, 0.18, 2)
    windows = [_qualifying_window(0), _qualifying_window(1), _qualifying_window(2, 1.0, 0.01)]
    for window in windows:
        detector.observe(window)
    marked = apply_congestion_flags(windows, detector.congested_window_indices)
    assert marked[0].congested is True
    assert marked[1].congested is True
    assert marked[2].congested is False
    assert detector.events[0].event_id == "bottleneck-0001"


def test_summary_current_values_from_last_processed_frame(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import sys

    class _FakeCrop:
        size = 1

    class _FakeFrame:
        shape = (120, 160, 3)

        def copy(self):
            return self

        def __getitem__(self, _key):
            return _FakeCrop()

    class _FakeVideoCapture:
        CAP_PROP_FPS = 5
        CAP_PROP_FRAME_WIDTH = 3
        CAP_PROP_FRAME_HEIGHT = 4

        def __init__(self, _path: str):
            self._remaining = 2

        def isOpened(self):
            return True

        def get(self, prop):
            mapping = {
                self.CAP_PROP_FPS: 10.0,
                self.CAP_PROP_FRAME_WIDTH: 160,
                self.CAP_PROP_FRAME_HEIGHT: 120,
            }
            return mapping.get(prop, 0)

        def read(self):
            if self._remaining <= 0:
                return False, None
            self._remaining -= 1
            return True, _FakeFrame()

        def release(self):
            return None

    class _FakeWriter:
        def isOpened(self):
            return True

        def write(self, _frame):
            return None

        def release(self):
            return None

    class _FakeCv2:
        CAP_PROP_FPS = _FakeVideoCapture.CAP_PROP_FPS
        CAP_PROP_FRAME_WIDTH = _FakeVideoCapture.CAP_PROP_FRAME_WIDTH
        CAP_PROP_FRAME_HEIGHT = _FakeVideoCapture.CAP_PROP_FRAME_HEIGHT
        VideoWriter_fourcc = staticmethod(lambda *_args: 0)
        FONT_HERSHEY_SIMPLEX = 0
        LINE_AA = 0

        @staticmethod
        def VideoCapture(path):
            return _FakeVideoCapture(path)

        @staticmethod
        def VideoWriter(*_args, **_kwargs):
            return _FakeWriter()

        @staticmethod
        def rectangle(*_args, **_kwargs):
            return None

        @staticmethod
        def putText(*_args, **_kwargs):
            return None

    vehicle_counts = [2, 5]

    class _TensorList:
        def __init__(self, values):
            self._values = values

        def cpu(self):
            return self

        def tolist(self):
            return self._values

    class _FakeBoxes:
        def __init__(self, count: int):
            self.xyxy = _TensorList([[10.0, 40.0, 40.0, 70.0] for _ in range(count)])
            self.conf = _TensorList([0.9] * count)
            self.cls = _TensorList([2.0] * count)
            self.id = _TensorList(list(range(1, count + 1)))

        def __len__(self):
            return len(self.xyxy._values)

    class _FakeResult:
        def __init__(self, count: int):
            self.boxes = _FakeBoxes(count)

    class _FakeModel:
        names = {2: "car"}

        def track(self, _frame, **_kwargs):
            count = vehicle_counts.pop(0)
            return [_FakeResult(count)]

    import types

    import traffic_analytics

    fake_ultra = types.SimpleNamespace(YOLO=lambda _path: _FakeModel())
    monkeypatch.setitem(sys.modules, "cv2", _FakeCv2())
    monkeypatch.setitem(sys.modules, "ultralytics", fake_ultra)

    video_path = tmp_path / "clip.mp4"
    video_path.write_bytes(b"fake")
    gps_path = tmp_path / "gps.csv"
    gps_path.write_text("timestamp,lat,lon\n0,28.6,77.2\n10,28.7,77.3\n", encoding="utf-8")

    summary = traffic_analytics.run_pipeline(
        input_path=video_path,
        gps_path=gps_path,
        output_dir=tmp_path / "out",
        model_path="yolov8n.pt",
        confidence=0.25,
        frame_skip=1,
        window_seconds=5.0,
        roi=parse_roi("0.0,0.0,1.0,1.0"),
        congestion_min_vehicles=6.0,
        congestion_min_occupancy=0.18,
        congestion_consecutive_windows=3,
        gps_source_type="synthetic_demo",
        gps_label=str(gps_path),
    )
    assert summary["current_vehicle_count"] == 5
    assert summary["current_occupancy"] >= 0.0


def test_validate_traffic_settings_rejects_invalid_values() -> None:
    base = dict(
        confidence=0.25,
        frame_skip=3,
        window_seconds=5.0,
        congestion_min_vehicles=6.0,
        congestion_min_occupancy=0.18,
        congestion_consecutive_windows=3,
        gps_source_type="synthetic_demo",
    )
    validate_traffic_settings(**base)
    with pytest.raises(ValueError, match="confidence"):
        validate_traffic_settings(**{**base, "confidence": 1.5})
    with pytest.raises(ValueError, match="frame_skip"):
        validate_traffic_settings(**{**base, "frame_skip": 0})
    with pytest.raises(ValueError, match="window_seconds"):
        validate_traffic_settings(**{**base, "window_seconds": 0})
    with pytest.raises(ValueError, match="congestion_min_occupancy"):
        validate_traffic_settings(**{**base, "congestion_min_occupancy": 1.2})
    with pytest.raises(ValueError, match="congestion_consecutive_windows"):
        validate_traffic_settings(**{**base, "congestion_consecutive_windows": 0})
    with pytest.raises(ValueError, match="gps_source_type"):
        validate_traffic_settings(**{**base, "gps_source_type": "from_filename"})


def test_traffic_module_imports_without_ultralytics() -> None:
    import traffic_analytics

    assert hasattr(traffic_analytics, "run_pipeline")
    assert hasattr(traffic_analytics, "parse_args")
