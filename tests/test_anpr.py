"""Tests for the FastALPR-backed ANPR evidence pipeline."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pytest

import anpr_pipeline
from anpr_pipeline import (
    PlateObservation,
    PlateTracker,
    build_confirmed_event,
    consensus,
    convert_alpr_results,
    is_bharat_series_plate,
    is_plausible_indian_plate,
    is_standard_indian_plate,
    mask_plate,
    normalize_plate,
    object_relative_center_distance,
    plate_geometry_is_plausible,
    run_pipeline,
    validate_pipeline_settings,
)
from urban_intelligence.gps import GPSPoint, GPSTrack


@dataclass
class FakeBBox:
    x1: int
    y1: int
    x2: int
    y2: int


@dataclass
class FakeDetection:
    confidence: float
    bounding_box: FakeBBox


@dataclass
class FakeOcr:
    text: str
    confidence: float


@dataclass
class FakeALPRResult:
    detection: FakeDetection
    ocr: FakeOcr | None


def observation(
    frame: int,
    bbox: tuple[int, int, int, int],
    *,
    plate: str = "DL01AB1234",
    ocr_confidence: float = 0.9,
    detector_confidence: float = 0.9,
    video_time_s: float | None = None,
) -> PlateObservation:
    return PlateObservation(
        frame_index=frame,
        video_time_s=video_time_s if video_time_s is not None else frame / 30.0,
        bbox=bbox,
        normalized_plate=normalize_plate(plate),
        ocr_confidence=ocr_confidence,
        detector_confidence=detector_confidence,
        latitude=28.6,
        longitude=77.2,
    )


def gps_track() -> GPSTrack:
    return GPSTrack(
        [
            GPSPoint(timestamp_s=0.0, latitude=28.6139, longitude=77.2090),
            GPSPoint(timestamp_s=60.0, latitude=28.6039, longitude=77.2190),
        ]
    )


def test_normalizes_plate_text() -> None:
    assert normalize_plate("dl-01 ab 1234") == "DL01AB1234"


def test_validates_standard_indian_plate_format() -> None:
    assert is_standard_indian_plate("DL01AB1234")
    assert not is_standard_indian_plate("NOTAPLATE")


def test_validates_bharat_series_plate_format() -> None:
    assert is_bharat_series_plate("22BH1234AB")
    assert is_plausible_indian_plate("22BH1234AB")
    assert not is_bharat_series_plate("DL01AB1234")


def test_rejects_invalid_plate_format() -> None:
    assert not is_plausible_indian_plate("NOTAPLATE")


def test_consensus_prefers_multi_frame_vote() -> None:
    plate, confidence, votes = consensus(
        [("DL01AB1234", 0.8), ("dl-01-ab-1234", 0.9), ("DL01A81234", 0.95)]
    )
    assert plate == "DL01AB1234"
    assert confidence == pytest.approx(0.85)
    assert votes == 2


def test_mask_plate_hides_middle_characters() -> None:
    assert mask_plate("DL01AB1234") == "DL******34"


def test_tracker_associates_moving_overlapping_boxes() -> None:
    tracker = PlateTracker()
    tracker.update([observation(0, (100, 100, 180, 140))], frame_width=640, frame_height=480)
    tracker.update([observation(1, (105, 102, 185, 142))], frame_width=640, frame_height=480)
    assert len(tracker.active_tracks) == 1
    assert len(tracker.active_tracks[0].observations) == 2


def test_tracker_uses_center_distance_when_iou_is_small() -> None:
    # Same-size boxes with low IoU but centre within ~0.4 object diagonals.
    tracker = PlateTracker(iou_threshold=0.90, center_distance_threshold=1.5)
    tracker.update([observation(0, (100, 100, 180, 140))], frame_width=640, frame_height=480)
    tracker.update([observation(1, (130, 120, 210, 160))], frame_width=640, frame_height=480)
    assert len(tracker.active_tracks) == 1


def test_tracker_keeps_spatially_separate_plates_separate() -> None:
    tracker = PlateTracker()
    tracker.update(
        [
            observation(0, (50, 50, 120, 90)),
            observation(0, (400, 300, 470, 340), plate="HR26AB1234"),
        ],
        frame_width=640,
        frame_height=480,
    )
    assert len(tracker.active_tracks) == 2


def test_same_frame_duplicate_observations_count_once() -> None:
    tracker = PlateTracker()
    tracker.update(
        [
            observation(0, (100, 100, 180, 140)),
            observation(0, (102, 102, 178, 138)),
        ],
        frame_width=640,
        frame_height=480,
    )
    assert len(tracker.active_tracks) == 1
    assert len(tracker.active_tracks[0].observations) == 1


def test_tracker_archives_expired_tracks_and_keeps_all_tracks() -> None:
    tracker = PlateTracker(track_gap_s=1.0)
    bbox = (100, 100, 180, 140)
    for frame, time_s in ((0, 0.0), (1, 0.2), (2, 0.4)):
        tracker.update(
            [observation(frame, bbox, video_time_s=time_s)],
            frame_width=640,
            frame_height=480,
        )
    assert len(tracker.active_tracks) == 1
    assert len(tracker.active_tracks[0].observations) == 3

    tracker.update(
        [observation(90, bbox, plate="HR26CD5678", video_time_s=3.5)],
        frame_width=640,
        frame_height=480,
    )

    assert len(tracker.completed_tracks) == 1
    assert len(tracker.active_tracks) == 1
    assert len(tracker.all_tracks) == 2
    assert tracker.completed_tracks[0].track_id == "1"
    assert tracker.active_tracks[0].track_id == "2"
    assert len(tracker.completed_tracks[0].observations) == 3
    assert len(tracker.active_tracks[0].observations) == 1

    earlier = build_confirmed_event(
        tracker.completed_tracks[0],
        min_observations=3,
        min_winning_votes=2,
        min_mean_ocr_confidence=0.80,
    )
    assert earlier is not None
    assert earlier["passes_automated_quality_gate"] is True
    assert earlier["requires_human_review"] is True


def test_confirmed_event_always_requires_human_review() -> None:
    track = anpr_pipeline.PlateTrack(track_id="1")
    track.observations = [
        observation(0, (100, 100, 180, 140), ocr_confidence=0.95),
        observation(1, (105, 102, 185, 142), ocr_confidence=0.92),
        observation(2, (110, 104, 190, 144), ocr_confidence=0.91),
    ]
    event = build_confirmed_event(
        track,
        min_observations=3,
        min_winning_votes=2,
        min_mean_ocr_confidence=0.80,
    )
    assert event is not None
    assert event["requires_human_review"] is True
    assert event["status"] == "pending_review"
    assert event["passes_automated_quality_gate"] is True
    assert event["plausible_indian_format"] is True
    assert event["best_observation_frame"] == 0


def test_invalid_format_high_confidence_is_not_confirmed() -> None:
    track = anpr_pipeline.PlateTrack(track_id="2")
    track.observations = [
        observation(0, (100, 100, 180, 140), plate="NOTAPLATE", ocr_confidence=0.99),
        observation(1, (105, 102, 185, 142), plate="NOTAPLATE", ocr_confidence=0.98),
        observation(2, (110, 104, 190, 144), plate="NOTAPLATE", ocr_confidence=0.97),
    ]
    assert (
        build_confirmed_event(
            track,
            min_observations=3,
            min_winning_votes=2,
            min_mean_ocr_confidence=0.80,
        )
        is None
    )


def test_low_confidence_track_is_not_confirmed() -> None:
    track = anpr_pipeline.PlateTrack(track_id="low")
    track.observations = [
        observation(0, (100, 100, 180, 140), ocr_confidence=0.50),
        observation(1, (105, 102, 185, 142), ocr_confidence=0.55),
        observation(2, (110, 104, 190, 144), ocr_confidence=0.52),
    ]
    assert (
        build_confirmed_event(
            track,
            min_observations=3,
            min_winning_votes=2,
            min_mean_ocr_confidence=0.80,
        )
        is None
    )


def test_strict_gate_rejects_low_vote_ratio_and_weak_detector_track() -> None:
    track = anpr_pipeline.PlateTrack(track_id="noise")
    track.observations = [
        observation(index, (100 + index, 100, 180 + index, 140), plate=plate)
        for index, plate in enumerate(
            [
                "DL01AB1234",
                "DL01AB1234",
                "DL01AB1234",
                "HR26CD5678",
                "KA01EF9012",
                "UP32GH3456",
            ]
        )
    ]
    assert (
        build_confirmed_event(
            track,
            min_observations=4,
            min_winning_votes=3,
            min_mean_ocr_confidence=0.80,
            min_vote_ratio=0.60,
            min_mean_detector_confidence=0.45,
        )
        is None
    )

    weak_detector_track = anpr_pipeline.PlateTrack(track_id="weak-detector")
    weak_detector_track.observations = [
        observation(
            index,
            (100 + index, 100, 180 + index, 140),
            detector_confidence=0.30,
        )
        for index in range(4)
    ]
    assert (
        build_confirmed_event(
            weak_detector_track,
            min_observations=4,
            min_winning_votes=3,
            min_mean_ocr_confidence=0.80,
            min_vote_ratio=0.60,
            min_mean_detector_confidence=0.45,
        )
        is None
    )


def test_plate_geometry_gate_rejects_tiny_square_and_extreme_boxes() -> None:
    assert plate_geometry_is_plausible(
        (100, 100, 260, 150),
        frame_width=1280,
        frame_height=720,
    )
    assert not plate_geometry_is_plausible(
        (100, 100, 105, 105),
        frame_width=1280,
        frame_height=720,
    )
    assert not plate_geometry_is_plausible(
        (100, 100, 150, 150),
        frame_width=1280,
        frame_height=720,
    )


def test_nearby_size_incompatible_plates_are_not_merged() -> None:
    tracker = PlateTracker(iou_threshold=0.90, center_distance_threshold=1.5, area_ratio_max=2.5)
    # Nearby centres but area ratio far above the guard.
    tracker.update([observation(0, (100, 100, 140, 120))], frame_width=640, frame_height=480)
    tracker.update(
        [observation(1, (110, 105, 250, 190), plate="HR26AB1234")],
        frame_width=640,
        frame_height=480,
    )
    assert len(tracker.active_tracks) == 2
    assert object_relative_center_distance((100, 100, 140, 120), (110, 105, 250, 190)) < 1.5


def test_extract_ocr_confidence_averages_character_scores() -> None:
    ocr = FakeOcr("DL01AB1234", [0.8, 1.0])
    assert anpr_pipeline.extract_ocr_confidence(ocr) == pytest.approx(0.9)


def test_convert_alpr_results_attaches_gps_at_video_timestamp() -> None:
    results = [
        FakeALPRResult(
            detection=FakeDetection(0.9, FakeBBox(10, 10, 50, 30)),
            ocr=FakeOcr("DL01AB1234", 0.88),
        )
    ]
    observations = convert_alpr_results(
        results,
        frame_index=150,
        video_time_s=5.0,
        gps_track=gps_track(),
    )
    assert len(observations) == 1
    assert observations[0].normalized_plate == "DL01AB1234"
    assert observations[0].latitude == pytest.approx(28.6130667, rel=1e-4)
    assert observations[0].longitude == pytest.approx(77.2098333, rel=1e-4)


def test_anpr_module_imports_without_optional_runtime() -> None:
    assert anpr_pipeline.normalize_plate("aa") == "AA"
    assert "create_alpr_engine" in dir(anpr_pipeline)


def test_validate_pipeline_settings_rejects_invalid_values() -> None:
    base = dict(
        detector_confidence=0.2,
        sample_fps=5.0,
        min_observations=3,
        min_winning_votes=2,
        min_mean_ocr_confidence=0.8,
        iou_threshold=0.3,
        center_distance_threshold=1.5,
        track_gap_s=2.0,
    )
    validate_pipeline_settings(**base)

    with pytest.raises(ValueError, match="confidence"):
        validate_pipeline_settings(**{**base, "detector_confidence": 1.5})
    with pytest.raises(ValueError, match="sample_fps"):
        validate_pipeline_settings(**{**base, "sample_fps": 0})
    with pytest.raises(ValueError, match="min_observations"):
        validate_pipeline_settings(**{**base, "min_observations": 0})
    with pytest.raises(ValueError, match="min_winning_votes"):
        validate_pipeline_settings(**{**base, "min_winning_votes": 0})
    with pytest.raises(ValueError, match="cannot exceed"):
        validate_pipeline_settings(**{**base, "min_winning_votes": 5})
    with pytest.raises(ValueError, match="min_mean_ocr_confidence"):
        validate_pipeline_settings(**{**base, "min_mean_ocr_confidence": -0.1})
    with pytest.raises(ValueError, match="iou_threshold"):
        validate_pipeline_settings(**{**base, "iou_threshold": 2.0})
    with pytest.raises(ValueError, match="center_distance_threshold"):
        validate_pipeline_settings(**{**base, "center_distance_threshold": 0})
    with pytest.raises(ValueError, match="track_gap_s"):
        validate_pipeline_settings(**{**base, "track_gap_s": -1})
    with pytest.raises(ValueError, match="gps_source_type"):
        validate_pipeline_settings(**{**base, "gps_source_type": "filename_guess"})


class _FakeCrop:
    size = 1

    def __init__(self, marker: int = 0) -> None:
        self.marker = marker


class _FakeFrame:
    def __init__(self, marker: int = 0) -> None:
        self.marker = marker
        self.shape = (120, 160, 3)

    def copy(self) -> _FakeFrame:
        return _FakeFrame(self.marker)

    def __getitem__(self, _key: object) -> _FakeCrop:
        return _FakeCrop(self.marker)


class _FakeVideoCapture:
    CAP_PROP_FPS = 5
    CAP_PROP_FRAME_WIDTH = 3
    CAP_PROP_FRAME_HEIGHT = 4

    def __init__(self, _path: str, *, frame_count: int = 5) -> None:
        self._frames_left = frame_count
        self._index = 0

    def isOpened(self) -> bool:
        return True

    def get(self, prop: int) -> float | int:
        if prop == self.CAP_PROP_FPS:
            return 5.0
        if prop == self.CAP_PROP_FRAME_WIDTH:
            return 160
        if prop == self.CAP_PROP_FRAME_HEIGHT:
            return 120
        return 0

    def read(self) -> tuple[bool, _FakeFrame | None]:
        if self._frames_left <= 0:
            return False, None
        self._frames_left -= 1
        frame = _FakeFrame(self._index)
        self._index += 1
        return True, frame

    def release(self) -> None:
        return None


class _FakeCv2:
    CAP_PROP_FPS = _FakeVideoCapture.CAP_PROP_FPS
    CAP_PROP_FRAME_WIDTH = _FakeVideoCapture.CAP_PROP_FRAME_WIDTH
    CAP_PROP_FRAME_HEIGHT = _FakeVideoCapture.CAP_PROP_FRAME_HEIGHT
    IMWRITE_JPEG_QUALITY = 1
    last_writes: list[tuple[str, int]] = []

    @staticmethod
    def VideoCapture(path: str) -> _FakeVideoCapture:
        return _FakeVideoCapture(path)

    @staticmethod
    def imwrite(path: str, frame: object, _params: list[int] | None = None) -> bool:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        marker = getattr(frame, "marker", 0)
        Path(path).write_bytes(f"fake-jpeg-{marker}".encode())
        _FakeCv2.last_writes.append((path, marker))
        return True


def _pipeline_kwargs(tmp_path: Path, predict_fn, **overrides):
    video_path = tmp_path / "clip.mp4"
    video_path.write_bytes(b"fake-video")
    gps_path = tmp_path / "gps.csv"
    gps_path.write_text(
        "timestamp,lat,lon\n0,28.6139,77.2090\n10,28.6138,77.2091\n",
        encoding="utf-8",
    )
    kwargs = dict(
        input_path=video_path,
        gps_path=gps_path,
        output_dir=tmp_path / "out",
        predict_fn=predict_fn,
        detector_confidence=0.2,
        sample_fps=5.0,
        min_observations=3,
        min_winning_votes=2,
        min_mean_ocr_confidence=0.80,
        iou_threshold=0.30,
        center_distance_threshold=1.5,
        track_gap_s=2.0,
        show_plate_text=False,
        gps_label=str(gps_path),
        gps_source_type="synthetic_demo",
        detector_model="test-detector",
        ocr_model="test-ocr",
        execution_provider="CPUExecutionProvider",
    )
    kwargs.update(overrides)
    return kwargs


def test_run_pipeline_masks_console_plates(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import sys

    monkeypatch.setitem(sys.modules, "cv2", _FakeCv2())

    frame_counter = {"value": 0}

    def fake_predict(frame) -> list[FakeALPRResult]:
        frame_counter["value"] += 1
        return [
            FakeALPRResult(
                detection=FakeDetection(0.95, FakeBBox(20, 20, 80, 50)),
                ocr=FakeOcr("DL01AB1234", 0.92),
            )
        ]

    result = run_pipeline(**_pipeline_kwargs(tmp_path, fake_predict))

    assert frame_counter["value"] >= 3
    assert result["confirmed_events"]
    event = result["confirmed_events"][0]
    assert event["normalized_plate"] == "DL******34"
    assert event["requires_human_review"] is True
    assert event["passes_automated_quality_gate"] is True
    metrics = result["metrics"]
    assert metrics["gps_source_type"] == "synthetic_demo"
    assert metrics["detector_model"] == "test-detector"
    assert metrics["ocr_model"] == "test-ocr"
    assert metrics["execution_provider"] == "CPUExecutionProvider"
    assert "gps_is_synthetic_demo" not in metrics
    assert (tmp_path / "out" / "anpr_observations.csv").is_file()
    assert (tmp_path / "out" / "metrics.json").is_file()
    assert json.loads((tmp_path / "out" / "anpr_result.json").read_text()) is not None


def test_run_pipeline_writes_null_result_when_unconfirmed(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import sys

    monkeypatch.setitem(sys.modules, "cv2", _FakeCv2())

    def fake_predict(frame) -> list[FakeALPRResult]:
        return [
            FakeALPRResult(
                detection=FakeDetection(0.95, FakeBBox(20, 20, 80, 50)),
                ocr=FakeOcr("NOTAPLATE", 0.99),
            )
        ]

    result = run_pipeline(**_pipeline_kwargs(tmp_path, fake_predict))
    assert result["confirmed_events"] == []
    assert result["strongest_event"] is None
    assert json.loads((tmp_path / "out" / "anpr_result.json").read_text()) is None


def test_no_plate_clip_reports_proposals_without_claiming_plate_evidence(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import sys

    monkeypatch.setitem(sys.modules, "cv2", _FakeCv2())

    def fake_predict(frame) -> list[FakeALPRResult]:
        index = getattr(frame, "marker", 0)
        return [
            FakeALPRResult(
                detection=FakeDetection(0.91, FakeBBox(20, 20, 24, 24)),
                ocr=FakeOcr("DL01AB1234", 0.99),
            ),
            FakeALPRResult(
                detection=FakeDetection(0.80, FakeBBox(40, 40, 100, 70)),
                ocr=FakeOcr(f"DL01AB12{index:02d}", 0.97),
            ),
            FakeALPRResult(
                detection=FakeDetection(0.25, FakeBBox(80, 50, 140, 80)),
                ocr=FakeOcr("HR26CD5678", 0.95),
            ),
        ]

    result = run_pipeline(
        **_pipeline_kwargs(
            tmp_path,
            fake_predict,
            detector_confidence=0.35,
            min_observations=4,
            min_winning_votes=3,
            min_vote_ratio=0.60,
            min_mean_detector_confidence=0.45,
        )
    )
    metrics = result["metrics"]
    assert metrics["plate_like_proposals"] == 15
    assert metrics["geometry_or_confidence_rejections"] == 10
    assert metrics["quality_eligible_observations"] == 5
    assert metrics["confirmed_tracks"] == 0
    assert metrics["quality_status"] == "no_verified_plate_evidence"
    assert result["confirmed_events"] == []


def test_evidence_uses_best_observation_frame_not_last(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import sys

    monkeypatch.setitem(sys.modules, "cv2", _FakeCv2())
    _FakeCv2.last_writes = []

    # Best OCR on frame index 1; later frames are weaker.
    confidences = [0.85, 0.99, 0.86, 0.84, 0.83]

    def fake_predict(frame) -> list[FakeALPRResult]:
        index = getattr(frame, "marker", 0)
        conf = confidences[index] if index < len(confidences) else 0.80
        return [
            FakeALPRResult(
                detection=FakeDetection(0.95, FakeBBox(20 + index, 20, 80 + index, 50)),
                ocr=FakeOcr("DL01AB1234", conf),
            )
        ]

    result = run_pipeline(**_pipeline_kwargs(tmp_path, fake_predict))
    assert result["confirmed_events"]
    event = result["confirmed_events"][0]
    assert event["best_observation_frame"] == 1
    assert event["last_frame"] == 4
    assert event["bbox"] == [21, 20, 81, 50]

    evidence_frame = Path(str(event["evidence_frame"]))
    evidence_crop = Path(str(event["evidence_crop"]))
    assert evidence_frame.is_file()
    assert evidence_crop.is_file()
    assert evidence_frame.read_bytes() == b"fake-jpeg-1"
    assert evidence_crop.read_bytes() == b"fake-jpeg-1"
    assert not (tmp_path / "out" / ".candidates").exists()


def test_gps_source_type_is_recorded_explicitly(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import sys

    monkeypatch.setitem(sys.modules, "cv2", _FakeCv2())

    def fake_predict(frame) -> list[FakeALPRResult]:
        return [
            FakeALPRResult(
                detection=FakeDetection(0.95, FakeBBox(20, 20, 80, 50)),
                ocr=FakeOcr("DL01AB1234", 0.92),
            )
        ]

    result = run_pipeline(
        **_pipeline_kwargs(
            tmp_path,
            fake_predict,
            gps_source_type="real_telemetry",
            gps_label="field_bus_gps.csv",
        )
    )
    assert result["metrics"]["gps_source"] == "field_bus_gps.csv"
    assert result["metrics"]["gps_source_type"] == "real_telemetry"
