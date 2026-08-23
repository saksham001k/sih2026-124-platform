"""Tests for the FastALPR-backed ANPR evidence pipeline."""

from __future__ import annotations

from dataclasses import dataclass

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
    run_pipeline,
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
    tracker = PlateTracker(iou_threshold=0.30, center_distance_threshold=0.08)
    tracker.update([observation(0, (100, 100, 180, 140))], frame_width=640, frame_height=480)
    tracker.update([observation(1, (105, 102, 185, 142))], frame_width=640, frame_height=480)
    assert len(tracker.tracks) == 1
    assert len(tracker.tracks[0].observations) == 2


def test_tracker_uses_center_distance_when_iou_is_small() -> None:
    tracker = PlateTracker(iou_threshold=0.90, center_distance_threshold=0.05)
    tracker.update([observation(0, (100, 100, 180, 140))], frame_width=640, frame_height=480)
    tracker.update([observation(1, (130, 120, 210, 160))], frame_width=640, frame_height=480)
    assert len(tracker.tracks) == 1


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
    assert len(tracker.tracks) == 2


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
    assert len(tracker.tracks) == 1
    assert len(tracker.tracks[0].observations) == 1


def test_tracker_expires_after_time_gap() -> None:
    tracker = PlateTracker(track_gap_s=1.0)
    tracker.update(
        [observation(0, (100, 100, 180, 140), video_time_s=0.0)],
        frame_width=640,
        frame_height=480,
    )
    tracker.update(
        [observation(90, (100, 100, 180, 140), video_time_s=3.5)],
        frame_width=640,
        frame_height=480,
    )
    assert len(tracker.tracks) == 1
    assert len(tracker.tracks[0].observations) == 1


def test_confirmed_event_requires_review_thresholds() -> None:
    track = anpr_pipeline.PlateTrack(track_id="abc")
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
    assert event["requires_human_review"] is False
    assert event["plausible_indian_format"] is True


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


def test_run_pipeline_masks_console_plates(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    import cv2
    import numpy as np

    video_path = tmp_path / "clip.mp4"
    gps_path = tmp_path / "gps.csv"
    gps_path.write_text(
        "timestamp,lat,lon\n0,28.6139,77.2090\n10,28.6138,77.2091\n",
        encoding="utf-8",
    )

    writer = cv2.VideoWriter(
        str(video_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        5.0,
        (160, 120),
    )
    for _ in range(5):
        writer.write(np.zeros((120, 160, 3), dtype=np.uint8))
    writer.release()

    frame_counter = {"value": 0}

    def fake_predict(frame) -> list[FakeALPRResult]:
        frame_counter["value"] += 1
        return [
            FakeALPRResult(
                detection=FakeDetection(0.95, FakeBBox(20, 20, 80, 50)),
                ocr=FakeOcr("DL01AB1234", 0.92),
            )
        ]

    result = run_pipeline(
        input_path=video_path,
        gps_path=gps_path,
        output_dir=tmp_path / "out",
        predict_fn=fake_predict,
        detector_confidence=0.2,
        sample_fps=5.0,
        min_observations=3,
        min_winning_votes=2,
        min_mean_ocr_confidence=0.80,
        iou_threshold=0.30,
        center_distance_threshold=0.08,
        track_gap_s=2.0,
        show_plate_text=False,
        gps_label=str(gps_path),
    )

    assert frame_counter["value"] >= 3
    assert result["confirmed_events"]
    assert result["confirmed_events"][0]["normalized_plate"] == "DL******34"
    assert (tmp_path / "out" / "anpr_observations.csv").is_file()
    assert (tmp_path / "out" / "metrics.json").is_file()
