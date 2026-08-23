"""Production-oriented ANPR evidence pipeline using FastALPR ONNX models."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import time
import uuid
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

from urban_intelligence.gps import GPSTrack, load_gps_csv

STANDARD_INDIAN_PLATE_PATTERN = re.compile(r"^[A-Z]{2}[0-9]{1,2}[A-Z]{1,3}[0-9]{4}$")
BHARAT_SERIES_PLATE_PATTERN = re.compile(r"^[0-9]{2}BH[0-9]{4}[A-Z]{1,2}$")

DEFAULT_DETECTOR_MODEL = "yolo-v9-t-384-license-plate-end2end"
DEFAULT_OCR_MODEL = "cct-xs-v2-global-model"
DEFAULT_EXECUTION_PROVIDER = "CPUExecutionProvider"


class PredictFn(Protocol):
    def __call__(self, frame) -> list[object]: ...


@dataclass(frozen=True, slots=True)
class PlateObservation:
    frame_index: int
    video_time_s: float
    bbox: tuple[int, int, int, int]
    normalized_plate: str
    ocr_confidence: float
    detector_confidence: float
    latitude: float
    longitude: float


@dataclass(slots=True)
class PlateTrack:
    track_id: str
    observations: list[PlateObservation] = field(default_factory=list)
    last_video_time_s: float = 0.0
    confirmed: bool = False

    @property
    def first_observation(self) -> PlateObservation:
        return self.observations[0]

    @property
    def last_observation(self) -> PlateObservation:
        return self.observations[-1]


def normalize_plate(text: str) -> str:
    return "".join(character for character in text.upper() if character.isalnum())


def is_standard_indian_plate(text: str) -> bool:
    return bool(STANDARD_INDIAN_PLATE_PATTERN.fullmatch(normalize_plate(text)))


def is_bharat_series_plate(text: str) -> bool:
    return bool(BHARAT_SERIES_PLATE_PATTERN.fullmatch(normalize_plate(text)))


def is_plausible_indian_plate(text: str) -> bool:
    normalized = normalize_plate(text)
    return is_standard_indian_plate(normalized) or is_bharat_series_plate(normalized)


def mask_plate(text: str) -> str:
    normalized = normalize_plate(text)
    if not normalized:
        return ""
    if len(normalized) <= 4:
        return "*" * len(normalized)
    return f"{normalized[:2]}{'*' * (len(normalized) - 4)}{normalized[-2:]}"


def consensus(candidates: list[tuple[str, float]]) -> tuple[str, float, int]:
    """Choose a multi-frame OCR result using exact vote count and mean confidence."""
    normalized = [(normalize_plate(text), confidence) for text, confidence in candidates]
    normalized = [(text, confidence) for text, confidence in normalized if text]
    if not normalized:
        return "", 0.0, 0
    counts = Counter(text for text, _ in normalized)
    winner = max(
        counts,
        key=lambda text: (
            counts[text],
            sum(score for value, score in normalized if value == text) / counts[text],
        ),
    )
    winner_scores = [score for text, score in normalized if text == winner]
    return winner, sum(winner_scores) / len(winner_scores), len(winner_scores)


def bbox_iou(
    left: tuple[int, int, int, int],
    right: tuple[int, int, int, int],
) -> float:
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


def normalized_center_distance(
    left: tuple[int, int, int, int],
    right: tuple[int, int, int, int],
    frame_width: int,
    frame_height: int,
) -> float:
    left_cx = (left[0] + left[2]) / 2
    left_cy = (left[1] + left[3]) / 2
    right_cx = (right[0] + right[2]) / 2
    right_cy = (right[1] + right[3]) / 2
    distance = math.hypot(left_cx - right_cx, left_cy - right_cy)
    diagonal = math.hypot(frame_width, frame_height)
    if diagonal <= 0:
        return float("inf")
    return distance / diagonal


@dataclass(slots=True)
class PlateTracker:
    """Associate sampled plate observations into vehicle-specific tracks."""

    iou_threshold: float = 0.30
    center_distance_threshold: float = 0.08
    track_gap_s: float = 2.0
    _tracks: list[PlateTrack] = field(default_factory=list, init=False)

    def __post_init__(self) -> None:
        if not 0.0 <= self.iou_threshold <= 1.0:
            raise ValueError("iou_threshold must be between 0.0 and 1.0")
        if self.center_distance_threshold <= 0:
            raise ValueError("center_distance_threshold must be positive")
        if self.track_gap_s <= 0:
            raise ValueError("track_gap_s must be positive")

    def _expire(self, current_time_s: float) -> None:
        self._tracks = [
            track
            for track in self._tracks
            if current_time_s - track.last_video_time_s <= self.track_gap_s
        ]

    def _match_score(
        self,
        observation: PlateObservation,
        track: PlateTrack,
        frame_width: int,
        frame_height: int,
    ) -> tuple[float, float] | None:
        latest = track.last_observation
        overlap = bbox_iou(observation.bbox, latest.bbox)
        center_distance = normalized_center_distance(
            observation.bbox,
            latest.bbox,
            frame_width,
            frame_height,
        )
        if overlap >= self.iou_threshold:
            return overlap, center_distance
        if center_distance <= self.center_distance_threshold:
            return overlap, center_distance
        return None

    def update(
        self,
        observations: list[PlateObservation],
        frame_width: int,
        frame_height: int,
    ) -> None:
        if not observations:
            return

        current_time_s = observations[0].video_time_s
        self._expire(current_time_s)

        ordered = sorted(
            observations,
            key=lambda item: (-item.detector_confidence, item.bbox[0], item.bbox[1]),
        )
        matched_track_ids: set[str] = set()
        matched_observations: list[PlateObservation] = []

        for observation in ordered:
            if any(
                bbox_iou(observation.bbox, matched.bbox) >= self.iou_threshold
                for matched in matched_observations
            ):
                continue

            best_track: PlateTrack | None = None
            best_score: tuple[float, float] | None = None
            for track in self._tracks:
                if track.track_id in matched_track_ids:
                    continue
                score = self._match_score(observation, track, frame_width, frame_height)
                if score is None:
                    continue
                if best_score is None or score[0] > best_score[0] or (
                    score[0] == best_score[0] and score[1] < best_score[1]
                ):
                    best_track = track
                    best_score = score

            if best_track is None:
                best_track = PlateTrack(track_id=str(uuid.uuid4()))
                self._tracks.append(best_track)

            best_track.observations.append(observation)
            best_track.last_video_time_s = observation.video_time_s
            matched_track_ids.add(best_track.track_id)
            matched_observations.append(observation)

    @property
    def tracks(self) -> list[PlateTrack]:
        return list(self._tracks)


def build_confirmed_event(
    track: PlateTrack,
    *,
    min_observations: int,
    min_winning_votes: int,
    min_mean_ocr_confidence: float,
) -> dict[str, object] | None:
    if len(track.observations) < min_observations:
        return None

    ocr_candidates = [
        (observation.normalized_plate, observation.ocr_confidence)
        for observation in track.observations
        if observation.normalized_plate
    ]
    plate, mean_ocr_confidence, winning_votes = consensus(ocr_candidates)
    if not plate or winning_votes < min_winning_votes:
        return None
    if mean_ocr_confidence < min_mean_ocr_confidence:
        return None

    best_observation = max(
        track.observations,
        key=lambda item: (item.ocr_confidence, item.detector_confidence, -item.frame_index),
    )
    detector_confidences = [item.detector_confidence for item in track.observations]
    plausible = is_plausible_indian_plate(plate)
    requires_review = (
        not plausible
        or winning_votes < min_winning_votes
        or mean_ocr_confidence < min_mean_ocr_confidence
    )

    return {
        "event_id": f"anpr-{track.track_id[:8]}",
        "normalized_plate": plate,
        "masked_plate": mask_plate(plate),
        "first_frame": track.first_observation.frame_index,
        "last_frame": track.last_observation.frame_index,
        "first_video_time_s": round(track.first_observation.video_time_s, 3),
        "last_video_time_s": round(track.last_observation.video_time_s, 3),
        "observation_count": len(track.observations),
        "winning_ocr_votes": winning_votes,
        "mean_ocr_confidence": round(mean_ocr_confidence, 4),
        "mean_detector_confidence": round(sum(detector_confidences) / len(detector_confidences), 4),
        "max_detector_confidence": round(max(detector_confidences), 4),
        "bbox": list(best_observation.bbox),
        "latitude": round(best_observation.latitude, 7),
        "longitude": round(best_observation.longitude, 7),
        "plausible_indian_format": plausible,
        "requires_human_review": requires_review,
        "status": "pending_review",
    }


def extract_ocr_confidence(ocr: object | None) -> float:
    if ocr is None:
        return 0.0
    confidence = getattr(ocr, "confidence", 0.0)
    if isinstance(confidence, list):
        return sum(float(value) for value in confidence) / len(confidence) if confidence else 0.0
    return float(confidence)


def convert_alpr_results(
    results: list[object],
    *,
    frame_index: int,
    video_time_s: float,
    gps_track: GPSTrack,
) -> list[PlateObservation]:
    location = gps_track.at(video_time_s)
    observations: list[PlateObservation] = []
    for result in results:
        detection = result.detection
        bbox_obj = detection.bounding_box
        bbox = (
            int(bbox_obj.x1),
            int(bbox_obj.y1),
            int(bbox_obj.x2),
            int(bbox_obj.y2),
        )
        ocr_text = ""
        ocr_confidence = 0.0
        if result.ocr is not None and result.ocr.text:
            ocr_text = normalize_plate(result.ocr.text)
            ocr_confidence = extract_ocr_confidence(result.ocr)
        observations.append(
            PlateObservation(
                frame_index=frame_index,
                video_time_s=video_time_s,
                bbox=bbox,
                normalized_plate=ocr_text,
                ocr_confidence=ocr_confidence,
                detector_confidence=float(detection.confidence),
                latitude=location.latitude,
                longitude=location.longitude,
            )
        )
    return observations


def create_alpr_engine(
    *,
    detector_model: str,
    ocr_model: str,
    detector_confidence: float,
    execution_provider: str,
) -> PredictFn:
    try:
        from fast_alpr import ALPR
    except ImportError as exc:
        raise SystemExit(
            "Install ANPR dependencies with: pip install -r requirements-anpr.txt"
        ) from exc

    engine = ALPR(
        detector_model=detector_model,
        detector_conf_thresh=detector_confidence,
        detector_providers=[execution_provider],
        ocr_model=ocr_model,
        ocr_device="cpu",
        ocr_providers=[execution_provider],
    )
    return engine.predict


def write_observations_csv(path: Path, observations: list[PlateObservation]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "frame_index",
                "video_time_s",
                "normalized_plate",
                "masked_plate",
                "ocr_confidence",
                "detector_confidence",
                "x1",
                "y1",
                "x2",
                "y2",
                "lat",
                "lon",
            ],
        )
        writer.writeheader()
        for observation in observations:
            writer.writerow(
                {
                    "frame_index": observation.frame_index,
                    "video_time_s": round(observation.video_time_s, 3),
                    "normalized_plate": observation.normalized_plate,
                    "masked_plate": mask_plate(observation.normalized_plate),
                    "ocr_confidence": round(observation.ocr_confidence, 4),
                    "detector_confidence": round(observation.detector_confidence, 4),
                    "x1": observation.bbox[0],
                    "y1": observation.bbox[1],
                    "x2": observation.bbox[2],
                    "y2": observation.bbox[3],
                    "lat": round(observation.latitude, 7),
                    "lon": round(observation.longitude, 7),
                }
            )


def save_event_evidence(
    event: dict[str, object],
    frame,
    evidence_dir: Path,
) -> None:
    import cv2

    event_id = str(event["event_id"])
    x1, y1, x2, y2 = map(int, event["bbox"])
    height, width = frame.shape[:2]
    x1 = max(0, min(x1, width - 1))
    y1 = max(0, min(y1, height - 1))
    x2 = max(x1 + 1, min(x2, width))
    y2 = max(y1 + 1, min(y2, height))

    frame_path = evidence_dir / f"{event_id}-frame.jpg"
    crop_path = evidence_dir / f"{event_id}-crop.jpg"
    cv2.imwrite(str(frame_path), frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
    crop = frame[y1:y2, x1:x2]
    if crop.size:
        cv2.imwrite(str(crop_path), crop, [cv2.IMWRITE_JPEG_QUALITY, 82])
        event["evidence_frame"] = str(frame_path)
        event["evidence_crop"] = str(crop_path)
    else:
        event["evidence_frame"] = str(frame_path)
        event["evidence_crop"] = ""


def run_pipeline(
    *,
    input_path: Path,
    gps_path: Path,
    output_dir: Path,
    predict_fn: PredictFn,
    detector_confidence: float,
    sample_fps: float,
    min_observations: int,
    min_winning_votes: int,
    min_mean_ocr_confidence: float,
    iou_threshold: float,
    center_distance_threshold: float,
    track_gap_s: float,
    show_plate_text: bool,
    gps_label: str,
) -> dict[str, object]:
    import cv2

    capture = cv2.VideoCapture(str(input_path))
    if not capture.isOpened():
        raise SystemExit(f"Incident clip not found or unreadable: {input_path}")

    source_fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
    frame_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    sample_interval = max(1, round(source_fps / sample_fps))
    gps_track = load_gps_csv(gps_path)
    tracker = PlateTracker(
        iou_threshold=iou_threshold,
        center_distance_threshold=center_distance_threshold,
        track_gap_s=track_gap_s,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    evidence_dir = output_dir / "evidence"
    evidence_dir.mkdir(parents=True, exist_ok=True)

    all_observations: list[PlateObservation] = []
    sampled_frames = 0
    total_detections = 0
    ocr_results = 0
    frame_index = 0
    evidence_frames: dict[int, object] = {}
    start = time.perf_counter()

    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            if frame_index % sample_interval == 0:
                sampled_frames += 1
                video_time_s = frame_index / source_fps
                results = predict_fn(frame)
                observations = convert_alpr_results(
                    results,
                    frame_index=frame_index,
                    video_time_s=video_time_s,
                    gps_track=gps_track,
                )
                total_detections += len(observations)
                ocr_results += sum(1 for item in observations if item.normalized_plate)
                tracker.update(observations, frame_width, frame_height)
                all_observations.extend(observations)
                evidence_frames[frame_index] = frame.copy()
            frame_index += 1
    finally:
        capture.release()

    confirmed_events: list[dict[str, object]] = []
    for track in tracker.tracks:
        event = build_confirmed_event(
            track,
            min_observations=min_observations,
            min_winning_votes=min_winning_votes,
            min_mean_ocr_confidence=min_mean_ocr_confidence,
        )
        if event is None:
            continue
        track.confirmed = True
        evidence_frame = evidence_frames.get(int(event["last_frame"]))
        if evidence_frame is not None:
            save_event_evidence(event, evidence_frame, evidence_dir)
        confirmed_events.append(event)

    elapsed_s = time.perf_counter() - start
    metrics = {
        "source_video": str(input_path),
        "gps_source": gps_label,
        "gps_is_synthetic_demo": gps_label.endswith("gps_data.csv"),
        "detector_confidence_threshold": detector_confidence,
        "sample_fps_requested": sample_fps,
        "source_fps": round(source_fps, 3),
        "sample_interval_frames": sample_interval,
        "sampled_frames": sampled_frames,
        "total_detections": total_detections,
        "ocr_results": ocr_results,
        "confirmed_tracks": len(confirmed_events),
        "wall_time_s": round(elapsed_s, 3),
        "sampled_inference_fps": round(sampled_frames / elapsed_s, 3) if elapsed_s else 0.0,
    }

    write_observations_csv(output_dir / "anpr_observations.csv", all_observations)
    (output_dir / "anpr_events.json").write_text(
        json.dumps(confirmed_events, indent=2),
        encoding="utf-8",
    )
    strongest_event = None
    if confirmed_events:
        strongest_event = max(
            confirmed_events,
            key=lambda item: (
                float(item["mean_ocr_confidence"]),
                int(item["winning_ocr_votes"]),
                int(item["observation_count"]),
            ),
        )
        (output_dir / "anpr_result.json").write_text(
            json.dumps(strongest_event, indent=2),
            encoding="utf-8",
        )
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    console_events = confirmed_events
    masked_strongest = strongest_event
    if not show_plate_text:
        console_events = [
            {**event, "normalized_plate": event["masked_plate"]} for event in confirmed_events
        ]
        if strongest_event is not None:
            masked_strongest = {
                **strongest_event,
                "normalized_plate": strongest_event["masked_plate"],
            }

    return {
        "metrics": metrics,
        "confirmed_events": console_events,
        "strongest_event": masked_strongest,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Incident video clip")
    parser.add_argument("--gps", default="gps_data.csv", help="Timestamped GPS CSV")
    parser.add_argument("--output-dir", default="artifacts/anpr")
    parser.add_argument("--detector-model", default=DEFAULT_DETECTOR_MODEL)
    parser.add_argument("--ocr-model", default=DEFAULT_OCR_MODEL)
    parser.add_argument("--execution-provider", default=DEFAULT_EXECUTION_PROVIDER)
    parser.add_argument("--confidence", type=float, default=0.20)
    parser.add_argument("--sample-fps", type=float, default=5.0)
    parser.add_argument("--min-observations", type=int, default=3)
    parser.add_argument("--min-winning-votes", type=int, default=2)
    parser.add_argument("--min-mean-ocr-confidence", type=float, default=0.80)
    parser.add_argument("--iou-threshold", type=float, default=0.30)
    parser.add_argument("--center-distance-threshold", type=float, default=0.08)
    parser.add_argument("--track-gap-s", type=float, default=2.0)
    parser.add_argument(
        "--show-plate-text",
        action="store_true",
        help="Print full plate text to the console (local review only)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    gps_path = Path(args.gps)
    if not input_path.is_file():
        raise SystemExit(f"Incident clip not found: {input_path}")
    if not gps_path.is_file():
        raise SystemExit(f"GPS CSV not found: {gps_path}")

    predict_fn = create_alpr_engine(
        detector_model=args.detector_model,
        ocr_model=args.ocr_model,
        detector_confidence=args.confidence,
        execution_provider=args.execution_provider,
    )
    result = run_pipeline(
        input_path=input_path,
        gps_path=gps_path,
        output_dir=Path(args.output_dir),
        predict_fn=predict_fn,
        detector_confidence=args.confidence,
        sample_fps=args.sample_fps,
        min_observations=args.min_observations,
        min_winning_votes=args.min_winning_votes,
        min_mean_ocr_confidence=args.min_mean_ocr_confidence,
        iou_threshold=args.iou_threshold,
        center_distance_threshold=args.center_distance_threshold,
        track_gap_s=args.track_gap_s,
        show_plate_text=args.show_plate_text,
        gps_label=str(gps_path),
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
