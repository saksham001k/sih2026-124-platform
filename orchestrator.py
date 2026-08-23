"""End-to-end edge pipeline: tracking, GPS alignment and event confirmation."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import time
from pathlib import Path
from typing import Any

from urban_intelligence import Detection, TemporalEventFilter, haversine_m, load_gps_csv

DETECTION_COLUMNS = [
    "frame_index",
    "video_time_s",
    "track_id",
    "class",
    "confidence",
    "x1",
    "y1",
    "x2",
    "y2",
    "lat",
    "lon",
]

EVENT_COLUMNS = [
    "event_id",
    "event_type",
    "class",
    "confidence",
    "first_frame",
    "confirmed_frame",
    "video_time_s",
    "lat",
    "lon",
    "track_id",
    "temporal_hits",
    "observation_count",
    "status",
    "evidence_frame",
    "evidence_crop",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="input_video.mp4", help="Dashcam video")
    parser.add_argument("--gps", default="gps_data.csv", help="Timestamped GPS CSV")
    parser.add_argument("--model", default="yolov8n.pt", help="Ultralytics model")
    parser.add_argument("--output-dir", default="artifacts/latest")
    parser.add_argument("--confidence", type=float, default=0.25)
    parser.add_argument("--frame-skip", type=int, default=3)
    parser.add_argument("--window-size", type=int, default=5)
    parser.add_argument("--min-hits", type=int, default=3)
    parser.add_argument("--dedupe-radius-m", type=float, default=12.0)
    parser.add_argument(
        "--classes",
        default="",
        help="Comma-separated allowed classes; empty keeps every model class",
    )
    return parser.parse_args()


def percentile_95(values: list[float]) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, int(0.95 * len(ordered)))
    return ordered[index]


def directory_size(path: Path) -> int:
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def write_csv(path: Path, rows: list[dict[str, Any]], columns: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def merge_or_add_event(
    events: list[dict[str, Any]], candidate: dict[str, Any], radius_m: float
) -> bool:
    """Merge spatially repeated observations; return True for a new event."""
    for event in events:
        if event["class"] != candidate["class"]:
            continue
        distance = haversine_m(
            float(event["lat"]),
            float(event["lon"]),
            float(candidate["lat"]),
            float(candidate["lon"]),
        )
        if distance <= radius_m:
            event["observation_count"] = int(event["observation_count"]) + 1
            event["confidence"] = max(float(event["confidence"]), float(candidate["confidence"]))
            return False
    events.append(candidate)
    return True


def main() -> None:
    args = parse_args()
    if args.frame_skip < 1:
        raise ValueError("--frame-skip must be at least 1")

    try:
        import cv2
        from ultralytics import YOLO
    except ImportError as exc:
        raise SystemExit("Install dependencies with: pip install -r requirements.txt") from exc

    input_path = Path(args.input)
    gps_path = Path(args.gps)
    if not input_path.is_file():
        raise SystemExit(f"Input video not found: {input_path}")
    if not gps_path.is_file():
        raise SystemExit(f"GPS CSV not found: {gps_path}")

    output_dir = Path(args.output_dir)
    evidence_dir = output_dir / "evidence"
    evidence_dir.mkdir(parents=True, exist_ok=True)

    gps_track = load_gps_csv(gps_path)
    allowed_classes = {item.strip().lower() for item in args.classes.split(",") if item.strip()}
    temporal_filter = TemporalEventFilter(args.window_size, args.min_hits)
    model = YOLO(args.model)

    capture = cv2.VideoCapture(str(input_path))
    if not capture.isOpened():
        raise SystemExit(f"Could not open input video: {input_path}")

    source_fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    output_video = output_dir / "annotated.mp4"
    writer = cv2.VideoWriter(
        str(output_video),
        cv2.VideoWriter_fourcc(*"mp4v"),
        source_fps,
        (width, height),
    )
    if not writer.isOpened():
        capture.release()
        raise SystemExit(f"Could not create output video: {output_video}")

    detections: list[dict[str, Any]] = []
    events: list[dict[str, Any]] = []
    inference_times_ms: list[float] = []
    frame_index = 0
    processed_frames = 0
    start = time.perf_counter()

    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break

            annotated = frame
            if frame_index % args.frame_skip == 0:
                processed_frames += 1
                inference_start = time.perf_counter()
                results = model.track(
                    frame,
                    conf=args.confidence,
                    persist=True,
                    tracker="bytetrack.yaml",
                    verbose=False,
                )
                inference_times_ms.append((time.perf_counter() - inference_start) * 1000)
                result = results[0]
                annotated = result.plot()
                location = gps_track.for_frame(frame_index, source_fps)
                frame_detections: list[Detection] = []

                if result.boxes is not None:
                    for box in result.boxes:
                        class_id = int(box.cls[0])
                        class_name = str(model.names[class_id])
                        if allowed_classes and class_name.lower() not in allowed_classes:
                            continue
                        confidence = float(box.conf[0])
                        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                        track_id = int(box.id[0]) if box.id is not None else None
                        detection = Detection(
                            frame_index=frame_index,
                            video_time_s=frame_index / source_fps,
                            class_name=class_name,
                            confidence=confidence,
                            bbox=(x1, y1, x2, y2),
                            latitude=location.latitude,
                            longitude=location.longitude,
                            track_id=track_id,
                        )
                        frame_detections.append(detection)
                        detections.append(
                            {
                                "frame_index": frame_index,
                                "video_time_s": round(detection.video_time_s, 3),
                                "track_id": "" if track_id is None else track_id,
                                "class": class_name,
                                "confidence": round(confidence, 4),
                                "x1": x1,
                                "y1": y1,
                                "x2": x2,
                                "y2": y2,
                                "lat": round(location.latitude, 7),
                                "lon": round(location.longitude, 7),
                            }
                        )

                for confirmation in temporal_filter.update(frame_detections):
                    detection = confirmation.detection
                    safe_track = str(detection.track_id or "untracked").replace("/", "-")
                    event_id = f"evt-{frame_index:06d}-{safe_track}"
                    frame_path = evidence_dir / f"{event_id}-frame.jpg"
                    crop_path = evidence_dir / f"{event_id}-crop.jpg"
                    x1, y1, x2, y2 = detection.bbox
                    crop = frame[max(y1, 0) : min(y2, height), max(x1, 0) : min(x2, width)]
                    candidate = {
                        "event_id": event_id,
                        "event_type": "confirmed_detection",
                        "class": confirmation.class_name,
                        "confidence": round(confirmation.average_confidence, 4),
                        "first_frame": confirmation.first_frame,
                        "confirmed_frame": confirmation.confirmed_frame,
                        "video_time_s": round(detection.video_time_s, 3),
                        "lat": round(detection.latitude, 7),
                        "lon": round(detection.longitude, 7),
                        "track_id": "" if detection.track_id is None else detection.track_id,
                        "temporal_hits": confirmation.hit_count,
                        "observation_count": 1,
                        "status": "pending_review",
                        "evidence_frame": str(frame_path),
                        "evidence_crop": str(crop_path) if crop.size else "",
                    }
                    if merge_or_add_event(events, candidate, args.dedupe_radius_m):
                        scale = min(1.0, 960 / annotated.shape[1])
                        context = cv2.resize(
                            annotated,
                            (
                                int(annotated.shape[1] * scale),
                                int(annotated.shape[0] * scale),
                            ),
                            interpolation=cv2.INTER_AREA,
                        )
                        cv2.imwrite(
                            str(frame_path),
                            context,
                            [cv2.IMWRITE_JPEG_QUALITY, 70],
                        )
                        if crop.size:
                            cv2.imwrite(
                                str(crop_path),
                                crop,
                                [cv2.IMWRITE_JPEG_QUALITY, 82],
                            )

            writer.write(annotated)
            frame_index += 1
            if frame_index % 100 == 0:
                print(f"Processed {frame_index}/{frame_count or '?'} source frames")
    finally:
        capture.release()
        writer.release()

    elapsed_s = time.perf_counter() - start
    detections_path = output_dir / "detections.csv"
    events_path = output_dir / "events.csv"
    write_csv(detections_path, detections, DETECTION_COLUMNS)
    write_csv(events_path, events, EVENT_COLUMNS)

    metrics = {
        "source_video": str(input_path),
        "model": args.model,
        "source_frames": frame_index,
        "source_fps": round(source_fps, 3),
        "processed_frames": processed_frames,
        "frame_skip": args.frame_skip,
        "detections": len(detections),
        "confirmed_events": len(events),
        "wall_time_s": round(elapsed_s, 3),
        "end_to_end_fps": round(frame_index / elapsed_s, 3) if elapsed_s else 0.0,
        "median_inference_ms": round(statistics.median(inference_times_ms), 3)
        if inference_times_ms
        else 0.0,
        "p95_inference_ms": round(percentile_95(inference_times_ms), 3),
        "input_video_bytes": input_path.stat().st_size,
        "metadata_bytes": detections_path.stat().st_size + events_path.stat().st_size,
        "evidence_bytes": directory_size(evidence_dir),
    }
    with (output_dir / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)

    print(json.dumps(metrics, indent=2))
    print(f"Artifacts written to: {output_dir}")


if __name__ == "__main__":
    main()
