"""Edge traffic analytics: COCO YOLO + ByteTrack ROI density and bottleneck heuristics."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import time
from collections import Counter
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from urban_intelligence.edge_runtime import GeoFence
from urban_intelligence.gps import load_gps_csv
from urban_intelligence.safety import (
    SAFETY_LIMITATIONS,
    RoadSafetyAnalyzer,
    TrackedRoadUser,
)
from urban_intelligence.traffic import (
    DEFAULT_ROI,
    GPS_SOURCE_TYPES,
    PERSON_CLASS,
    TRAFFIC_SUMMARY_LIMITATIONS,
    VEHICLE_CLASSES,
    CongestionDetector,
    NormalizedROI,
    TrafficDetection,
    TrafficWindowAggregator,
    UniqueVehicleCounter,
    apply_congestion_flags,
    parse_roi,
    summarize_frame,
    validate_traffic_settings,
)


def percentile_95(values: list[float]) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, int(0.95 * len(ordered)))
    return ordered[index]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Dashcam video")
    parser.add_argument("--gps", default="gps_data.csv", help="Timestamped GPS CSV")
    parser.add_argument(
        "--gps-source-type",
        choices=GPS_SOURCE_TYPES,
        default="synthetic_demo",
        help="Explicit GPS provenance (never inferred from filename)",
    )
    parser.add_argument("--model", default="yolov8n.pt", help="Pretrained COCO Ultralytics model")
    parser.add_argument("--output-dir", default="artifacts/traffic")
    parser.add_argument("--confidence", type=float, default=0.25)
    parser.add_argument("--frame-skip", type=int, default=3)
    parser.add_argument("--window-seconds", type=float, default=5.0)
    parser.add_argument(
        "--roi",
        default=",".join(str(value) for value in DEFAULT_ROI),
        help="Normalized ROI as x1,y1,x2,y2",
    )
    parser.add_argument("--congestion-min-vehicles", type=float, default=6.0)
    parser.add_argument("--congestion-min-occupancy", type=float, default=0.18)
    parser.add_argument("--congestion-consecutive-windows", type=int, default=3)
    parser.add_argument(
        "--crossing-roi",
        default="0.05,0.35,0.95,1.0",
        help="Normalized pedestrian conflict zone as x1,y1,x2,y2",
    )
    parser.add_argument(
        "--school-zones",
        help="Optional geofence JSON; context never infers that a person is a child",
    )
    return parser.parse_args()


def write_csv(path: Path, rows: list[dict[str, Any]], columns: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def load_school_zones(path: Path | None) -> list[GeoFence]:
    if path is None:
        return []
    if not path.is_file():
        raise ValueError(f"school-zone file not found: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, list):
        raise ValueError("school-zone JSON must contain a list")
    zones: list[GeoFence] = []
    for item in value:
        if not isinstance(item, dict):
            raise ValueError("each school zone must be an object")
        zones.append(
            GeoFence(
                key=str(item["key"]),
                latitude=float(item["latitude"]),
                longitude=float(item["longitude"]),
                radius_m=float(item["radius_m"]),
            )
        )
    return zones


def run_pipeline(
    *,
    input_path: Path,
    gps_path: Path,
    output_dir: Path,
    model_path: str,
    confidence: float,
    frame_skip: int,
    window_seconds: float,
    roi: NormalizedROI,
    congestion_min_vehicles: float,
    congestion_min_occupancy: float,
    congestion_consecutive_windows: int,
    gps_source_type: str,
    gps_label: str,
    crossing_roi: NormalizedROI | None = None,
    school_zones: Sequence[GeoFence] = (),
) -> dict[str, Any]:
    import cv2
    from ultralytics import YOLO

    validate_traffic_settings(
        confidence=confidence,
        frame_skip=frame_skip,
        window_seconds=window_seconds,
        congestion_min_vehicles=congestion_min_vehicles,
        congestion_min_occupancy=congestion_min_occupancy,
        congestion_consecutive_windows=congestion_consecutive_windows,
        gps_source_type=gps_source_type,
        roi=roi,
    )

    capture = cv2.VideoCapture(str(input_path))
    if not capture.isOpened():
        raise SystemExit(f"Video not found or unreadable: {input_path}")

    source_fps = float(capture.get(cv2.CAP_PROP_FPS) or 30.0)
    frame_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if frame_width <= 0 or frame_height <= 0:
        capture.release()
        raise SystemExit(f"Invalid video frame size for: {input_path}")

    gps_track = load_gps_csv(gps_path)
    model = YOLO(model_path)
    names = model.names

    output_dir.mkdir(parents=True, exist_ok=True)
    annotated_path = output_dir / "traffic_annotated.mp4"
    writer = cv2.VideoWriter(
        str(annotated_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        max(1.0, source_fps / frame_skip),
        (frame_width, frame_height),
    )
    if not writer.isOpened():
        capture.release()
        raise SystemExit(f"Unable to open annotated video writer: {annotated_path}")

    unique_counter = UniqueVehicleCounter()
    window_agg = TrafficWindowAggregator(window_seconds=window_seconds)
    congestion = CongestionDetector(
        min_mean_vehicles=congestion_min_vehicles,
        min_mean_occupancy=congestion_min_occupancy,
        consecutive_windows=congestion_consecutive_windows,
    )
    safety = RoadSafetyAnalyzer(
        crossing_roi=crossing_roi or parse_roi("0.05,0.35,0.95,1.0")
    )

    detection_rows: list[dict[str, Any]] = []
    safety_rows: list[dict[str, Any]] = []
    inference_ms: list[float] = []
    vehicle_counts_seen: list[int] = []
    occupancy_seen: list[float] = []
    processed_frames = 0
    source_frames = 0
    frame_index = 0
    current_vehicle_count = 0
    current_occupancy = 0.0
    start = time.perf_counter()

    rx1, ry1, rx2, ry2 = roi.pixel_bounds(frame_width, frame_height)

    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            source_frames += 1
            if frame_index % frame_skip != 0:
                frame_index += 1
                continue

            video_time_s = frame_index / source_fps
            infer_start = time.perf_counter()
            results = model.track(
                frame,
                persist=True,
                conf=confidence,
                verbose=False,
                tracker="bytetrack.yaml",
            )
            inference_ms.append((time.perf_counter() - infer_start) * 1000.0)
            processed_frames += 1

            detections: list[TrafficDetection] = []
            result = results[0]
            boxes = result.boxes
            if boxes is not None and len(boxes) > 0:
                xyxy = boxes.xyxy.cpu().tolist()
                confs = boxes.conf.cpu().tolist()
                classes = boxes.cls.cpu().tolist()
                track_ids = (
                    boxes.id.cpu().tolist()
                    if boxes.id is not None
                    else [None] * len(xyxy)
                )
                for coords, conf, class_id, track_id in zip(
                    xyxy, confs, classes, track_ids, strict=True
                ):
                    class_name = str(names[int(class_id)])
                    if class_name not in VEHICLE_CLASSES and class_name != PERSON_CLASS:
                        continue
                    track_value = None if track_id is None else int(track_id)
                    detections.append(
                        TrafficDetection(
                            frame_index=frame_index,
                            video_time_s=video_time_s,
                            track_id=track_value,
                            class_name=class_name,
                            confidence=float(conf),
                            bbox=(
                                float(coords[0]),
                                float(coords[1]),
                                float(coords[2]),
                                float(coords[3]),
                            ),
                        )
                    )

            snapshot = summarize_frame(
                detections,
                roi=roi,
                frame_width=frame_width,
                frame_height=frame_height,
                frame_index=frame_index,
                video_time_s=video_time_s,
            )
            newly_counted = unique_counter.observe(snapshot.roi_detections)
            vehicle_counts_seen.append(snapshot.vehicle_count)
            occupancy_seen.append(snapshot.occupancy)
            current_vehicle_count = snapshot.vehicle_count
            current_occupancy = snapshot.occupancy

            location = gps_track.at(video_time_s)
            tracked_users = [
                TrackedRoadUser(
                    frame_index=item.frame_index,
                    video_time_s=item.video_time_s,
                    track_id=item.track_id,
                    class_name=item.class_name,
                    confidence=item.confidence,
                    bbox=item.bbox,
                    latitude=location.latitude,
                    longitude=location.longitude,
                )
                for item in detections
                if item.track_id is not None and item.track_id >= 0
            ]
            school_zone_active = any(zone.contains(location) for zone in school_zones)
            safety_events = safety.observe(
                tracked_users,
                frame_width=frame_width,
                frame_height=frame_height,
                school_zone_active=school_zone_active,
            )
            if safety_events:
                evidence_dir = output_dir / "safety_evidence"
                evidence_dir.mkdir(exist_ok=True)
                for event in safety_events:
                    row = event.as_dict()
                    evidence_path = evidence_dir / f"{event.event_id}-frame.jpg"
                    if cv2.imwrite(str(evidence_path), frame):
                        row["evidence_frame"] = str(evidence_path)
                    else:
                        row["evidence_frame"] = ""
                    safety_rows.append(row)

            for detection in snapshot.roi_detections:
                location = gps_track.at(detection.video_time_s)
                detection_rows.append(
                    {
                        "frame_index": detection.frame_index,
                        "video_time_s": round(detection.video_time_s, 3),
                        "track_id": "" if detection.track_id is None else detection.track_id,
                        "class": detection.class_name,
                        "confidence": round(detection.confidence, 4),
                        "x1": round(detection.bbox[0], 2),
                        "y1": round(detection.bbox[1], 2),
                        "x2": round(detection.bbox[2], 2),
                        "y2": round(detection.bbox[3], 2),
                        "in_roi": True,
                        "lat": round(location.latitude, 7),
                        "lon": round(location.longitude, 7),
                    }
                )

            closed = window_agg.add_frame(
                snapshot,
                newly_counted=newly_counted,
                gps_lookup=gps_track.at,
            )
            for window in closed:
                congestion.observe(window)

            annotated = frame.copy()
            cv2.rectangle(annotated, (rx1, ry1), (rx2, ry2), (0, 255, 255), 2)
            for detection in snapshot.roi_detections:
                x1, y1, x2, y2 = map(int, detection.bbox)
                color = (0, 200, 0) if detection.class_name in VEHICLE_CLASSES else (255, 180, 0)
                cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
                label = detection.class_name
                if detection.track_id is not None:
                    label = f"{label} #{detection.track_id}"
                cv2.putText(
                    annotated,
                    label,
                    (x1, max(20, y1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    color,
                    1,
                    cv2.LINE_AA,
                )

            congestion_label = "CONGESTION" if congestion.episode_active else "CLEAR"
            congestion_color = (0, 0, 255) if congestion.episode_active else (0, 200, 0)
            cv2.putText(
                annotated,
                f"ROI vehicles: {snapshot.vehicle_count}",
                (12, 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                annotated,
                f"Occupancy: {snapshot.occupancy * 100:.1f}%",
                (12, 56),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                annotated,
                congestion_label,
                (12, 84),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                congestion_color,
                2,
                cv2.LINE_AA,
            )
            writer.write(annotated)
            frame_index += 1
    finally:
        capture.release()
        writer.release()

    closed = window_agg.flush(gps_track.at)
    for window in closed:
        congestion.observe(window)
    congestion.finalize()

    if processed_frames:
        final_time_s = max(0.0, (frame_index - 1) / source_fps)
        visible_people = {
            item.track_id
            for item in tracked_users
            if item.class_name == PERSON_CLASS
        }
        for event in safety.flush(now_s=final_time_s, visible_person_ids=visible_people):
            safety_rows.append({**event.as_dict(), "evidence_frame": ""})

    windows = apply_congestion_flags(
        window_agg.windows,
        congestion.congested_window_indices,
    )
    elapsed_s = time.perf_counter() - start
    processing_fps = processed_frames / elapsed_s if elapsed_s else 0.0

    timeseries_rows: list[dict[str, Any]] = []
    for window in windows:
        row = {
            "window_index": window.window_index,
            "start_time_s": window.start_time_s,
            "end_time_s": window.end_time_s,
            "midpoint_time_s": window.midpoint_time_s,
            "processed_frames": window.processed_frames,
            "mean_vehicle_count": window.mean_vehicle_count,
            "peak_vehicle_count": window.peak_vehicle_count,
            "mean_occupancy": window.mean_occupancy,
            "peak_occupancy": window.peak_occupancy,
            "unique_entries": window.unique_entries,
            "person_count": window.person_count,
            "latitude": round(window.latitude, 7),
            "longitude": round(window.longitude, 7),
            "congested": window.congested,
            "is_partial": window.is_partial,
        }
        for class_name in sorted(VEHICLE_CLASSES):
            row[f"count_{class_name}"] = window.class_counts.get(class_name, 0)
        timeseries_rows.append(row)

    bottleneck_rows = [
        {
            "event_id": event.event_id,
            "start_time_s": event.start_time_s,
            "end_time_s": event.end_time_s,
            "triggering_window": event.triggering_window_index,
            "mean_vehicle_count": event.mean_vehicle_count,
            "mean_occupancy": event.mean_occupancy,
            "latitude": round(event.latitude, 7),
            "longitude": round(event.longitude, 7),
            "consecutive_windows": event.consecutive_windows,
            "status": event.status,
            "method": event.method,
        }
        for event in congestion.events
    ]

    summary = {
        "source_video": str(input_path),
        "model": model_path,
        "tracker": "bytetrack",
        "confidence": confidence,
        "frame_skip": frame_skip,
        "window_seconds": window_seconds,
        "roi": [roi.x1, roi.y1, roi.x2, roi.y2],
        "gps_source": gps_label,
        "gps_source_type": gps_source_type,
        "source_frames": source_frames,
        "processed_frames": processed_frames,
        "source_fps": round(source_fps, 3),
        "processing_fps": round(processing_fps, 3),
        "inference_median_ms": round(statistics.median(inference_ms), 3) if inference_ms else 0.0,
        "inference_p95_ms": round(percentile_95(inference_ms), 3),
        "total_unique_tracked_vehicles": unique_counter.total_unique,
        "totals_by_class": {
            name: int(unique_counter.totals_by_class.get(name, 0))
            for name in sorted(VEHICLE_CLASSES)
        },
        "max_vehicle_count": max(vehicle_counts_seen) if vehicle_counts_seen else 0,
        "current_vehicle_count": current_vehicle_count,
        "average_vehicle_count": (
            round(sum(vehicle_counts_seen) / len(vehicle_counts_seen), 4)
            if vehicle_counts_seen
            else 0.0
        ),
        "max_occupancy": round(max(occupancy_seen), 4) if occupancy_seen else 0.0,
        "current_occupancy": round(current_occupancy, 4),
        "average_occupancy": (
            round(sum(occupancy_seen) / len(occupancy_seen), 4) if occupancy_seen else 0.0
        ),
        "bottleneck_event_count": len(congestion.events),
        "safety_event_count": len(safety_rows),
        "safety_events_by_type": dict(
            sorted(Counter(str(row["event_type"]) for row in safety_rows).items())
        ),
        "school_zone_count": len(school_zones),
        "congestion_min_vehicles": congestion_min_vehicles,
        "congestion_min_occupancy": congestion_min_occupancy,
        "congestion_consecutive_windows": congestion_consecutive_windows,
        "limitations": TRAFFIC_SUMMARY_LIMITATIONS,
        "safety_limitations": SAFETY_LIMITATIONS,
    }

    timeseries_columns = [
        "window_index",
        "start_time_s",
        "end_time_s",
        "midpoint_time_s",
        "processed_frames",
        "mean_vehicle_count",
        "peak_vehicle_count",
        "mean_occupancy",
        "peak_occupancy",
        "unique_entries",
        *[f"count_{name}" for name in sorted(VEHICLE_CLASSES)],
        "person_count",
        "latitude",
        "longitude",
        "congested",
        "is_partial",
    ]
    write_csv(output_dir / "traffic_timeseries.csv", timeseries_rows, timeseries_columns)
    write_csv(
        output_dir / "bottleneck_events.csv",
        bottleneck_rows,
        [
            "event_id",
            "start_time_s",
            "end_time_s",
            "triggering_window",
            "mean_vehicle_count",
            "mean_occupancy",
            "latitude",
            "longitude",
            "consecutive_windows",
            "status",
            "method",
        ],
    )
    (output_dir / "safety_events.json").write_text(
        json.dumps(safety_rows, indent=2),
        encoding="utf-8",
    )
    write_csv(
        output_dir / "traffic_detections.csv",
        detection_rows,
        [
            "frame_index",
            "video_time_s",
            "track_id",
            "class",
            "confidence",
            "x1",
            "y1",
            "x2",
            "y2",
            "in_roi",
            "lat",
            "lon",
        ],
    )
    (output_dir / "traffic_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    return summary


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    gps_path = Path(args.gps)
    model_path = Path(args.model)

    if not input_path.is_file():
        raise SystemExit(f"Video not found: {input_path}")
    if not gps_path.is_file():
        raise SystemExit(f"GPS CSV not found: {gps_path}")
    # Allow Ultralytics to download default weights by name (e.g. yolov8n.pt).
    if model_path.suffix and model_path.parent != Path() and not model_path.is_file():
        if not Path(args.model).is_file() and "/" in args.model:
            raise SystemExit(f"Model not found: {args.model}")

    try:
        roi = parse_roi(args.roi)
        crossing_roi = parse_roi(args.crossing_roi)
        school_zones = load_school_zones(
            Path(args.school_zones) if args.school_zones else None
        )
        validate_traffic_settings(
            confidence=args.confidence,
            frame_skip=args.frame_skip,
            window_seconds=args.window_seconds,
            congestion_min_vehicles=args.congestion_min_vehicles,
            congestion_min_occupancy=args.congestion_min_occupancy,
            congestion_consecutive_windows=args.congestion_consecutive_windows,
            gps_source_type=args.gps_source_type,
            roi=roi,
        )
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    summary = run_pipeline(
        input_path=input_path,
        gps_path=gps_path,
        output_dir=Path(args.output_dir),
        model_path=args.model,
        confidence=args.confidence,
        frame_skip=args.frame_skip,
        window_seconds=args.window_seconds,
        roi=roi,
        congestion_min_vehicles=args.congestion_min_vehicles,
        congestion_min_occupancy=args.congestion_min_occupancy,
        congestion_consecutive_windows=args.congestion_consecutive_windows,
        gps_source_type=args.gps_source_type,
        gps_label=str(gps_path),
        crossing_roi=crossing_roi,
        school_zones=school_zones,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
