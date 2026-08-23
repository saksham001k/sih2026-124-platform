"""Inspect waterlogging and road assets with visual evidence plus GIS inventory context."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import time
from pathlib import Path
from typing import Any

from urban_intelligence.assets import (
    DIRECT_HAZARD_CLASSES,
    AssetInventoryInspector,
    AssetObservation,
    load_asset_inventory,
    normalize_asset_class,
)
from urban_intelligence.gps import load_gps_csv
from urban_intelligence.models import Detection
from urban_intelligence.temporal import TemporalEventFilter

VISUAL_CLASSES = frozenset(
    {
        "road_divider",
        "zebra_crossing",
        "traffic_signboard",
        *DIRECT_HAZARD_CLASSES,
    }
)


def percentile_95(values: list[float]) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(0.95 * len(ordered)))]


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    columns = [
        "frame_index",
        "video_time_s",
        "class",
        "confidence",
        "track_id",
        "x1",
        "y1",
        "x2",
        "y2",
        "lat",
        "lon",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def _safe_crop(frame: Any, bbox: tuple[int, int, int, int]) -> Any:
    height, width = frame.shape[:2]
    x1, y1, x2, y2 = bbox
    return frame[max(0, y1) : min(height, y2), max(0, x1) : min(width, x2)]


def run_pipeline(
    *,
    input_path: Path,
    gps_path: Path,
    output_dir: Path,
    model_path: str,
    inventory_path: Path | None = None,
    confidence: float = 0.25,
    frame_skip: int = 3,
    window_size: int = 5,
    min_hits: int = 3,
    gps_source_type: str = "unknown",
) -> dict[str, Any]:
    """Run a single urban-assets model and write reviewable evidence artifacts."""
    if not input_path.is_file() or not gps_path.is_file():
        raise ValueError("input video and GPS CSV must exist")
    if frame_skip < 1 or window_size < 1 or not 1 <= min_hits <= window_size:
        raise ValueError("frame/window settings are invalid")
    if not 0 <= confidence <= 1:
        raise ValueError("confidence must be between 0 and 1")
    if gps_source_type not in {"real_telemetry", "synthetic_demo", "unknown"}:
        raise ValueError("gps_source_type is invalid")

    try:
        import cv2
        from ultralytics import YOLO
    except ImportError as exc:
        raise RuntimeError("Install dependencies with: pip install -r requirements.txt") from exc

    inventory = load_asset_inventory(inventory_path) if inventory_path else []
    inspector = AssetInventoryInspector(inventory)
    temporal = TemporalEventFilter(window_size=window_size, min_hits=min_hits)
    gps_track = load_gps_csv(gps_path)
    model = YOLO(model_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    evidence_dir = output_dir / "evidence"
    evidence_dir.mkdir(exist_ok=True)

    capture = cv2.VideoCapture(str(input_path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open input video: {input_path}")
    source_fps = float(capture.get(cv2.CAP_PROP_FPS) or 30.0)
    source_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    video_path = output_dir / "assets_annotated.mp4"
    writer = cv2.VideoWriter(
        str(video_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        source_fps,
        (width, height),
    )
    if not writer.isOpened():
        capture.release()
        raise RuntimeError(f"Could not create output video: {video_path}")

    rows: list[dict[str, Any]] = []
    events: list[dict[str, Any]] = []
    inference_ms: list[float] = []
    processed_frames = 0
    frame_index = 0
    last_frame: Any | None = None
    started = time.perf_counter()

    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            last_frame = frame
            annotated = frame
            if frame_index % frame_skip == 0:
                processed_frames += 1
                video_time_s = frame_index / source_fps
                location = gps_track.at(video_time_s)
                infer_started = time.perf_counter()
                results = model.track(
                    frame,
                    conf=confidence,
                    persist=True,
                    tracker="bytetrack.yaml",
                    verbose=False,
                )
                inference_ms.append((time.perf_counter() - infer_started) * 1000)
                result = results[0]
                annotated = result.plot()
                observations: list[AssetObservation] = []
                hazards: list[Detection] = []
                if result.boxes is not None:
                    for box in result.boxes:
                        class_id = int(box.cls[0])
                        class_name = normalize_asset_class(str(model.names[class_id]))
                        if class_name not in VISUAL_CLASSES:
                            continue
                        score = float(box.conf[0])
                        bbox = tuple(int(value) for value in box.xyxy[0].tolist())
                        track_id = int(box.id[0]) if box.id is not None else None
                        observation = AssetObservation(
                            frame_index=frame_index,
                            video_time_s=video_time_s,
                            class_name=class_name,
                            confidence=score,
                            latitude=location.latitude,
                            longitude=location.longitude,
                            bbox=bbox,
                        )
                        observations.append(observation)
                        rows.append(
                            {
                                "frame_index": frame_index,
                                "video_time_s": round(video_time_s, 3),
                                "class": class_name,
                                "confidence": round(score, 4),
                                "track_id": "" if track_id is None else track_id,
                                "x1": bbox[0],
                                "y1": bbox[1],
                                "x2": bbox[2],
                                "y2": bbox[3],
                                "lat": round(location.latitude, 7),
                                "lon": round(location.longitude, 7),
                            }
                        )
                        if class_name in DIRECT_HAZARD_CLASSES:
                            hazards.append(
                                Detection(
                                    frame_index=frame_index,
                                    video_time_s=video_time_s,
                                    class_name=class_name,
                                    confidence=score,
                                    bbox=bbox,
                                    latitude=location.latitude,
                                    longitude=location.longitude,
                                    track_id=track_id,
                                )
                            )

                for confirmation in temporal.update(hazards):
                    detection = confirmation.detection
                    event_id = f"asset-visual-{frame_index:06d}-{len(events) + 1:03d}"
                    frame_path = evidence_dir / f"{event_id}-frame.jpg"
                    crop_path = evidence_dir / f"{event_id}-crop.jpg"
                    crop = _safe_crop(frame, detection.bbox)
                    frame_written = bool(cv2.imwrite(str(frame_path), annotated))
                    crop_written = bool(crop.size and cv2.imwrite(str(crop_path), crop))
                    events.append(
                        {
                            "event_id": event_id,
                            "event_type": "visual_asset_hazard",
                            "class": confirmation.class_name,
                            "confidence": round(confirmation.average_confidence, 4),
                            "frame_index": frame_index,
                            "video_time_s": round(video_time_s, 3),
                            "lat": round(location.latitude, 7),
                            "lon": round(location.longitude, 7),
                            "temporal_hits": confirmation.hit_count,
                            "gps_source_type": gps_source_type,
                            "status": "pending_review",
                            "requires_human_review": True,
                            "method": "temporal_visual_detection",
                            "evidence_frame": str(frame_path) if frame_written else "",
                            "evidence_crop": str(crop_path) if crop_written else "",
                        }
                    )

                inventory_events = inspector.observe(
                    frame_index=frame_index,
                    video_time_s=video_time_s,
                    location=location,
                    observations=observations,
                )
                for item in inventory_events:
                    record = item.as_dict()
                    record["gps_source_type"] = gps_source_type
                    record["evidence_frame"] = ""
                    if bool(cv2.imwrite(str(evidence_dir / f"{item.event_id}-frame.jpg"), frame)):
                        record["evidence_frame"] = str(
                            evidence_dir / f"{item.event_id}-frame.jpg"
                        )
                    events.append(record)
            writer.write(annotated)
            frame_index += 1
    finally:
        capture.release()
        writer.release()

    for item in inspector.finalize():
        record = item.as_dict()
        record["gps_source_type"] = gps_source_type
        record["evidence_frame"] = ""
        if last_frame is not None:
            path = evidence_dir / f"{item.event_id}-frame.jpg"
            if bool(cv2.imwrite(str(path), last_frame)):
                record["evidence_frame"] = str(path)
        events.append(record)

    elapsed = time.perf_counter() - started
    _write_csv(output_dir / "asset_detections.csv", rows)
    (output_dir / "asset_events.json").write_text(
        json.dumps(events, indent=2), encoding="utf-8"
    )
    metrics = {
        "source_video": str(input_path),
        "model": model_path,
        "inventory": str(inventory_path) if inventory_path else None,
        "source_frames": source_frames,
        "source_fps": round(source_fps, 3),
        "processed_frames": processed_frames,
        "visual_detections": len(rows),
        "review_events": len(events),
        "inventory_assets": len(inventory),
        "gps_source_type": gps_source_type,
        "wall_time_s": round(elapsed, 3),
        "end_to_end_fps": round(frame_index / elapsed, 3) if elapsed else 0.0,
        "median_inference_ms": round(statistics.median(inference_ms), 3)
        if inference_ms
        else 0.0,
        "p95_inference_ms": round(percentile_95(inference_ms), 3),
        "claim_policy": (
            "Missing-asset candidates require an explicit camera-visible GIS inventory "
            "entry and always require human review."
        ),
    }
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--gps", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--inventory", help="Optional camera-visibility asset inventory JSON")
    parser.add_argument("--output-dir", default="artifacts/assets/latest")
    parser.add_argument("--confidence", type=float, default=0.25)
    parser.add_argument("--frame-skip", type=int, default=3)
    parser.add_argument("--window-size", type=int, default=5)
    parser.add_argument("--min-hits", type=int, default=3)
    parser.add_argument(
        "--gps-source-type",
        choices=("real_telemetry", "synthetic_demo", "unknown"),
        default="unknown",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metrics = run_pipeline(
        input_path=Path(args.input),
        gps_path=Path(args.gps),
        output_dir=Path(args.output_dir),
        model_path=args.model,
        inventory_path=Path(args.inventory) if args.inventory else None,
        confidence=args.confidence,
        frame_skip=args.frame_skip,
        window_size=args.window_size,
        min_hits=args.min_hits,
        gps_source_type=args.gps_source_type,
    )
    print(json.dumps(metrics, indent=2))
    print(f"Artifacts written to: {args.output_dir}")


if __name__ == "__main__":
    main()
