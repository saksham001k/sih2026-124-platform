"""Road hazard detection: YOLO inference on video with GPS tagging."""

import sys
from pathlib import Path

import cv2
import pandas as pd
from tqdm import tqdm
from ultralytics import YOLO

VIDEO_PATH = "input_video.mp4"
GPS_PATH = "gps_data.csv"
OUTPUT_VIDEO = "output_with_boxes.mp4"
DETECTIONS_CSV = "detections.csv"
MODEL_NAME = "yolov8n.pt"

FRAME_SKIP = 3
CONFIDENCE_THRESHOLD = 0.25
START_LAT = 28.6139
START_LON = 77.2090
GPS_DELTA = 0.0001


def open_video(path: str) -> cv2.VideoCapture:
    video_path = Path(path)
    if not video_path.is_file():
        print(f"Error: Video file not found: {path}", file=sys.stderr)
        sys.exit(1)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(
            f"Error: Video file is corrupted or cannot be opened: {path}",
            file=sys.stderr,
        )
        sys.exit(1)

    return cap


def load_gps_data(path: str) -> pd.DataFrame:
    gps_path = Path(path)
    if gps_path.is_file():
        try:
            df = pd.read_csv(gps_path)
        except pd.errors.EmptyDataError:
            print(f"Warning: {path} is empty. Generating fake GPS trajectory.", file=sys.stderr)
            return generate_fake_gps(1)
        if "lat" not in df.columns or "lon" not in df.columns:
            print(f"Warning: {path} missing lat/lon columns. Generating fake GPS.", file=sys.stderr)
            return generate_fake_gps(1)
        if df.empty:
            print(f"Warning: {path} has no rows. Generating fake GPS trajectory.", file=sys.stderr)
            return generate_fake_gps(1)
        return df

    print(f"Warning: {path} not found. Generating fake GPS trajectory.", file=sys.stderr)
    return generate_fake_gps(1)


def generate_fake_gps(num_frames: int) -> pd.DataFrame:
    rows = []
    for i in range(max(num_frames, 1)):
        rows.append(
            {
                "frame_index": i,
                "lat": START_LAT - (GPS_DELTA * i),
                "lon": START_LON + (GPS_DELTA * i),
            }
        )
    return pd.DataFrame(rows)


def ensure_gps_length(gps_df: pd.DataFrame, num_frames: int) -> pd.DataFrame:
    if len(gps_df) >= num_frames:
        return gps_df
    return generate_fake_gps(num_frames)


def get_gps_coords(gps_df: pd.DataFrame, frame_index: int) -> tuple[float, float]:
    row = gps_df.iloc[frame_index % len(gps_df)]
    return float(row["lat"]), float(row["lon"])


def draw_box(frame, x1: int, y1: int, x2: int, y2: int, label: str) -> None:
    color = (0, 255, 0)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    (text_w, text_h), baseline = cv2.getTextSize(
        label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
    )
    cv2.rectangle(
        frame,
        (x1, y1 - text_h - baseline - 4),
        (x1 + text_w, y1),
        color,
        -1,
    )
    cv2.putText(
        frame,
        label,
        (x1, y1 - baseline - 2),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 0, 0),
        2,
    )


def main() -> None:
    cap = open_video(VIDEO_PATH)

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if width <= 0 or height <= 0:
        print(
            f"Error: Video file is corrupted or has invalid dimensions: {VIDEO_PATH}",
            file=sys.stderr,
        )
        cap.release()
        sys.exit(1)

    estimated_frames = frame_count if frame_count > 0 else 300
    gps_df = load_gps_data(GPS_PATH)
    gps_df = ensure_gps_length(gps_df, estimated_frames)

    try:
        model = YOLO(MODEL_NAME)
    except Exception as exc:
        print(f"Error: Failed to load YOLO model '{MODEL_NAME}': {exc}", file=sys.stderr)
        cap.release()
        sys.exit(1)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(OUTPUT_VIDEO, fourcc, fps, (width, height))
    if not writer.isOpened():
        print(f"Error: Could not create output video: {OUTPUT_VIDEO}", file=sys.stderr)
        cap.release()
        sys.exit(1)

    detections: list[dict] = []
    frame_index = 0
    total = frame_count if frame_count > 0 else None

    try:
        with tqdm(total=total, unit="frame", desc="Processing") as progress:
            while True:
                ret, frame = cap.read()
                if not ret:
                    if frame_index == 0:
                        print(
                            f"Error: Video contains no readable frames: {VIDEO_PATH}",
                            file=sys.stderr,
                        )
                        sys.exit(1)
                    break

                lat, lon = get_gps_coords(gps_df, frame_index)

                if frame_index % FRAME_SKIP == 0:
                    try:
                        results = model(frame, conf=CONFIDENCE_THRESHOLD, verbose=False)
                    except Exception as exc:
                        print(
                            f"Error: YOLO inference failed on frame {frame_index}: {exc}",
                            file=sys.stderr,
                        )
                        cap.release()
                        writer.release()
                        sys.exit(1)

                    result = results[0]
                    if result.boxes is not None:
                        for box in result.boxes:
                            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                            cls_id = int(box.cls[0])
                            confidence = float(box.conf[0])
                            class_name = model.names[cls_id]
                            label = f"{class_name.capitalize()} {confidence:.2f}"
                            draw_box(frame, x1, y1, x2, y2, label)
                            detections.append(
                                {
                                    "frame_index": frame_index,
                                    "class": class_name,
                                    "confidence": round(confidence, 4),
                                    "lat": lat,
                                    "lon": lon,
                                }
                            )

                writer.write(frame)
                frame_index += 1
                progress.update(1)
    except cv2.error as exc:
        print(f"Error: OpenCV failure while processing video: {exc}", file=sys.stderr)
        sys.exit(1)
    finally:
        cap.release()
        writer.release()

    detections_df = pd.DataFrame(
        detections,
        columns=["frame_index", "class", "confidence", "lat", "lon"],
    )
    detections_df.to_csv(DETECTIONS_CSV, index=False)

    print(f"\nSaved annotated video to {OUTPUT_VIDEO} ({frame_index} frames @ {fps:.2f} FPS)")
    print(f"Saved detections to {DETECTIONS_CSV}")
    print(f"Total detections: {len(detections_df)}")

    if detections_df.empty:
        print("Class distribution: (no detections)")
    else:
        print("Class distribution:")
        print(detections_df["class"].value_counts().to_string())


if __name__ == "__main__":
    main()
