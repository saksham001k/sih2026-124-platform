"""Run YOLOv8n object detection on a video and save annotated output."""

import sys
from pathlib import Path

import cv2
from tqdm import tqdm
from ultralytics import YOLO

VIDEO_PATH = "input_video.mp4"
OUTPUT_PATH = "output_with_boxes.mp4"
MODEL_NAME = "yolov8n.pt"
CONFIDENCE_THRESHOLD = 0.25


def format_detection(class_name: str, confidence: float) -> str:
    return f"{class_name.capitalize()} ({confidence:.2f})"


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

    try:
        model = YOLO(MODEL_NAME)
    except Exception as exc:
        print(f"Error: Failed to load YOLO model '{MODEL_NAME}': {exc}", file=sys.stderr)
        cap.release()
        sys.exit(1)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(OUTPUT_PATH, fourcc, fps, (width, height))
    if not writer.isOpened():
        print(f"Error: Could not create output video: {OUTPUT_PATH}", file=sys.stderr)
        cap.release()
        sys.exit(1)

    total = frame_count if frame_count > 0 else None
    frame_num = 0

    try:
        with tqdm(total=total, unit="frame", desc="Processing") as progress:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                results = model(frame, conf=CONFIDENCE_THRESHOLD, verbose=False)

                detections = []
                result = results[0]
                if result.boxes is not None:
                    for box in result.boxes:
                        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                        cls_id = int(box.cls[0])
                        confidence = float(box.conf[0])
                        class_name = model.names[cls_id]
                        box_label = f"{class_name.capitalize()} {confidence:.2f}"
                        draw_box(frame, x1, y1, x2, y2, box_label)
                        detections.append(format_detection(class_name, confidence))

                detection_text = ", ".join(detections) if detections else "none"
                progress.write(f"Frame: {frame_num} | Detections: {detection_text}")

                writer.write(frame)
                frame_num += 1
                progress.update(1)
    except cv2.error as exc:
        print(f"Error: OpenCV failure while processing video: {exc}", file=sys.stderr)
        sys.exit(1)
    finally:
        cap.release()
        writer.release()

    if frame_num == 0:
        print(f"Error: Video contains no readable frames: {VIDEO_PATH}", file=sys.stderr)
        sys.exit(1)

    print(f"Saved annotated video to {OUTPUT_PATH} ({frame_num} frames @ {fps:.2f} FPS)")


if __name__ == "__main__":
    main()
