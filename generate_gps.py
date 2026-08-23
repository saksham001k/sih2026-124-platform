"""Generate synthetic GPS trajectory aligned with an input video."""

import cv2
import pandas as pd

VIDEO_PATH = "input_video.mp4"
CSV_PATH = "gps_data.csv"

START_LAT = 28.6139
START_LON = 77.2090
DELTA_DEG_PER_SEC = 0.0001  # south-east: lat decreases, lon increases
DURATION_SEC = 120


def main() -> None:
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video file: {VIDEO_PATH}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    print(f"Video: {VIDEO_PATH}")
    print(f"  FPS: {fps:.2f}, frames: {frame_count}")

    timestamps = list(range(DURATION_SEC + 1))
    rows = []
    for t in timestamps:
        lat = START_LAT - (DELTA_DEG_PER_SEC * t)
        lon = START_LON + (DELTA_DEG_PER_SEC * t)
        rows.append({"timestamp": t, "lat": lat, "lon": lon})

    df = pd.DataFrame(rows)
    df.to_csv(CSV_PATH, index=False)

    print(f"Saved {len(df)} GPS points to {CSV_PATH}")
    print(f"  Start: ({START_LAT}, {START_LON})")
    print(f"  End:   ({rows[-1]['lat']:.6f}, {rows[-1]['lon']:.6f})")


if __name__ == "__main__":
    main()
