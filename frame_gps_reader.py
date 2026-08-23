"""Read video frames and print matching GPS coordinates by frame index."""

import sys

import cv2
import pandas as pd

VIDEO_PATH = "input_video.mp4"
CSV_PATH = "gps_data.csv"


def load_gps_data(path: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(path)
    except FileNotFoundError:
        print(f"Error: GPS file not found: {path}", file=sys.stderr)
        sys.exit(1)
    except pd.errors.EmptyDataError:
        print(f"Error: GPS file is empty: {path}", file=sys.stderr)
        sys.exit(1)

    for col in ("lat", "lon"):
        if col not in df.columns:
            print(f"Error: GPS file missing required column '{col}'", file=sys.stderr)
            sys.exit(1)

    if df.empty:
        print(f"Error: GPS file has no rows: {path}", file=sys.stderr)
        sys.exit(1)

    return df


def gps_for_frame(gps_df: pd.DataFrame, frame_index: int) -> tuple[float, float]:
    """Return lat/lon for a frame index, looping CSV when video exceeds row count."""
    row = gps_df.iloc[frame_index % len(gps_df)]
    return float(row["lat"]), float(row["lon"])


def main() -> None:
    gps_df = load_gps_data(CSV_PATH)

    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print(f"Error: Could not open video file: {VIDEO_PATH}", file=sys.stderr)
        sys.exit(1)

    frame_num = 0
    try:
        while True:
            ret, _ = cap.read()
            if not ret:
                if frame_num == 0:
                    print("Error: Video opened but no frames could be read.", file=sys.stderr)
                    sys.exit(1)
                break

            lat, lon = gps_for_frame(gps_df, frame_num)
            print(f"Frame: {frame_num} | Lat: {lat:.4f} | Lon: {lon:.4f}")
            frame_num += 1
    except cv2.error as exc:
        print(f"Error: OpenCV failure while reading frames: {exc}", file=sys.stderr)
        sys.exit(1)
    finally:
        cap.release()


if __name__ == "__main__":
    main()
