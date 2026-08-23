"""Stream video frames and GPS data together in real-time."""

import time

import cv2
import pandas as pd

VIDEO_PATH = "input_video.mp4"
CSV_PATH = "gps_data.csv"


def lookup_gps(gps_df: pd.DataFrame, elapsed_sec: float) -> tuple[float, float]:
    """Return lat/lon for the given elapsed time (seconds)."""
    idx = min(int(elapsed_sec), len(gps_df) - 1)
    row = gps_df.iloc[idx]
    return float(row["lat"]), float(row["lon"])


def main() -> None:
    gps_df = pd.read_csv(CSV_PATH)
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video file: {VIDEO_PATH}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_delay = 1.0 / fps
    frame_num = 0
    start_time = time.perf_counter()

    print(f"Streaming {VIDEO_PATH} with GPS from {CSV_PATH} (Ctrl+C to stop)\n")

    try:
        while True:
            ret, _ = cap.read()
            if not ret:
                break

            elapsed = time.perf_counter() - start_time
            lat, lon = lookup_gps(gps_df, elapsed)

            print(f"Frame: {frame_num}, Lat: {lat:.6f}, Lon: {lon:.6f}")

            frame_num += 1
            target_time = frame_num * frame_delay
            sleep_time = target_time - (time.perf_counter() - start_time)
            if sleep_time > 0:
                time.sleep(sleep_time)
    except KeyboardInterrupt:
        print("\nStopped by user.")
    finally:
        cap.release()
        print(f"Finished after {frame_num} frames.")


if __name__ == "__main__":
    main()
