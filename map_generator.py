"""Generate an interactive GIS heatmap from detections.csv."""

import sys
from pathlib import Path

import folium
import pandas as pd
from folium.plugins import HeatMap

DETECTIONS_CSV = "detections.csv"
OUTPUT_HTML = "heatmap.html"
HIGH_CONFIDENCE = 0.7


def load_detections(path: str) -> pd.DataFrame:
    csv_path = Path(path)
    if not csv_path.is_file():
        print(f"Error: {path} not found. Run quick_detect.py first to create it.")
        sys.exit(1)

    try:
        df = pd.read_csv(csv_path)
    except pd.errors.EmptyDataError:
        print(f"Error: {path} is empty. No detections to map.")
        sys.exit(1)

    if df.empty:
        print(f"Error: {path} has no detection rows. Nothing to map.")
        sys.exit(1)

    for col in ("lat", "lon", "confidence", "class"):
        if col not in df.columns:
            print(f"Error: {path} is missing required column '{col}'.")
            sys.exit(1)

    return df


def main() -> None:
    df = load_detections(DETECTIONS_CSV)

    center_lat = float(df["lat"].mean())
    center_lon = float(df["lon"].mean())

    fmap = folium.Map(location=[center_lat, center_lon], zoom_start=15)

    heat_data = [
        [float(row["lat"]), float(row["lon"]), float(row["confidence"])]
        for _, row in df.iterrows()
    ]
    HeatMap(heat_data, radius=15, blur=10).add_to(fmap)

    high_conf = df[df["confidence"] > HIGH_CONFIDENCE]
    for _, row in high_conf.iterrows():
        popup_text = f"{row['class']} ({float(row['confidence']):.2f})"
        folium.Marker(
            location=[float(row["lat"]), float(row["lon"])],
            popup=popup_text,
            icon=folium.Icon(color="red", icon="info-sign"),
        ).add_to(fmap)

    fmap.save(OUTPUT_HTML)
    print("✅ Heatmap saved to heatmap.html")


if __name__ == "__main__":
    main()
