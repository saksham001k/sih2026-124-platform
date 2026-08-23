"""Streamlit dashboard for the SIH 2026 Urban Intelligence Platform."""

from pathlib import Path

import folium
import pandas as pd
import streamlit as st
from folium.plugins import HeatMap
from streamlit_folium import st_folium

DETECTIONS_CSV = "detections.csv"
HIGH_CONFIDENCE = 0.7


st.set_page_config(layout="wide", page_title="SIH 2026 - Urban Intelligence Platform")


@st.cache_data
def load_detections(path: str) -> pd.DataFrame:
    csv_path = Path(path)
    if not csv_path.is_file():
        return pd.DataFrame()

    try:
        df = pd.read_csv(csv_path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()

    return df


@st.cache_data
def filter_detections(df: pd.DataFrame, min_confidence: float) -> pd.DataFrame:
    if df.empty:
        return df
    return df[df["confidence"] >= min_confidence].copy()


def build_map(df: pd.DataFrame) -> folium.Map:
    if df.empty:
        return folium.Map(location=[28.6139, 77.2090], zoom_start=15)

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

    return fmap


def main() -> None:
    st.title("🚌 SIH 2026 - Urban Intelligence Platform")

    df = load_detections(DETECTIONS_CSV)
    if df.empty:
        st.error(
            f"`{DETECTIONS_CSV}` is missing or empty. "
            "Run `quick_detect.py` first to generate detections."
        )
        st.stop()

    required = {"lat", "lon", "confidence", "class"}
    missing = required - set(df.columns)
    if missing:
        st.error(f"`{DETECTIONS_CSV}` is missing columns: {', '.join(sorted(missing))}")
        st.stop()

    st.sidebar.header("Filters")
    min_confidence = st.sidebar.slider(
        "Minimum confidence",
        min_value=0.0,
        max_value=1.0,
        value=0.25,
        step=0.05,
    )

    filtered = filter_detections(df, min_confidence)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Detections", len(filtered))
    with col2:
        if filtered.empty:
            top_class = "—"
        else:
            top_class = str(filtered["class"].value_counts().idxmax())
        st.metric("Top Detected Class", top_class)
    with col3:
        unique_classes = int(filtered["class"].nunique()) if not filtered.empty else 0
        st.metric("Number of Unique Classes", unique_classes)

    st.subheader("Detection Heatmap")
    if filtered.empty:
        st.warning("No detections match the current confidence filter.")
    else:
        fmap = build_map(filtered)
        st_folium(fmap, width=700, height=400)

    st.subheader("Detections Table")
    st.dataframe(filtered, use_container_width=True)


if __name__ == "__main__":
    main()
