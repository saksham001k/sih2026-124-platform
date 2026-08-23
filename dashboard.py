"""Streamlit command centre for reviewed edge-AI event artifacts."""

from __future__ import annotations

import json
from pathlib import Path

import folium
import pandas as pd
import streamlit as st
from folium.plugins import HeatMap
from streamlit_folium import st_folium

st.set_page_config(
    page_title="DrishtiPath Command Centre",
    page_icon="🚌",
    layout="wide",
)

st.markdown(
    """
    <style>
      .block-container {padding-top: 1.5rem; padding-bottom: 2rem;}
      [data-testid="stMetric"] {
        background: #101c2c;
        border: 1px solid #26364a;
        padding: 0.8rem 1rem;
        border-radius: 0.75rem;
      }
      .status-pill {
        display: inline-block; padding: .25rem .65rem; border-radius: 999px;
        background: #163b31; color: #7ef0bd; font-size: .82rem;
      }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data(ttl=2)
def load_csv(path: str) -> pd.DataFrame:
    csv_path = Path(path)
    if not csv_path.is_file() or csv_path.stat().st_size == 0:
        return pd.DataFrame()
    try:
        return pd.read_csv(csv_path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


@st.cache_data(ttl=2)
def load_json(path: str) -> dict:
    json_path = Path(path)
    if not json_path.is_file():
        return {}
    with json_path.open(encoding="utf-8") as handle:
        return json.load(handle)


def human_bytes(value: int | float) -> str:
    size = float(value)
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024 or unit == "GB":
            return f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} GB"


def build_event_map(events: pd.DataFrame) -> folium.Map:
    center = [28.6139, 77.2090]
    if not events.empty:
        center = [float(events["lat"].mean()), float(events["lon"].mean())]
    event_map = folium.Map(location=center, zoom_start=15, control_scale=True)
    if events.empty:
        return event_map

    HeatMap(
        [
            [float(row["lat"]), float(row["lon"]), float(row["confidence"])]
            for _, row in events.iterrows()
        ],
        radius=20,
        blur=14,
    ).add_to(event_map)

    colors = {
        "pothole": "red",
        "person": "orange",
        "truck": "darkblue",
        "car": "blue",
    }
    for _, row in events.iterrows():
        class_name = str(row["class"])
        popup = (
            f"<b>{class_name.title()}</b><br>"
            f"Confidence: {float(row['confidence']):.2f}<br>"
            f"Status: {row['status']}"
        )
        folium.Marker(
            [float(row["lat"]), float(row["lon"])],
            popup=popup,
            icon=folium.Icon(color=colors.get(class_name, "cadetblue"), icon="info-sign"),
        ).add_to(event_map)
    return event_map


def resolve_evidence(path_value: object, artifacts_dir: Path) -> Path | None:
    if not isinstance(path_value, str) or not path_value:
        return None
    candidate = Path(path_value)
    if candidate.is_file():
        return candidate
    evidence_relative = artifacts_dir / "evidence" / candidate.name
    return evidence_relative if evidence_relative.is_file() else None


def main() -> None:
    st.title("DrishtiPath Urban Intelligence Command Centre")
    st.caption("Fleet-scale road and traffic evidence · Edge verified · Bandwidth aware")

    artifacts_value = st.sidebar.text_input("Artifacts directory", "artifacts/latest")
    artifacts_dir = Path(artifacts_value)
    if st.sidebar.button("Refresh data", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

    events = load_csv(str(artifacts_dir / "events.csv"))
    detections = load_csv(str(artifacts_dir / "detections.csv"))
    metrics = load_json(str(artifacts_dir / "metrics.json"))
    bandwidth = load_json(str(artifacts_dir / "bandwidth_report.json"))

    if events.empty and detections.empty:
        st.info("No processed events yet. Run `python orchestrator.py` and refresh this page.")
        st.stop()

    if not events.empty:
        available_classes = sorted(events["class"].dropna().astype(str).unique())
        selected_classes = st.sidebar.multiselect(
            "Event classes", available_classes, default=available_classes
        )
        minimum_confidence = st.sidebar.slider("Minimum event confidence", 0.0, 1.0, 0.25, 0.05)
        filtered_events = events[
            events["class"].astype(str).isin(selected_classes)
            & (events["confidence"] >= minimum_confidence)
        ].copy()
    else:
        filtered_events = events

    input_bytes = int(metrics.get("input_video_bytes", 0))
    payload_bytes = int(bandwidth.get("edge_payload_bytes", 0))
    reduction = float(bandwidth.get("reduction_percent", 0.0))

    metric_columns = st.columns(5)
    metric_columns[0].metric("Confirmed events", len(filtered_events))
    metric_columns[1].metric("Raw detections", int(metrics.get("detections", len(detections))))
    metric_columns[2].metric("Edge FPS", f"{float(metrics.get('end_to_end_fps', 0)):.1f}")
    metric_columns[3].metric("P95 inference", f"{float(metrics.get('p95_inference_ms', 0)):.0f} ms")
    metric_columns[4].metric("Bandwidth saved", f"{reduction:.1f}%")

    overview_tab, evidence_tab, health_tab = st.tabs(
        ["Operational map", "Evidence review", "System metrics"]
    )

    with overview_tab:
        st.markdown(
            '<span class="status-pill">● Edge pipeline online</span>',
            unsafe_allow_html=True,
        )
        st_folium(build_event_map(filtered_events), height=560, use_container_width=True)
        if not filtered_events.empty:
            display_columns = [
                column
                for column in (
                    "event_id",
                    "class",
                    "confidence",
                    "video_time_s",
                    "lat",
                    "lon",
                    "observation_count",
                    "status",
                )
                if column in filtered_events.columns
            ]
            st.dataframe(
                filtered_events[display_columns],
                hide_index=True,
                use_container_width=True,
            )

    with evidence_tab:
        if filtered_events.empty:
            st.warning("No events match the selected filters.")
        else:
            selected_id = st.selectbox("Select event", filtered_events["event_id"])
            event = filtered_events[filtered_events["event_id"] == selected_id].iloc[0]
            left, right = st.columns([1.5, 1])
            with left:
                frame_path = resolve_evidence(event.get("evidence_frame"), artifacts_dir)
                if frame_path:
                    st.image(str(frame_path), caption="Context frame", use_container_width=True)
                else:
                    st.info("Context image was not generated for this event.")
            with right:
                crop_path = resolve_evidence(event.get("evidence_crop"), artifacts_dir)
                if crop_path:
                    st.image(str(crop_path), caption="Detection crop", use_container_width=True)
                st.subheader(str(event["class"]).title())
                st.write(f"Confidence: **{float(event['confidence']):.2f}**")
                st.write(f"Video time: **{float(event['video_time_s']):.2f} s**")
                st.write(f"Location: `{float(event['lat']):.6f}, {float(event['lon']):.6f}`")
                st.write(f"Temporal hits: **{int(event['temporal_hits'])}**")
                st.warning("Status: Pending human verification")

    with health_tab:
        left, right = st.columns(2)
        with left:
            st.subheader("Measured processing")
            st.json(metrics)
        with right:
            st.subheader("Measured data transfer")
            if bandwidth:
                chart = pd.DataFrame(
                    {
                        "Mode": ["Full video", "Edge evidence"],
                        "Bytes": [input_bytes, payload_bytes],
                    }
                ).set_index("Mode")
                st.bar_chart(chart)
                st.caption(
                    f"{human_bytes(input_bytes)} raw → "
                    f"{human_bytes(payload_bytes)} evidence payload"
                )
                st.json(bandwidth)
            else:
                st.info("Run `python bandwidth_demo.py` to populate transfer metrics.")


if __name__ == "__main__":
    main()
