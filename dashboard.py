"""Streamlit command centre for reviewed edge-AI event artifacts."""

from __future__ import annotations

import json
from html import escape
from pathlib import Path
from typing import Any

import folium
import pandas as pd
import pydeck as pdk
import streamlit as st
from folium.plugins import HeatMap
from streamlit_folium import st_folium

from urban_intelligence.classes import normalize_class_name
from urban_intelligence.command_center import build_operational_scene, filter_scene_at
from urban_intelligence.demo_jobs import (
    UploadValidationError,
    list_completed_runs,
    load_manifest,
    modules_for_profile,
    preflight_checks,
    prepare_run,
    run_analysis,
)
from urban_intelligence.gps import load_gps_csv
from urban_intelligence.review import load_reviews, review_summary, save_review

st.set_page_config(
    page_title="DrishtiPath Command Centre",
    page_icon="🚌",
    layout="wide",
)

st.markdown(
    """
    <style>
      :root {
        --dp-cyan: #24e0cf;
        --dp-blue: #4f8cff;
        --dp-amber: #ffb547;
        --dp-red: #ff4e59;
        --dp-panel: rgba(12, 27, 45, .86);
        --dp-border: rgba(111, 151, 181, .20);
      }
      .block-container {padding-top: 1rem; padding-bottom: 2.5rem; max-width: 1560px;}
      [data-testid="stAppViewContainer"] {
        background:
          radial-gradient(circle at 78% -8%, rgba(36, 224, 207, .14), transparent 30%),
          radial-gradient(circle at 8% 36%, rgba(79, 140, 255, .09), transparent 25%),
          #07111f;
      }
      [data-testid="stHeader"] {background: rgba(7, 17, 31, .55);}
      [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #091625 0%, #07111f 100%);
        border-right: 1px solid var(--dp-border);
      }
      [data-testid="stMetric"] {
        background: linear-gradient(145deg, rgba(17, 35, 55, .94), rgba(8, 22, 38, .94));
        border: 1px solid var(--dp-border);
        padding: 0.9rem 1rem;
        border-radius: 0.9rem;
        box-shadow: 0 16px 36px rgba(0, 0, 0, .12);
      }
      [data-testid="stMetricValue"] {color: #f5fbff; letter-spacing: -.025em;}
      [data-testid="stTabs"] button {font-weight: 650; letter-spacing: .01em;}
      [data-testid="stSegmentedControl"] button {
        border-color: rgba(111, 151, 181, .28); font-weight: 700;
      }
      [data-testid="stSegmentedControl"] button[aria-pressed="true"] {
        color: #06141f; background: linear-gradient(90deg, #24e0cf, #62c6ff);
      }
      button:focus-visible, [tabindex="0"]:focus-visible {
        outline: 2px solid #7ef0e5 !important; outline-offset: 2px;
      }
      [data-testid="stFileUploaderDropzone"] {
        border: 1px dashed rgba(36, 224, 207, .55);
        background: rgba(14, 35, 52, .65);
        border-radius: 1rem;
      }
      .status-pill {
        display: inline-block; padding: .25rem .65rem; border-radius: 999px;
        background: #163b31; color: #7ef0bd; font-size: .82rem;
      }
      .mission-card {
        padding: 1rem 1.1rem; border: 1px solid #26364a; border-radius: .9rem;
        background: linear-gradient(135deg, rgba(16, 28, 44, .96), rgba(8, 20, 35, .96));
        min-height: 98px;
        transition: transform .18s ease, border-color .18s ease, box-shadow .18s ease;
      }
      .mission-card:hover {
        transform: translateY(-3px); border-color: rgba(36, 224, 207, .55);
        box-shadow: 0 16px 38px rgba(0, 0, 0, .24);
      }
      .eyebrow {color: #62e6da; font-size: .75rem; letter-spacing: .12em; font-weight: 700;}
      .muted {color: #91a4b7; font-size: .88rem;}
      .command-hero {
        position: relative; overflow: hidden; padding: 1.35rem 1.5rem; margin: .15rem 0 1rem;
        border: 1px solid rgba(36, 224, 207, .24); border-radius: 1.1rem;
        background:
          linear-gradient(112deg, rgba(13, 32, 50, .98), rgba(8, 22, 38, .92)),
          radial-gradient(circle at 80% 20%, rgba(36, 224, 207, .22), transparent 36%);
        box-shadow: 0 24px 60px rgba(0, 0, 0, .22);
      }
      .command-hero::after {
        content: ""; position: absolute; inset: 0; pointer-events: none; opacity: .18;
        background-image:
          linear-gradient(rgba(93, 149, 183, .17) 1px, transparent 1px),
          linear-gradient(90deg, rgba(93, 149, 183, .17) 1px, transparent 1px);
        background-size: 34px 34px;
        mask-image: linear-gradient(90deg, transparent 25%, black 100%);
        animation: dp-grid-drift 12s linear infinite;
      }
      .hero-title {font-size: 2rem; line-height: 1.08; font-weight: 760; margin: .28rem 0 .45rem;}
      .hero-meta {color: #a7bac9; font-size: .9rem;}
      .live-dot {
        display: inline-block; width: .55rem; height: .55rem; margin-right: .4rem;
        border-radius: 50%; background: var(--dp-cyan); box-shadow: 0 0 0 rgba(36,224,207,.6);
        animation: dp-pulse 1.8s infinite;
      }
      @keyframes dp-pulse {
        0% {box-shadow: 0 0 0 0 rgba(36,224,207,.55)}
        70% {box-shadow: 0 0 0 9px rgba(36,224,207,0)}
        100% {box-shadow: 0 0 0 0 rgba(36,224,207,0)}
      }
      @keyframes dp-grid-drift {
        from {background-position: 0 0, 0 0}
        to {background-position: 68px 34px, 34px 68px}
      }
      .quality-ribbon {
        display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: .7rem;
        margin: .75rem 0 1rem;
      }
      .quality-chip {
        padding: .8rem .9rem; border-radius: .85rem;
        border: 1px solid rgba(111, 151, 181, .22);
        background: linear-gradient(145deg, rgba(16, 35, 54, .88), rgba(8, 22, 38, .88));
      }
      .quality-chip strong {display:block; color:#eafaff; margin-bottom:.18rem;}
      @media (max-width: 800px) {.quality-ribbon {grid-template-columns: 1fr;}}
      .intel-panel {
        padding: 1rem 1.05rem; border-radius: 1rem; min-height: 118px;
        border: 1px solid var(--dp-border); background: var(--dp-panel);
      }
      .legend-row {
        display: flex; align-items: center; gap: .55rem; margin: .58rem 0; color: #b8cad8;
      }
      .legend-swatch {width: .7rem; height: .7rem; border-radius: 50%; display: inline-block;}
      .timeline-card {
        border-left: 2px solid var(--dp-cyan); padding: .65rem .85rem; margin: .42rem 0;
        background: rgba(14, 31, 48, .72); border-radius: 0 .7rem .7rem 0;
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
def load_json(path: str) -> Any:
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


def dataframe_records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    if frame.empty:
        return []
    return frame.where(pd.notnull(frame), None).to_dict(orient="records")


@st.cache_data(ttl=2)
def load_route_records(path: str) -> list[dict[str, float]]:
    gps_path = Path(path)
    if not gps_path.is_file():
        return []
    try:
        track = load_gps_csv(gps_path)
    except (OSError, ValueError):
        return []
    return [
        {
            "timestamp_s": point.timestamp_s,
            "latitude": point.latitude,
            "longitude": point.longitude,
        }
        for point in track.points
    ]


def build_command_deck(scene: dict[str, Any]) -> pdk.Deck:
    """Build the online, tiled 3D operational map."""
    layers: list[pdk.Layer] = []
    route = scene["route"]
    if len(route) >= 2:
        layers.append(
            pdk.Layer(
                "PathLayer",
                data=[{"path": route}],
                get_path="path",
                get_color=[36, 224, 207, 210],
                width_min_pixels=4,
                get_width=5,
                joint_rounded=True,
                cap_rounded=True,
                pickable=False,
            )
        )

    for name, radius in (("traffic", 18), ("hazards", 9)):
        if not scene[name]:
            continue
        layers.append(
            pdk.Layer(
                "ColumnLayer",
                data=scene[name],
                get_position="position",
                get_fill_color="color",
                get_elevation="elevation",
                radius=radius,
                disk_resolution=12,
                elevation_scale=1,
                extruded=True,
                pickable=True,
                auto_highlight=True,
            )
        )

    for name in ("bottlenecks", "anpr"):
        if not scene[name]:
            continue
        layers.append(
            pdk.Layer(
                "ScatterplotLayer",
                data=scene[name],
                get_position="position",
                get_fill_color="color",
                get_line_color=[245, 251, 255, 235],
                get_radius="radius",
                radius_min_pixels=8,
                radius_max_pixels=22,
                line_width_min_pixels=2,
                stroked=True,
                filled=True,
                pickable=True,
                auto_highlight=True,
            )
        )

    center = scene["center"]
    return pdk.Deck(
        layers=layers,
        initial_view_state=pdk.ViewState(
            latitude=center["latitude"],
            longitude=center["longitude"],
            zoom=center["zoom"],
            pitch=58,
            bearing=-24,
        ),
        map_style="https://basemaps.cartocdn.com/gl/dark-matter-gl-style/style.json",
        tooltip={"text": "{kind}\n{label}\nStatus: {status}"},
    )


def build_offline_figure(scene: dict[str, Any]):
    """Build a tile-free Plotly 3D scene for unreliable venue internet."""
    import plotly.graph_objects as go

    figure = go.Figure()
    route = scene["route"]
    if route:
        figure.add_trace(
            go.Scatter3d(
                x=[position[0] for position in route],
                y=[position[1] for position in route],
                z=[0] * len(route),
                mode="lines",
                line={"color": "#24E0CF", "width": 7},
                name="Bus route",
                hoverinfo="name",
            )
        )

    layer_styles = {
        "traffic": ("Traffic density", "#1ADCC6", 5),
        "hazards": ("Road hazards", "#FF6A57", 8),
        "bottlenecks": ("Bottlenecks", "#FF3148", 11),
        "anpr": ("ANPR evidence", "#469BFF", 10),
    }
    for layer_name, (display_name, color, marker_size) in layer_styles.items():
        points = scene[layer_name]
        if not points:
            continue
        line_x: list[float | None] = []
        line_y: list[float | None] = []
        line_z: list[float | None] = []
        for point in points:
            longitude, latitude = point["position"]
            line_x.extend([longitude, longitude, None])
            line_y.extend([latitude, latitude, None])
            line_z.extend([0, point["elevation"], None])
        figure.add_trace(
            go.Scatter3d(
                x=line_x,
                y=line_y,
                z=line_z,
                mode="lines",
                line={"color": color, "width": 6},
                hoverinfo="skip",
                showlegend=False,
            )
        )
        figure.add_trace(
            go.Scatter3d(
                x=[point["position"][0] for point in points],
                y=[point["position"][1] for point in points],
                z=[point["elevation"] for point in points],
                mode="markers",
                marker={"color": color, "size": marker_size, "opacity": 0.92},
                text=[f"{point['label']}<br>{point['status']}" for point in points],
                hovertemplate="%{text}<extra></extra>",
                name=display_name,
            )
        )

    figure.update_layout(
        height=590,
        margin={"l": 0, "r": 0, "t": 8, "b": 0},
        paper_bgcolor="#081625",
        plot_bgcolor="#081625",
        font={"color": "#DCECF4"},
        legend={"orientation": "h", "y": 0.98, "x": 0.02, "bgcolor": "rgba(0,0,0,0)"},
        scene={
            "bgcolor": "#081625",
            "camera": {"eye": {"x": 1.55, "y": -1.55, "z": 1.12}},
            "aspectmode": "auto",
            "xaxis": {
                "title": "Longitude",
                "gridcolor": "rgba(82,132,163,.22)",
                "showbackground": False,
            },
            "yaxis": {
                "title": "Latitude",
                "gridcolor": "rgba(82,132,163,.22)",
                "showbackground": False,
            },
            "zaxis": {
                "title": "Intensity",
                "gridcolor": "rgba(82,132,163,.22)",
                "showbackground": False,
            },
        },
    )
    return figure


def load_operational_scene(
    *,
    active_run: Path,
    road_dir: Path,
    traffic_dir: Path,
    anpr_dir: Path,
    assets_dir: Path | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    road_events = load_csv(str(road_dir / "events.csv"))
    if assets_dir is not None:
        asset_value = load_json(str(assets_dir / "asset_events.json"))
        asset_events = asset_value if isinstance(asset_value, list) else []
        if asset_events:
            road_events = pd.concat(
                [road_events, pd.DataFrame(asset_events)], ignore_index=True
            )
    traffic_windows = load_csv(str(traffic_dir / "traffic_timeseries.csv"))
    bottlenecks = load_csv(str(traffic_dir / "bottleneck_events.csv"))
    anpr_value = load_json(str(anpr_dir / "anpr_events.json"))
    anpr_events = anpr_value if isinstance(anpr_value, list) else []
    traffic_summary = load_json(str(traffic_dir / "traffic_summary.json"))
    route_records = load_route_records(str(active_run / "input" / "gps.csv"))
    scene = build_operational_scene(
        route_records=route_records,
        road_events=dataframe_records(road_events),
        traffic_windows=dataframe_records(traffic_windows),
        bottleneck_events=dataframe_records(bottlenecks),
        anpr_events=anpr_events,
    )
    return scene, traffic_summary if isinstance(traffic_summary, dict) else {}


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
        "longitudinal_crack": "darkred",
        "transverse_crack": "orange",
        "alligator_crack": "pink",
        "person": "orange",
        "truck": "darkblue",
        "car": "blue",
    }
    for _, row in events.iterrows():
        class_name = normalize_class_name(str(row["class"]))
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


def build_bottleneck_map(events: pd.DataFrame, gps_source_type: str) -> folium.Map:
    center = [28.6139, 77.2090]
    lat_col = "latitude" if "latitude" in events.columns else "lat"
    lon_col = "longitude" if "longitude" in events.columns else "lon"
    if not events.empty:
        center = [float(events[lat_col].mean()), float(events[lon_col].mean())]
    event_map = folium.Map(location=center, zoom_start=15, control_scale=True)
    for _, row in events.iterrows():
        popup = (
            f"<b>{row.get('event_id', 'bottleneck')}</b><br>"
            f"Vehicles: {float(row.get('mean_vehicle_count', 0)):.1f}<br>"
            f"Occupancy: {float(row.get('mean_occupancy', 0)):.2f}<br>"
            f"Status: {row.get('status', 'pending_review')}<br>"
            f"GPS type: {gps_source_type}"
        )
        folium.Marker(
            [float(row[lat_col]), float(row[lon_col])],
            popup=popup,
            icon=folium.Icon(color="red", icon="warning-sign"),
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


def render_review_controls(*, module: str, event_id: str, run_root: Path) -> None:
    """Collect a durable human decision without changing model-produced evidence."""
    path = run_root / "review_feedback.json"
    reviews = load_reviews(path)
    review_key = f"{module}:{event_id}"
    existing = reviews.get(review_key, {})
    label_to_value = {
        "Confirm finding": "confirmed",
        "Reject false positive": "rejected_false_positive",
        "Needs field inspection": "needs_field_inspection",
    }
    value_to_label = {value: label for label, value in label_to_value.items()}
    current_label = value_to_label.get(
        str(existing.get("decision", "needs_field_inspection")),
        "Needs field inspection",
    )
    st.markdown("#### Human review")
    decision_label = st.segmented_control(
        "Decision",
        list(label_to_value),
        default=current_label,
        key=f"review-decision-{module}-{event_id}",
    )
    note = st.text_input(
        "Reviewer note",
        value=str(existing.get("note", "")),
        max_chars=500,
        key=f"review-note-{module}-{event_id}",
        placeholder="Example: zebra paint, real pothole, inspect on route…",
    )
    if st.button(
        "Save review decision",
        key=f"review-save-{module}-{event_id}",
        width="stretch",
    ):
        record = save_review(
            path,
            module=module,
            event_id=event_id,
            decision=label_to_value[decision_label or current_label],
            note=note,
        )
        st.success(
            f"Saved · {str(record['decision']).replace('_', ' ').title()} · "
            f"active-learning label: {record['active_learning_label']}"
        )

    if path.is_file():
        summary = review_summary(load_reviews(path))
        st.caption(
            f"Mission review ledger · {summary['confirmed']} confirmed · "
            f"{summary['rejected_false_positive']} false positives · "
            f"{summary['needs_field_inspection']} field checks"
        )
        st.download_button(
            "Download review feedback",
            data=path.read_bytes(),
            file_name="drishtipath-review-feedback.json",
            mime="application/json",
            key=f"review-download-{module}-{event_id}",
            width="stretch",
        )


def render_road_hazard_section(
    *,
    artifacts_dir: Path,
    events: pd.DataFrame,
    detections: pd.DataFrame,
    metrics: dict,
    bandwidth: dict,
) -> None:
    if events.empty and detections.empty:
        if metrics:
            st.warning(
                "The scan completed but no road-hazard observation passed the quality gates. "
                "This does not prove the road is defect-free. Re-run High recall and add "
                "missed frames to the India-specific fine-tuning set."
            )
            st.json(metrics)
        else:
            st.info(
                "No road-hazard artifacts in this directory. Run a new scan or point the "
                "engineering artifact path at an existing mission."
            )
        return

    if not events.empty:
        events = events.copy()
        events["class"] = events["class"].astype(str).map(normalize_class_name)
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
    bandwidth_label = (
        f"{reduction:.1f}%" if bandwidth.get("measurement_valid", False) else "Not measured"
    )

    metric_columns = st.columns(6)
    metric_columns[0].metric("Confirmed hazards", len(filtered_events))
    metric_columns[1].metric(
        "Model proposals",
        int(metrics.get("model_proposals", metrics.get("detections", len(detections)))),
    )
    metric_columns[2].metric(
        "Quality-eligible",
        int(metrics.get("detections", len(detections))),
    )
    metric_columns[3].metric("Edge FPS", f"{float(metrics.get('end_to_end_fps', 0)):.1f}")
    metric_columns[4].metric(
        "P95 inference",
        f"{float(metrics.get('p95_inference_ms', 0)):.0f} ms",
    )
    metric_columns[5].metric("Bandwidth saved", bandwidth_label)

    profile = str(metrics.get("quality_profile", "custom")).replace("_", " ").title()
    st.caption(
        f"Quality policy: {profile} · Only temporally confirmed events enter the map. "
        "Model proposals are diagnostics, not verified hazards."
    )
    if len(filtered_events) <= 1 and int(metrics.get("processed_frames", 0)) >= 300:
        st.warning(
            "Low hazard yield for a long clip. Treat this as a possible recall warning—not a "
            "clean-road result. Review the annotated video and export missed examples for "
            "fine-tuning."
        )

    overview_tab, evidence_tab, health_tab = st.tabs(
        ["Operational map", "Evidence review", "System metrics"]
    )

    with overview_tab:
        st.markdown(
            '<span class="status-pill">● Edge pipeline online</span>',
            unsafe_allow_html=True,
        )
        st_folium(build_event_map(filtered_events), height=560, width=None)
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
                width="stretch",
            )
        annotated_path = artifacts_dir / "annotated.mp4"
        if annotated_path.is_file():
            with st.expander("Play annotated road-hazard video"):
                st.video(str(annotated_path))

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
                    st.image(str(frame_path), caption="Context frame", width="stretch")
                else:
                    st.info("Context image was not generated for this event.")
            with right:
                crop_path = resolve_evidence(event.get("evidence_crop"), artifacts_dir)
                if crop_path:
                    st.image(str(crop_path), caption="Detection crop", width="stretch")
                st.subheader(str(event["class"]).title())
                st.write(f"Confidence: **{float(event['confidence']):.2f}**")
                st.write(f"Video time: **{float(event['video_time_s']):.2f} s**")
                st.write(f"Location: `{float(event['lat']):.6f}, {float(event['lon']):.6f}`")
                st.write(f"Temporal hits: **{int(event['temporal_hits'])}**")
                st.warning("Status: Pending human verification")
                render_review_controls(
                    module="road",
                    event_id=str(event["event_id"]),
                    run_root=artifacts_dir.parent,
                )

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


def render_traffic_section(artifacts_dir: Path) -> None:
    summary = load_json(str(artifacts_dir / "traffic_summary.json"))
    timeseries = load_csv(str(artifacts_dir / "traffic_timeseries.csv"))
    bottlenecks = load_csv(str(artifacts_dir / "bottleneck_events.csv"))
    safety_value = load_json(str(artifacts_dir / "safety_events.json"))
    safety_events = safety_value if isinstance(safety_value, list) else []

    if not summary and timeseries.empty and bottlenecks.empty:
        st.info(
            "No traffic analytics artifacts found. Run:\n\n"
            "```bash\n"
            "python traffic_analytics.py \\\n"
            "  --input clips/input_video.mp4 \\\n"
            "  --gps gps_data.csv \\\n"
            "  --gps-source-type synthetic_demo \\\n"
            "  --output-dir artifacts/traffic_input_video\n"
            "```\n\n"
            "Then point the sidebar at that output directory."
        )
        return

    gps_source_type = str(summary.get("gps_source_type", "unknown"))
    if gps_source_type == "synthetic_demo":
        st.warning(
            "GPS provenance is labelled **synthetic_demo**. Coordinates are demo "
            "interpolation, not claimed real bus telemetry."
        )

    totals = summary.get("totals_by_class", {})
    row_one = st.columns(3)
    row_one[0].metric(
        "Unique tracked vehicles", int(summary.get("total_unique_tracked_vehicles", 0))
    )
    row_one[1].metric(
        "Current ROI vehicles", int(summary.get("current_vehicle_count", 0))
    )
    row_one[2].metric(
        "Peak ROI vehicles", int(summary.get("max_vehicle_count", 0))
    )
    row_two = st.columns(3)
    row_two[0].metric(
        "Mean occupancy",
        f"{float(summary.get('average_occupancy', 0)) * 100:.1f}%",
        help="Image-space proxy across processed frames",
    )
    row_two[1].metric(
        "Peak occupancy", f"{float(summary.get('max_occupancy', 0)) * 100:.1f}%"
    )
    row_two[2].metric(
        "Bottleneck events", int(summary.get("bottleneck_event_count", len(bottlenecks)))
    )

    st.caption(
        "ROI occupancy is an image-space proxy. Bottleneck detection is a configurable "
        "prototype heuristic (`configurable_roi_heuristic`), not a calibrated municipal standard. "
        "COCO does not identify school children."
    )

    annotated_path = artifacts_dir / "traffic_annotated.mp4"
    if annotated_path.is_file():
        with st.expander("Play annotated traffic video"):
            st.video(str(annotated_path))

    if totals:
        class_df = pd.DataFrame(
            {"class": list(totals.keys()), "unique_vehicles": list(totals.values())}
        ).set_index("class")
        st.subheader("Unique vehicles by class")
        st.bar_chart(class_df)

    left, right = st.columns(2)
    with left:
        st.subheader("Vehicle count timeline")
        if not timeseries.empty and "mean_vehicle_count" in timeseries.columns:
            chart = timeseries.set_index("midpoint_time_s")[["mean_vehicle_count"]]
            st.line_chart(chart)
        else:
            st.info("No traffic timeseries available.")
    with right:
        st.subheader("ROI occupancy timeline")
        if not timeseries.empty and "mean_occupancy" in timeseries.columns:
            chart = timeseries.set_index("midpoint_time_s")[["mean_occupancy"]]
            st.line_chart(chart)
        else:
            st.info("No occupancy timeseries available.")

    st.subheader("Bottleneck events")
    if bottlenecks.empty:
        st.info("No bottleneck events were emitted for these settings.")
    else:
        st.dataframe(bottlenecks, hide_index=True, width="stretch")
        st_folium(
            build_bottleneck_map(bottlenecks, gps_source_type),
            height=420,
            width=None,
        )

    with st.expander("Traffic summary JSON"):
        st.json(summary)

    st.subheader("Vulnerable-road-user & driving review")
    st.caption(
        "Candidates reuse tracked traffic detections. School-child identity, legal speed, "
        "collision certainty, and driver intent are never inferred automatically."
    )
    if safety_events:
        safe_frame = pd.DataFrame(safety_events)
        visible = [
            column
            for column in (
                "event_id",
                "event_type",
                "subtype",
                "video_time_s",
                "severity",
                "school_zone_context",
                "status",
            )
            if column in safe_frame.columns
        ]
        st.dataframe(safe_frame[visible], hide_index=True, width="stretch")
    else:
        st.info("No safety candidate crossed the configured image-space review gate.")


def render_asset_section(artifacts_dir: Path) -> None:
    metrics = load_json(str(artifacts_dir / "metrics.json"))
    value = load_json(str(artifacts_dir / "asset_events.json"))
    events = value if isinstance(value, list) else []
    detections = load_csv(str(artifacts_dir / "asset_detections.csv"))
    if not metrics and not events and detections.empty:
        st.info("No urban-assets scan is available for this mission.")
        return
    columns = st.columns(4)
    columns[0].metric("Review events", len(events))
    columns[1].metric("Visual detections", int(metrics.get("visual_detections", 0)))
    columns[2].metric("Inventory assets", int(metrics.get("inventory_assets", 0)))
    columns[3].metric("Edge FPS", f"{float(metrics.get('end_to_end_fps', 0)):.1f}")
    st.warning(
        "A missing-asset result is an inventory-based absence candidate, never a learned "
        "'missing object' box. It requires camera-visible inventory metadata and human review."
    )
    if events:
        frame = pd.DataFrame(events)
        visible = [
            column
            for column in (
                "event_id",
                "event_type",
                "class",
                "confidence",
                "sampled_frames",
                "visual_hits",
                "lat",
                "lon",
                "status",
            )
            if column in frame.columns
        ]
        st.dataframe(frame[visible], hide_index=True, width="stretch")
        map_rows = frame.copy()
        if "confidence" not in map_rows:
            map_rows["confidence"] = map_rows.get("evidence_strength", 0.5)
        st_folium(build_event_map(map_rows), height=460, width=None)
    else:
        st.info("No temporally confirmed asset event requires review.")
    annotated = artifacts_dir / "assets_annotated.mp4"
    if annotated.is_file():
        with st.expander("Play annotated urban-assets video"):
            st.video(str(annotated))


def render_incident_section(run_dir: Path | None) -> None:
    if run_dir is None:
        st.info("Select a completed one-upload mission to review incident correlation.")
        return
    value = load_json(str(run_dir / "incidents" / "incidents.json"))
    incidents = value if isinstance(value, list) else []
    if not incidents:
        st.info("No safety candidate was available for ANPR correlation.")
        return
    frame = pd.DataFrame(incidents)
    forbidden = {"plate", "plate_text", "normalized_plate", "full_plate", "ocr_text"}
    safe_columns = [column for column in frame.columns if column not in forbidden]
    linked = int(
        (frame["anpr_link_status"] == "candidate_pending_review").sum()
        if "anpr_link_status" in frame
        else 0
    )
    columns = st.columns(3)
    columns[0].metric("Incident candidates", len(frame))
    columns[1].metric("Masked ANPR links", linked)
    columns[2].metric("Human review", len(frame))
    st.error(
        "Hit-and-run and rash-driving outputs are investigative candidates—not legal "
        "findings. Plate text remains masked in the command centre."
    )
    st.dataframe(frame[safe_columns], hide_index=True, width="stretch")


def render_fleet_section(export_dir: Path) -> None:
    summary = load_json(str(export_dir / "fleet_summary.json"))
    clusters_value = load_json(str(export_dir / "deficiency_clusters.json"))
    clusters = clusters_value if isinstance(clusters_value, list) else []
    od = load_csv(str(export_dir / "od_matrix.csv"))
    if not summary:
        st.info(
            "No central fleet export found. Start `fleet_server.py`, deliver edge outboxes, "
            "then run `python fleet_export.py`."
        )
        return
    columns = st.columns(6)
    columns[0].metric("Fleet vehicles", int(summary.get("vehicle_count", 0)))
    columns[1].metric("Missions", int(summary.get("mission_count", 0)))
    columns[2].metric("Events", int(summary.get("event_count", 0)))
    columns[3].metric("Deficiency clusters", int(summary.get("deficiency_cluster_count", 0)))
    columns[4].metric("OD pairs", int(summary.get("od_pair_count", 0)))
    columns[5].metric("Review queue", int(summary.get("pending_review", 0)))
    st.caption(
        "Repeated sightings within the configured radius are grouped across buses. "
        "OD rows represent observed bus missions—not inferred passenger journeys."
    )
    left, right = st.columns([1.25, 1])
    with left:
        st.subheader("Infrastructure deficiency map")
        if clusters:
            cluster_frame = pd.DataFrame(clusters)
            st.map(cluster_frame, latitude="latitude", longitude="longitude")
            visible = [
                column
                for column in (
                    "cluster_id",
                    "class",
                    "sightings",
                    "unique_vehicles",
                    "latitude",
                    "longitude",
                )
                if column in cluster_frame.columns
            ]
            st.dataframe(cluster_frame[visible], hide_index=True, width="stretch")
        else:
            st.info("No deficiency cluster is available.")
    with right:
        st.subheader("Observed mission OD matrix")
        if not od.empty:
            st.dataframe(od, hide_index=True, width="stretch")
        else:
            st.info("No origin–destination pair is available.")
        with st.expander("Fleet summary JSON"):
            st.json(summary)


def render_edge_benchmark_section(report_path: Path) -> None:
    report = load_json(str(report_path))
    if not report:
        st.info(
            "No edge benchmark report found. Run:\n\n"
            "```bash\n"
            "python optimize_model.py \\\n"
            "  --model models/road_hazards.pt \\\n"
            "  --format onnx \\\n"
            "  --precision fp32 \\\n"
            "  --source clips/1.mp4 \\\n"
            "  --device cpu \\\n"
            "  --sampling uniform \\\n"
            "  --warmup-runs 5 \\\n"
            "  --benchmark-frames 50 \\\n"
            "  --report artifacts/edge_bench/road_hazards_onnx_fp32.json\n"
            "```"
        )
        return

    if not report.get("raspberry_pi_benchmarked", False):
        hardware_scope = report.get("runtime_provenance", {}).get(
            "hardware_scope", "current_machine_only"
        )
        st.warning(
            "This report was generated on the current machine only "
            f"(`hardware_scope: {hardware_scope}`). "
            "Do not present it as Raspberry Pi performance."
        )

    source = report.get("source_model", {})
    exported = report.get("exported_model", {})
    comparison = report.get("artifact_comparison", {})
    source_bench = report.get("source_benchmark", {})
    exported_bench = report.get("exported_benchmark", {})
    parity = report.get("prediction_parity", {})
    validation = report.get("validation", {"status": "not_run"})
    config = report.get("benchmark_configuration", {})

    change_label = comparison.get("change_label", "size_reduction")
    change_percent = float(comparison.get("change_percent", 0))
    size_metric_label = (
        "Size increase" if change_label == "size_increase" else "Size reduction"
    )

    row_one = st.columns(3)
    row_one[0].metric("Source size (MiB)", f"{float(source.get('mib', 0)):.2f}")
    row_one[1].metric("Exported size (MiB)", f"{float(exported.get('mib', 0)):.2f}")
    row_one[2].metric(size_metric_label, f"{change_percent:.1f}%")

    row_two = st.columns(3)
    row_two[0].metric(
        "Source wall P95 (ms)", f"{float(source_bench.get('wall_p95_ms', 0)):.1f}"
    )
    row_two[1].metric(
        "Exported wall P95 (ms)", f"{float(exported_bench.get('wall_p95_ms', 0)):.1f}"
    )
    row_two[2].metric(
        "Exported measured FPS",
        f"{float(exported_bench.get('measured_end_to_end_fps', 0)):.1f}",
    )

    st.caption(
        "Measured FPS = timed samples / total wall time. "
        f"Median-derived theoretical FPS (not measured throughput): "
        f"{float(exported_bench.get('fps_from_median_wall_ms', 0)):.1f}."
    )
    st.caption(
        f"Sampling: {config.get('sampling', 'unknown')} · "
        f"unique frames: {config.get('unique_frames_loaded', 0)} / "
        f"{config.get('source_total_frames', 0)}"
    )

    st.subheader("Prediction parity")
    st.caption(
        parity.get(
            "note",
            "Prediction parity is not validation accuracy. "
            "A single matched detection is insufficient evidence.",
        )
    )
    if int(parity.get("matched_detections", 0)) <= 1:
        st.warning(
            "Matched detections are too few for strong parity evidence. "
            "This is not validation accuracy."
        )
    parity_cols = st.columns(4)
    parity_cols[0].metric("Frames compared", int(parity.get("frames_compared", 0)))
    parity_cols[1].metric("Matched detections", int(parity.get("matched_detections", 0)))
    parity_cols[2].metric(
        "Source match recall", f"{float(parity.get('source_match_recall', 0)):.2f}"
    )
    parity_cols[3].metric(
        "Exported match precision",
        f"{float(parity.get('exported_match_precision', 0)):.2f}",
    )

    st.subheader("Validation")
    if validation.get("status") == "not_run":
        st.info("Validation was not run for this report.")
    else:
        st.json(validation)

    runtime = report.get("runtime_provenance", {})
    st.caption(
        f"Runtime: {runtime.get('operating_system', 'unknown')} · "
        f"{runtime.get('machine_architecture', 'unknown')} · "
        f"Ultralytics {runtime.get('ultralytics_version', 'unknown')} · "
        f"device {runtime.get('requested_inference_device', 'unknown')}"
    )

    with st.expander("Edge benchmark report JSON"):
        st.json(report)


def render_live_edge_section(mission_dir: Path) -> None:
    """Render privacy-safe status emitted by ``edge_agent.py``."""
    st.subheader("Live edge mission")
    st.caption(
        "Capture and analytics are measured separately. The dashcam may capture at 30 FPS "
        "while mixed-rate edge models analyze selected fresh frames without building a backlog."
    )
    status = load_json(str(mission_dir / "device_status.json"))
    report = load_json(str(mission_dir / "metrics.json"))
    if not isinstance(status, dict) or not status:
        st.info(
            "No live edge status found. Start the agent on a Raspberry Pi or use a recorded "
            "route replay, then point the sidebar to its mission directory."
        )
        st.code(
            "python edge_agent.py --source 0 --fixed-gps 28.6139,77.2090 "
            "--gps-source-type synthetic_demo --profile pi4 "
            "--output-dir artifacts/edge_live/latest",
            language="bash",
        )
        return

    capture = status.get("capture", {})
    schedules = status.get("scheduler", {}).get("schedules", {})
    temperature = status.get("cpu_temperature_c")
    available_memory = status.get("memory_available_mb")
    device_name = status.get("device_model") or (
        f"{status.get('platform', 'Unknown')} · {status.get('machine', 'unknown')}"
    )
    pi_label = "RASPBERRY PI VERIFIED" if status.get("raspberry_pi") else "NON-PI RUNTIME"
    mission_status = str(status.get("mission_status", "running")).replace("_", " ").upper()

    st.markdown(
        "<div class='mission-card'>"
        f"<div class='eyebrow'><span class='live-dot'></span>{escape(mission_status)}</div>"
        f"<div style='font-size:1.25rem;font-weight:720;margin:.3rem 0'>"
        f"{escape(str(device_name))}</div>"
        f"<div class='muted'>{escape(pi_label)} · privacy-safe device telemetry</div>"
        "</div>",
        unsafe_allow_html=True,
    )

    captured = int(capture.get("captured_frames", 0))
    dropped = int(capture.get("dropped_analysis_frames", 0))
    drop_rate = dropped / captured * 100 if captured else 0.0
    analytics_attempts = int(report.get("analytics_attempts", 0)) if report else sum(
        int(values.get("attempted", 0)) for values in schedules.values()
    )
    columns = st.columns(6)
    columns[0].metric("Captured frames", captured)
    columns[1].metric("Analytics runs", analytics_attempts)
    columns[2].metric("Dropped for freshness", dropped, f"{drop_rate:.1f}%")
    columns[3].metric(
        "CPU temperature",
        "Unavailable" if temperature is None else f"{float(temperature):.1f} °C",
    )
    columns[4].metric(
        "Available memory",
        "Unavailable" if available_memory is None else f"{float(available_memory):.0f} MB",
    )
    columns[5].metric("Evidence queued", int(report.get("pending_evidence_packets", 0)))

    if schedules:
        rows = []
        for name, values in schedules.items():
            rows.append(
                {
                    "Capability": str(name).replace("_", " ").title(),
                    "Activation": values.get("activation", "unknown"),
                    "Target FPS": values.get("target_fps", 0),
                    "Attempts": values.get("attempted", 0),
                    "Succeeded": values.get("succeeded", 0),
                    "Mean latency (ms)": values.get("mean_latency_ms", 0),
                    "Max latency (ms)": values.get("max_latency_ms", 0),
                    "Last error": values.get("last_error", ""),
                }
            )
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")

    budget = report.get("compute_budget", {}) if isinstance(report, dict) else {}
    if budget:
        utilization = float(budget.get("estimated_utilization", 0))
        if budget.get("overloaded"):
            st.error(
                f"Measured schedule demand is {utilization * 100:.1f}% of one inference "
                "worker. Reduce model rates or resolution before field deployment."
            )
        else:
            st.success(
                f"Measured schedule demand is {utilization * 100:.1f}% of one inference "
                "worker. This is a device-specific measurement, not a general Pi claim."
            )
    if status.get("active_geofences"):
        st.write("Active geofences:", ", ".join(status["active_geofences"]))
    with st.expander("Privacy-safe device status JSON"):
        st.json(status)


def render_anpr_section(artifacts_dir: Path, stage: dict[str, Any] | None = None) -> None:
    events_value = load_json(str(artifacts_dir / "anpr_events.json"))
    metrics = load_json(str(artifacts_dir / "metrics.json"))
    events = events_value if isinstance(events_value, list) else []

    if stage and stage.get("status") == "failed":
        st.error(f"ANPR stage failed: {stage.get('message', 'Unknown error')}")
        return
    if not metrics and not events:
        st.info("No ANPR output is available for this run.")
        return

    gps_source_type = str(metrics.get("gps_source_type", "unknown"))
    if gps_source_type == "synthetic_demo":
        st.warning("ANPR coordinates use synthetic demo GPS, not real bus telemetry.")

    columns = st.columns(5)
    columns[0].metric("Sampled frames", int(metrics.get("sampled_frames", 0)))
    columns[1].metric(
        "Plate-like proposals",
        int(metrics.get("plate_like_proposals", metrics.get("total_detections", 0))),
    )
    columns[2].metric(
        "Quality-eligible",
        int(metrics.get("quality_eligible_observations", metrics.get("total_detections", 0))),
    )
    columns[3].metric(
        "Gate rejected",
        int(metrics.get("geometry_or_confidence_rejections", 0)),
    )
    columns[4].metric("Verified tracks", len(events))

    if not events:
        st.success(
            "No verified plate evidence. Any raw plate-like regions were rejected or failed "
            "the multi-frame detector/OCR consensus gate."
        )
        with st.expander("Why proposals are not number plates"):
            st.write(
                "FastALPR first proposes rectangular regions. DrishtiPath reports a plate only "
                "after size/shape checks, repeated spatial tracking, Indian-format validation, "
                "OCR agreement, detector confidence and vote-ratio thresholds all pass."
            )
            st.json(metrics.get("plate_geometry_gate", {}))
        return

    safe_rows = [
        {
            "event_id": item.get("event_id"),
            "plate": item.get("masked_plate", ""),
            "observations": item.get("observation_count", 0),
            "winning_votes": item.get("winning_ocr_votes", 0),
            "vote_ratio": item.get("winning_vote_ratio", 0),
            "ocr_confidence": item.get("mean_ocr_confidence", 0),
            "detector_confidence": item.get("mean_detector_confidence", 0),
            "status": item.get("status", "pending_review"),
        }
        for item in events
    ]
    st.dataframe(pd.DataFrame(safe_rows), hide_index=True, width="stretch")

    event_ids = [str(item.get("event_id", "")) for item in events]
    selected_id = st.selectbox("Select ANPR evidence", event_ids)
    selected = next(item for item in events if str(item.get("event_id", "")) == selected_id)
    left, right = st.columns([1.5, 1])
    with left:
        frame_path = resolve_evidence(selected.get("evidence_frame"), artifacts_dir)
        if frame_path:
            st.image(str(frame_path), caption="Best observation frame", width="stretch")
        else:
            st.info("No context frame was promoted for this event.")
    with right:
        crop_path = resolve_evidence(selected.get("evidence_crop"), artifacts_dir)
        if crop_path:
            st.image(str(crop_path), caption="Plate crop", width="stretch")
        st.subheader(str(selected.get("masked_plate", "Masked plate")))
        st.write(f"OCR confidence: **{float(selected.get('mean_ocr_confidence', 0)):.3f}**")
        st.write(f"Winning votes: **{int(selected.get('winning_ocr_votes', 0))}**")
        st.write(
            "Location: "
            f"`{float(selected.get('latitude', 0)):.6f}, "
            f"{float(selected.get('longitude', 0)):.6f}`"
        )
        st.warning("Pending authorized human review. Plate text remains masked in this UI.")
        render_review_controls(
            module="anpr",
            event_id=str(selected.get("event_id", "")),
            run_root=artifacts_dir.parent,
        )


def render_run_summary(run_dir: Path) -> dict[str, Any]:
    manifest = load_manifest(run_dir)
    if not manifest:
        st.warning(f"Run manifest is missing: {run_dir}")
        return {}

    status = str(manifest.get("status", "unknown"))
    gps_type = str(manifest.get("gps_source_type", "unknown"))
    requested = manifest.get("requested_modules", [])
    columns = st.columns(4)
    columns[0].metric("Run status", status.replace("_", " ").title())
    columns[1].metric("Modules", len(requested))
    columns[2].metric("Completed", int(manifest.get("successful_stages", 0)))
    columns[3].metric("Failed", int(manifest.get("failed_stages", 0)))
    if gps_type == "synthetic_demo":
        st.warning("SYNTHETIC DEMO GPS · Coordinates are illustrative, not real telemetry.")
    elif gps_type == "real_telemetry":
        st.success("REAL TELEMETRY · Coordinates came from the uploaded GPS CSV.")
    for warning in manifest.get("warnings", []):
        st.warning(str(warning))

    stage_columns = st.columns(max(1, len(requested)))
    for column, module in zip(stage_columns, requested, strict=False):
        stage = manifest.get("stages", {}).get(module, {})
        stage_status = str(stage.get("status", "queued"))
        icon = {"completed": "✅", "failed": "❌", "running": "⏳"}.get(
            stage_status, "○"
        )
        column.markdown(
            f"<div class='mission-card'><div class='eyebrow'>{escape(module.upper())}</div>"
            f"<div>{icon} {escape(stage_status.replace('_', ' ').title())}</div>"
            f"<div class='muted'>{escape(str(stage.get('message', '')))}</div></div>",
            unsafe_allow_html=True,
        )
    return manifest


def render_scene_legend(scene: dict[str, Any], gps_type: str) -> None:
    counts = scene["counts"]
    gps_label = "REAL TELEMETRY" if gps_type == "real_telemetry" else "SYNTHETIC DEMO"
    gps_color = "#7EF0BD" if gps_type == "real_telemetry" else "#FFCF70"
    st.markdown(
        "<div class='intel-panel'>"
        "<div class='eyebrow'>OPERATIONAL LAYERS</div>"
        "<div class='legend-row'><span class='legend-swatch' style='background:#24E0CF'></span>"
        "Bus trajectory</div>"
        f"<div class='legend-row'><span class='legend-swatch' style='background:#FF6A57'></span>"
        f"Road hazards · {counts['hazards']}</div>"
        f"<div class='legend-row'><span class='legend-swatch' style='background:#1ADCC6'></span>"
        f"Traffic windows · {counts['traffic_windows']}</div>"
        f"<div class='legend-row'><span class='legend-swatch' style='background:#FF3148'></span>"
        f"Bottlenecks · {counts['bottlenecks']}</div>"
        f"<div class='legend-row'><span class='legend-swatch' style='background:#469BFF'></span>"
        f"Masked ANPR · {counts['anpr']}</div>"
        "</div>",
        unsafe_allow_html=True,
    )
    st.markdown(
        f"<div class='intel-panel' style='margin-top:.75rem'>"
        f"<div class='eyebrow'>GPS PROVENANCE</div>"
        f"<div style='font-size:1.15rem;font-weight:700;color:{gps_color};margin:.4rem 0'>"
        f"{gps_label}</div>"
        "<div class='muted'>Every visual coordinate preserves its declared source type.</div>"
        "</div>",
        unsafe_allow_html=True,
    )


def render_timeline_review(
    scene: dict[str, Any],
    *,
    road_dir: Path,
    anpr_dir: Path,
) -> None:
    timeline = scene["timeline"]
    st.subheader("Intelligence timeline")
    if not timeline:
        st.info("No confirmed hazard, bottleneck, or ANPR events entered the review queue.")
        return

    left, right = st.columns([1.05, 1.4])
    with left:
        selected_index = st.selectbox(
            "Select evidence",
            range(len(timeline)),
            key="command_timeline_event",
            format_func=lambda index: (
                f"{float(timeline[index]['time_s']):06.1f}s · "
                f"{timeline[index]['title']} · {timeline[index]['detail']}"
            ),
        )
        selected = timeline[selected_index]
        st.markdown(
            "<div class='timeline-card'>"
            f"<div class='eyebrow'>{escape(str(selected['kind']).upper())} · "
            f"{float(selected['time_s']):.1f} SECONDS</div>"
            f"<div style='font-size:1.15rem;font-weight:700;margin:.25rem 0'>"
            f"{escape(str(selected['title']))}</div>"
            f"<div class='muted'>{escape(str(selected['detail']))}</div>"
            f"<div class='muted'>Status · {escape(str(selected['status']))}</div>"
            "</div>",
            unsafe_allow_html=True,
        )
        st.caption(
            "Timeline events are generated by automated quality gates and remain pending "
            "human verification."
        )

    with right:
        source_dir = road_dir if selected["kind"] == "road" else anpr_dir
        frame_path = resolve_evidence(selected.get("evidence_frame"), source_dir)
        crop_path = resolve_evidence(selected.get("evidence_crop"), source_dir)
        if frame_path or crop_path:
            image_columns = st.columns([1.5, 1])
            if frame_path:
                image_columns[0].image(
                    str(frame_path),
                    caption="Context evidence",
                    width="stretch",
                )
            if crop_path:
                image_columns[1].image(
                    str(crop_path),
                    caption="Detection crop",
                    width="stretch",
                )
        else:
            st.info("This event has metadata only; no evidence image was promoted.")


def render_command_center(
    *,
    active_run: Path,
    road_dir: Path,
    traffic_dir: Path,
    anpr_dir: Path,
    assets_dir: Path,
    manifest: dict[str, Any],
) -> None:
    scene, traffic_summary = load_operational_scene(
        active_run=active_run,
        road_dir=road_dir,
        traffic_dir=traffic_dir,
        anpr_dir=anpr_dir,
        assets_dir=assets_dir,
    )
    road_metrics = load_json(str(road_dir / "metrics.json"))
    road_metrics = road_metrics if isinstance(road_metrics, dict) else {}
    gps_type = str(manifest.get("gps_source_type", "unknown"))
    run_status = str(manifest.get("status", "unknown")).replace("_", " ").upper()
    video_name = (
        manifest.get("input", {}).get("video", {}).get("original_name", "Dashcam mission")
    )

    st.markdown(
        "<div class='command-hero'>"
        "<div class='eyebrow'><span class='live-dot'></span>DRISHTIPATH · LIVE MISSION VIEW</div>"
        f"<div class='hero-title'>{escape(str(video_name))}</div>"
        f"<div class='hero-meta'>Run {escape(active_run.name)} · {escape(run_status)} · "
        "Edge-processed intelligence with review-safe evidence</div>"
        "</div>",
        unsafe_allow_html=True,
    )

    replay_times = [
        float(item.get("time_s", 0))
        for item in [*scene.get("route_timed", []), *scene.get("timeline", [])]
    ]
    max_replay_s = max(replay_times, default=0.0)
    replay_left, replay_right = st.columns([3, 1])
    with replay_right:
        replay_enabled = st.toggle(
            "Mission replay",
            value=False,
            help="Scrub through the route and reveal intelligence as it was observed.",
        )
    if replay_enabled and max_replay_s > 0:
        with replay_left:
            replay_cutoff = st.slider(
                "Mission time",
                min_value=0.0,
                max_value=float(max_replay_s),
                value=float(max_replay_s),
                step=max(0.1, float(max_replay_s) / 200),
                format="%.1f s",
            )
        scene = filter_scene_at(scene, replay_cutoff)
        st.caption(
            f"Replay position {replay_cutoff:.1f}s · showing only intelligence available "
            "at that moment"
        )
    else:
        replay_left.caption("Live aggregate view · enable Mission replay to scrub the route")

    unique_vehicles = int(traffic_summary.get("total_unique_tracked_vehicles", 0))
    mean_occupancy = float(traffic_summary.get("average_occupancy", 0)) * 100
    edge_fps = float(
        road_metrics.get(
            "end_to_end_fps",
            traffic_summary.get("processing_fps", 0),
        )
    )
    metric_columns = st.columns(6)
    metric_columns[0].metric("Confirmed hazards", scene["counts"]["hazards"])
    metric_columns[1].metric("Unique vehicles", unique_vehicles)
    metric_columns[2].metric("Bottlenecks", scene["counts"]["bottlenecks"])
    metric_columns[3].metric("ANPR review", scene["counts"]["anpr"])
    metric_columns[4].metric("Edge throughput", f"{edge_fps:.1f} FPS")
    metric_columns[5].metric("Mean occupancy", f"{mean_occupancy:.1f}%")

    map_column, legend_column = st.columns([2.7, 0.9])
    with map_column:
        renderer = st.radio(
            "Operational scene renderer",
            ["Live 3D map", "Offline-safe 3D"],
            horizontal=True,
            help=(
                "Live mode uses external Carto map tiles. Offline-safe mode uses no "
                "external basemap and is recommended for unreliable venue internet."
            ),
        )
        has_scene = bool(scene["route"]) or any(scene["counts"].values())
        if not has_scene:
            st.info("The selected run has no geospatial observations to display.")
        elif renderer == "Offline-safe 3D":
            st.plotly_chart(build_offline_figure(scene), width="stretch")
            st.caption("Tile-free 3D renderer · available without venue internet")
        else:
            try:
                st.pydeck_chart(build_command_deck(scene), width="stretch")
                st.caption(
                    "3D operational intelligence map · height represents observed intensity, "
                    "not physical object height"
                )
            except Exception as exc:
                st.warning(f"Live map unavailable ({exc}). Showing offline-safe 3D.")
                st.plotly_chart(build_offline_figure(scene), width="stretch")

    with legend_column:
        render_scene_legend(scene, gps_type)

    render_timeline_review(scene, road_dir=road_dir, anpr_dir=anpr_dir)

    with st.expander("Pipeline integrity and provenance"):
        render_run_summary(active_run)


def render_scan_launcher(runs_root: Path) -> Path | None:
    st.markdown(
        "<div class='eyebrow'>NEW INTELLIGENCE MISSION</div>",
        unsafe_allow_html=True,
    )
    st.header("Upload once. Analyze everything.")
    st.caption(
        "Drop a dashcam video, choose GPS provenance and run the edge pipelines "
        "without opening a terminal."
    )

    left, right = st.columns([1.25, 1])
    with left:
        video_upload = st.file_uploader(
            "Dashcam video",
            type=["mp4", "mov", "avi", "mkv"],
            help="Maximum 500 MB and 15 minutes for the judge demo workflow.",
        )
        if video_upload is not None:
            with st.expander("Preview uploaded dashcam clip", expanded=True):
                st.video(video_upload.getvalue())
        gps_mode_label = st.radio(
            "GPS source",
            ["Synthetic demo route", "Upload real telemetry"],
            horizontal=True,
        )
        gps_source_type = (
            "synthetic_demo" if gps_mode_label == "Synthetic demo route" else "real_telemetry"
        )
        gps_upload = None
        if gps_source_type == "real_telemetry":
            gps_upload = st.file_uploader(
                "GPS telemetry CSV",
                type=["csv"],
                help="Requires timestamp/timestamp_s plus lat/lon or latitude/longitude.",
            )

    with right:
        profile_label = st.radio(
            "Scan mode",
            ["Quick scan", "Full city scan", "Custom scan"],
            help="Quick: road + traffic. Full: road + traffic + assets + ANPR.",
        )
        profile = {
            "Quick scan": "quick",
            "Full city scan": "full",
            "Custom scan": "custom",
        }[profile_label]
        quality_label = st.segmented_control(
            "Analysis sensitivity",
            ["High recall", "Balanced", "Strict review"],
            default="High recall",
            help=(
                "High recall searches harder for road damage and may produce more review "
                "candidates. Strict review favours precision. ANPR always uses strict gates."
            ),
        )
        quality_profile = {
            "High recall": "high_recall",
            "Balanced": "balanced",
            "Strict review": "strict",
        }[quality_label or "High recall"]
        module_labels = {
            "Road hazards": "road",
            "Traffic analytics": "traffic",
            "Urban assets & waterlogging": "assets",
            "ANPR evidence": "anpr",
        }
        custom_values: tuple[str, ...] = ()
        if profile == "custom":
            selected_labels = st.multiselect(
                "Analysis modules",
                list(module_labels),
                default=["Road hazards", "Traffic analytics"],
            )
            custom_values = tuple(module_labels[label] for label in selected_labels)
        try:
            modules = modules_for_profile(profile, custom_values)
        except ValueError as exc:
            st.error(str(exc))
            modules = ()

        quality_copy = {
            "high_recall": (
                "768 px + augmentation",
                "2 hits in 7 observations",
                "Best for missed potholes",
            ),
            "balanced": (
                "704 px inference",
                "3 hits in 5 observations",
                "Balanced venue demo",
            ),
            "strict": (
                "640 px inference",
                "Higher confidence gate",
                "Best for low false alerts",
            ),
        }[quality_profile]
        st.markdown(
            "<div class='quality-ribbon'>"
            + "".join(
                f"<div class='quality-chip'><strong>{escape(title)}</strong>"
                f"<span class='muted'>{escape(value)}</span></div>"
                for title, value in zip(
                    ("VISION", "TEMPORAL GATE", "MISSION FIT"),
                    quality_copy,
                    strict=True,
                )
            )
            + "</div>",
            unsafe_allow_html=True,
        )

        with st.expander("Model settings", expanded=False):
            road_model = st.text_input("Road-hazard model", "models/road_hazards.pt")
            traffic_model = st.text_input("Traffic model", "yolov8n.pt")
            asset_model = st.text_input("Urban-assets model", "models/urban_assets.pt")
            asset_inventory = st.file_uploader(
                "Optional asset inventory JSON",
                type=["json"],
                help=(
                    "Required only for missing-divider/crossing/sign candidates. "
                    "Waterlogging and visible damage do not require inventory."
                ),
            )
            school_zones = st.file_uploader(
                "Optional school-zone geofences JSON",
                type=["json"],
                help=(
                    "Adds school-zone context to person/vehicle crossing conflicts. "
                    "It never infers a person's age."
                ),
            )

    st.subheader("Preflight")
    checks = (
        preflight_checks(
            modules,
            road_model=road_model,
            traffic_model=traffic_model,
            asset_model=asset_model,
        )
        if modules
        else {}
    )
    check_columns = st.columns(max(1, len(checks)))
    for column, (name, (passed, detail)) in zip(check_columns, checks.items(), strict=False):
        column.markdown(f"{'✅' if passed else '❌'} **{name}**")
        column.caption(detail)

    all_checks_pass = bool(checks) and all(passed for passed, _ in checks.values())
    runnable_module_exists = any(passed for passed, _ in checks.values())
    if checks and not all_checks_pass and runnable_module_exists:
        st.warning(
            "Some optional modules are unavailable. The mission can still run: each stage "
            "is isolated, and successful evidence will be preserved."
        )
    inputs_ready = video_upload is not None and (
        gps_source_type == "synthetic_demo" or gps_upload is not None
    )
    start_scan = st.button(
        "▶ Run DrishtiPath Scan",
        type="primary",
        width="stretch",
        disabled=not (inputs_ready and runnable_module_exists and modules),
    )

    if not start_scan:
        if video_upload is not None:
            st.caption(
                f"Ready: {video_upload.name} · {human_bytes(video_upload.size)} · "
                f"{len(modules)} module(s) selected"
            )
        return None

    progress_bar = st.progress(0, text="Preparing isolated mission workspace")
    stage_text = st.empty()
    mission_status = st.status("DrishtiPath mission starting", expanded=True)

    def update_progress(
        module: str,
        status: str,
        message: str,
        completed: int,
        total: int,
    ) -> None:
        del module
        progress_bar.progress(
            int((completed / max(1, total)) * 100),
            text=message,
        )
        stage_text.caption(message)
        mission_status.write(f"{status.upper()} · {message}")

    try:
        stage_text.caption("Validating upload and GPS data")
        request = prepare_run(
            runs_root=runs_root,
            original_video_name=video_upload.name,
            video_payload=video_upload.getbuffer(),
            gps_source_type=gps_source_type,
            modules=modules,
            scan_profile=profile,
            quality_profile=quality_profile,
            gps_payload=gps_upload.getbuffer() if gps_upload is not None else None,
            original_gps_name=gps_upload.name if gps_upload is not None else "gps.csv",
            road_model=road_model,
            traffic_model=traffic_model,
            asset_model=asset_model,
            asset_inventory_payload=(
                asset_inventory.getbuffer() if asset_inventory is not None else None
            ),
            original_asset_inventory_name=(
                asset_inventory.name if asset_inventory is not None else "assets.json"
            ),
            school_zones_payload=(
                school_zones.getbuffer() if school_zones is not None else None
            ),
            original_school_zones_name=(
                school_zones.name if school_zones is not None else "school_zones.json"
            ),
        )
        manifest = run_analysis(request, progress=update_progress)
    except (UploadValidationError, ValueError, FileNotFoundError, OSError) as exc:
        progress_bar.empty()
        stage_text.empty()
        mission_status.update(label="Mission validation failed", state="error")
        st.error(str(exc))
        return None

    st.session_state["active_run_dir"] = str(request.run_dir)
    progress_bar.progress(100, text="Mission analysis complete")
    if manifest.get("status") == "completed":
        mission_status.update(label="Mission complete", state="complete", expanded=False)
        st.success("Scan complete. Open Mission control or Evidence review to inspect the results.")
    else:
        mission_status.update(
            label="Mission completed with module warnings",
            state="error",
            expanded=True,
        )
        st.warning("Scan finished with one or more failed modules. Successful results were kept.")
    return request.run_dir


def main() -> None:
    st.markdown(
        "<div class='command-hero'>"
        "<div class='eyebrow'><span class='live-dot'></span>DRISHTIPATH · URBAN INTELLIGENCE</div>"
        "<div class='hero-title'>Turn every bus into a moving city sensor.</div>"
        "<div class='hero-meta'>One upload · Edge AI · 3D GIS · Review-safe evidence · "
        "Bandwidth-aware operations</div></div>",
        unsafe_allow_html=True,
    )

    runs_root = Path("artifacts/ui_runs")
    presentation_mode = st.sidebar.toggle(
        "Presentation mode",
        value=True,
        help="Keeps judge-facing workflows focused and hides engineering diagnostics.",
    )
    recent_runs = list_completed_runs(runs_root)
    if recent_runs:
        selected_run = st.sidebar.selectbox(
            "Recent dashboard runs",
            recent_runs,
            format_func=lambda path: path.name,
            index=None,
            placeholder="Select a previous run",
        )
        if selected_run is not None:
            st.session_state["active_run_dir"] = str(selected_run)
    if st.sidebar.button("Clear active run", width="stretch"):
        st.session_state.pop("active_run_dir", None)

    with st.sidebar.expander("Engineering artifact paths", expanded=False):
        artifacts_value = st.text_input("Artifacts directory", "artifacts/latest")
        edge_report_value = st.text_input(
            "Edge benchmark report",
            "artifacts/edge_bench/road_hazards_onnx_fp32.json",
        )
        live_edge_value = st.text_input(
            "Live edge mission",
            "artifacts/edge_live/latest",
        )
        fleet_export_value = st.text_input(
            "Fleet export",
            "artifacts/fleet/export",
        )
    if st.sidebar.button("Refresh data", width="stretch"):
        st.cache_data.clear()
        st.rerun()

    edge_report_path = Path(edge_report_value)
    workspaces = ["New scan", "Mission control", "Evidence review", "Operations"]
    if not presentation_mode:
        workspaces.append("Engineering")
    workspace = st.segmented_control(
        "Command workspace",
        workspaces,
        default="New scan",
        selection_mode="single",
        label_visibility="collapsed",
    )

    if workspace == "New scan":
        new_run = render_scan_launcher(runs_root)
        if new_run is not None:
            st.session_state["active_run_dir"] = str(new_run)

    active_value = st.session_state.get("active_run_dir")
    active_run = Path(active_value) if active_value else None
    road_dir = active_run / "road" if active_run else Path(artifacts_value)
    traffic_dir = active_run / "traffic" if active_run else Path(artifacts_value)
    anpr_dir = active_run / "anpr" if active_run else Path(artifacts_value)
    assets_dir = active_run / "assets" if active_run else Path(artifacts_value)
    manifest = load_manifest(active_run) if active_run else {}

    events = load_csv(str(road_dir / "events.csv"))
    detections = load_csv(str(road_dir / "detections.csv"))
    metrics = load_json(str(road_dir / "metrics.json"))
    bandwidth = load_json(str(road_dir / "bandwidth_report.json"))

    if workspace == "Mission control":
        if active_run:
            render_command_center(
                active_run=active_run,
                road_dir=road_dir,
                traffic_dir=traffic_dir,
                anpr_dir=anpr_dir,
                assets_dir=assets_dir,
                manifest=manifest,
            )
        else:
            st.info("Upload a mission or select a recent run to activate the 3D command centre.")

    elif workspace == "Evidence review":
        evidence_view = st.segmented_control(
            "Evidence layer",
            ["Road hazards", "Urban assets", "ANPR", "Safety & incidents"],
            default="Road hazards",
            selection_mode="single",
        )
        if evidence_view == "Road hazards":
            render_road_hazard_section(
                artifacts_dir=road_dir,
                events=events,
                detections=detections,
                metrics=metrics,
                bandwidth=bandwidth,
            )
        elif evidence_view == "Urban assets":
            render_asset_section(assets_dir)
        elif evidence_view == "ANPR":
            stage = manifest.get("stages", {}).get("anpr") if manifest else None
            render_anpr_section(anpr_dir, stage)
        else:
            render_incident_section(active_run)

    elif workspace == "Operations":
        operation_view = st.segmented_control(
            "Operational layer",
            ["Traffic analytics", "Fleet intelligence"],
            default="Traffic analytics",
            selection_mode="single",
        )
        if operation_view == "Traffic analytics":
            render_traffic_section(traffic_dir)
        else:
            render_fleet_section(Path(fleet_export_value))

    elif workspace == "Engineering":
        engineering_view = st.segmented_control(
            "Engineering view",
            ["Edge benchmark", "Live edge"],
            default="Edge benchmark",
            selection_mode="single",
        )
        if engineering_view == "Edge benchmark":
            render_edge_benchmark_section(edge_report_path)
        else:
            render_live_edge_section(Path(live_edge_value))


if __name__ == "__main__":
    main()
