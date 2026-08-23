"""Streamlit command centre for reviewed edge-AI event artifacts."""

from __future__ import annotations

import json
from html import escape
from pathlib import Path
from typing import Any

import folium
import pandas as pd
import streamlit as st
from folium.plugins import HeatMap
from streamlit_folium import st_folium

from urban_intelligence.classes import normalize_class_name
from urban_intelligence.demo_jobs import (
    UploadValidationError,
    list_completed_runs,
    load_manifest,
    modules_for_profile,
    preflight_checks,
    prepare_run,
    run_analysis,
)

st.set_page_config(
    page_title="DrishtiPath Command Centre",
    page_icon="🚌",
    layout="wide",
)

st.markdown(
    """
    <style>
      .block-container {padding-top: 1.5rem; padding-bottom: 2rem;}
      [data-testid="stAppViewContainer"] {
        background:
          radial-gradient(circle at 80% 0%, rgba(0, 214, 200, .10), transparent 30%),
          #07111f;
      }
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
      .mission-card {
        padding: 1rem 1.1rem; border: 1px solid #26364a; border-radius: .9rem;
        background: linear-gradient(135deg, rgba(16, 28, 44, .96), rgba(8, 20, 35, .96));
      }
      .eyebrow {color: #62e6da; font-size: .75rem; letter-spacing: .12em; font-weight: 700;}
      .muted {color: #91a4b7; font-size: .88rem;}
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


def render_road_hazard_section(
    *,
    artifacts_dir: Path,
    events: pd.DataFrame,
    detections: pd.DataFrame,
    metrics: dict,
    bandwidth: dict,
) -> None:
    if events.empty and detections.empty:
        st.info(
            "No road-hazard artifacts in this directory. Run `python orchestrator.py` "
            "and point the sidebar at `artifacts/latest`."
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

    metric_columns = st.columns(5)
    metric_columns[0].metric("Confirmed events", len(filtered_events))
    metric_columns[1].metric("Raw detections", int(metrics.get("detections", len(detections))))
    metric_columns[2].metric("Edge FPS", f"{float(metrics.get('end_to_end_fps', 0)):.1f}")
    metric_columns[3].metric("P95 inference", f"{float(metrics.get('p95_inference_ms', 0)):.0f} ms")
    metric_columns[4].metric("Bandwidth saved", bandwidth_label)

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


def render_traffic_section(artifacts_dir: Path) -> None:
    summary = load_json(str(artifacts_dir / "traffic_summary.json"))
    timeseries = load_csv(str(artifacts_dir / "traffic_timeseries.csv"))
    bottlenecks = load_csv(str(artifacts_dir / "bottleneck_events.csv"))

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
        st.dataframe(bottlenecks, hide_index=True, use_container_width=True)
        st_folium(
            build_bottleneck_map(bottlenecks, gps_source_type),
            height=420,
            use_container_width=True,
        )

    with st.expander("Traffic summary JSON"):
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

    columns = st.columns(4)
    columns[0].metric("Sampled frames", int(metrics.get("sampled_frames", 0)))
    columns[1].metric("Plate detections", int(metrics.get("total_detections", 0)))
    columns[2].metric("OCR results", int(metrics.get("ocr_results", 0)))
    columns[3].metric("Review queue", len(events))

    if not events:
        st.info("No plate track passed the multi-frame quality gate.")
        return

    safe_rows = [
        {
            "event_id": item.get("event_id"),
            "plate": item.get("masked_plate", ""),
            "observations": item.get("observation_count", 0),
            "winning_votes": item.get("winning_ocr_votes", 0),
            "ocr_confidence": item.get("mean_ocr_confidence", 0),
            "status": item.get("status", "pending_review"),
        }
        for item in events
    ]
    st.dataframe(pd.DataFrame(safe_rows), hide_index=True, use_container_width=True)

    event_ids = [str(item.get("event_id", "")) for item in events]
    selected_id = st.selectbox("Select ANPR evidence", event_ids)
    selected = next(item for item in events if str(item.get("event_id", "")) == selected_id)
    left, right = st.columns([1.5, 1])
    with left:
        frame_path = resolve_evidence(selected.get("evidence_frame"), artifacts_dir)
        if frame_path:
            st.image(str(frame_path), caption="Best observation frame", use_container_width=True)
        else:
            st.info("No context frame was promoted for this event.")
    with right:
        crop_path = resolve_evidence(selected.get("evidence_crop"), artifacts_dir)
        if crop_path:
            st.image(str(crop_path), caption="Plate crop", use_container_width=True)
        st.subheader(str(selected.get("masked_plate", "Masked plate")))
        st.write(f"OCR confidence: **{float(selected.get('mean_ocr_confidence', 0)):.3f}**")
        st.write(f"Winning votes: **{int(selected.get('winning_ocr_votes', 0))}**")
        st.write(
            "Location: "
            f"`{float(selected.get('latitude', 0)):.6f}, "
            f"{float(selected.get('longitude', 0)):.6f}`"
        )
        st.warning("Pending authorized human review. Plate text remains masked in this UI.")


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
            help="Quick: road + traffic. Full: road + traffic + ANPR.",
        )
        profile = {
            "Quick scan": "quick",
            "Full city scan": "full",
            "Custom scan": "custom",
        }[profile_label]
        module_labels = {
            "Road hazards": "road",
            "Traffic analytics": "traffic",
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

        with st.expander("Model settings", expanded=False):
            road_model = st.text_input("Road-hazard model", "models/road_hazards.pt")
            traffic_model = st.text_input("Traffic model", "yolov8n.pt")

    st.subheader("Preflight")
    checks = (
        preflight_checks(
            modules,
            road_model=road_model,
            traffic_model=traffic_model,
        )
        if modules
        else {}
    )
    check_columns = st.columns(max(1, len(checks)))
    for column, (name, (passed, detail)) in zip(check_columns, checks.items(), strict=False):
        column.markdown(f"{'✅' if passed else '❌'} **{name}**")
        column.caption(detail)

    all_checks_pass = bool(checks) and all(passed for passed, _ in checks.values())
    inputs_ready = video_upload is not None and (
        gps_source_type == "synthetic_demo" or gps_upload is not None
    )
    start_scan = st.button(
        "▶ Run DrishtiPath Scan",
        type="primary",
        use_container_width=True,
        disabled=not (inputs_ready and all_checks_pass and modules),
    )

    if not start_scan:
        if video_upload is not None:
            st.caption(
                f"Ready: {video_upload.name} · {human_bytes(video_upload.size)} · "
                f"{len(modules)} module(s) selected"
            )
        return None

    progress_bar = st.progress(0)
    stage_text = st.empty()

    def update_progress(
        module: str,
        status: str,
        message: str,
        completed: int,
        total: int,
    ) -> None:
        del module, status
        progress_bar.progress(int((completed / max(1, total)) * 100))
        stage_text.caption(message)

    try:
        stage_text.caption("Validating upload and GPS data")
        request = prepare_run(
            runs_root=runs_root,
            original_video_name=video_upload.name,
            video_payload=video_upload.getbuffer(),
            gps_source_type=gps_source_type,
            modules=modules,
            scan_profile=profile,
            gps_payload=gps_upload.getbuffer() if gps_upload is not None else None,
            original_gps_name=gps_upload.name if gps_upload is not None else "gps.csv",
            road_model=road_model,
            traffic_model=traffic_model,
        )
        manifest = run_analysis(request, progress=update_progress)
    except (UploadValidationError, ValueError, FileNotFoundError, OSError) as exc:
        progress_bar.empty()
        stage_text.empty()
        st.error(str(exc))
        return None

    st.session_state["active_run_dir"] = str(request.run_dir)
    progress_bar.progress(100)
    if manifest.get("status") == "completed":
        st.success("Scan complete. Open the result tabs to review intelligence.")
    else:
        st.warning("Scan finished with one or more failed modules. Successful results were kept.")
    return request.run_dir


def main() -> None:
    st.title("DrishtiPath Urban Intelligence Command Centre")
    st.caption("One upload · Edge intelligence · GIS evidence · Human-reviewed enforcement")

    runs_root = Path("artifacts/ui_runs")
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
    if st.sidebar.button("Clear active run", use_container_width=True):
        st.session_state.pop("active_run_dir", None)

    with st.sidebar.expander("Legacy artifact paths"):
        artifacts_value = st.text_input("Artifacts directory", "artifacts/latest")
        edge_report_value = st.text_input(
            "Edge benchmark report",
            "artifacts/edge_bench/road_hazards_onnx_fp32.json",
        )
    if st.sidebar.button("Refresh data", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

    edge_report_path = Path(edge_report_value)

    scan_tab, overview_tab, road_tab, traffic_tab, anpr_tab, edge_tab = st.tabs(
        [
            "New scan",
            "Mission summary",
            "Road hazards",
            "Traffic analytics",
            "ANPR review",
            "Edge benchmark",
        ]
    )
    with scan_tab:
        new_run = render_scan_launcher(runs_root)
        if new_run is not None:
            st.session_state["active_run_dir"] = str(new_run)

    active_value = st.session_state.get("active_run_dir")
    active_run = Path(active_value) if active_value else None
    road_dir = active_run / "road" if active_run else Path(artifacts_value)
    traffic_dir = active_run / "traffic" if active_run else Path(artifacts_value)
    anpr_dir = active_run / "anpr" if active_run else Path(artifacts_value)
    manifest = load_manifest(active_run) if active_run else {}

    with overview_tab:
        if active_run:
            st.subheader(active_run.name)
            render_run_summary(active_run)
        else:
            st.info("Run a new scan or select a previous dashboard run from the sidebar.")

    events = load_csv(str(road_dir / "events.csv"))
    detections = load_csv(str(road_dir / "detections.csv"))
    metrics = load_json(str(road_dir / "metrics.json"))
    bandwidth = load_json(str(road_dir / "bandwidth_report.json"))
    with road_tab:
        render_road_hazard_section(
            artifacts_dir=road_dir,
            events=events,
            detections=detections,
            metrics=metrics,
            bandwidth=bandwidth,
        )
    with traffic_tab:
        render_traffic_section(traffic_dir)
    with anpr_tab:
        stage = manifest.get("stages", {}).get("anpr") if manifest else None
        render_anpr_section(anpr_dir, stage)
    with edge_tab:
        render_edge_benchmark_section(edge_report_path)


if __name__ == "__main__":
    main()
