"""Safe upload sessions and failure-isolated demo pipeline orchestration."""

from __future__ import annotations

import importlib.util
import json
import re
import secrets
import shutil
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from urban_intelligence.gps import load_gps_csv

VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv"}
SUPPORTED_MODULES = ("road", "traffic", "assets", "anpr")
SCAN_PROFILES: dict[str, tuple[str, ...]] = {
    "quick": ("road", "traffic"),
    "full": SUPPORTED_MODULES,
}
MAX_UPLOAD_BYTES = 500 * 1024 * 1024
MAX_VIDEO_DURATION_S = 15 * 60
SAFE_RUN_ID_PATTERN = re.compile(r"^run-[A-Za-z0-9-]+$")

ProgressCallback = Callable[[str, str, str, int, int], None]
StageRunner = Callable[["AnalysisRequest"], Mapping[str, Any]]


class UploadValidationError(ValueError):
    """Raised when a dashboard upload cannot be safely processed."""


@dataclass(frozen=True, slots=True)
class VideoMetadata:
    frame_count: int
    fps: float
    duration_s: float
    width: int
    height: int


@dataclass(frozen=True, slots=True)
class AnalysisRequest:
    run_id: str
    run_dir: Path
    input_path: Path
    gps_path: Path
    gps_source_type: str
    modules: tuple[str, ...]
    scan_profile: str
    road_model: str = "models/road_hazards.pt"
    traffic_model: str = "yolov8n.pt"
    asset_model: str = "models/urban_assets.pt"
    asset_inventory_path: Path | None = None
    school_zones_path: Path | None = None


def modules_for_profile(profile: str, custom_modules: tuple[str, ...] = ()) -> tuple[str, ...]:
    """Resolve a named scan profile into a stable, duplicate-free module list."""
    if profile in SCAN_PROFILES:
        return SCAN_PROFILES[profile]
    if profile != "custom":
        raise ValueError("scan profile must be quick, full, or custom")
    invalid = set(custom_modules) - set(SUPPORTED_MODULES)
    if invalid:
        raise ValueError(f"unsupported analysis modules: {', '.join(sorted(invalid))}")
    selected = tuple(module for module in SUPPORTED_MODULES if module in custom_modules)
    if not selected:
        raise ValueError("custom scan requires at least one module")
    return selected


def generate_run_id(now: datetime | None = None, token: str | None = None) -> str:
    moment = now or datetime.now(UTC)
    safe_token = token or secrets.token_hex(3)
    if not safe_token.isalnum():
        raise ValueError("run token must be alphanumeric")
    return f"run-{moment.strftime('%Y%m%dT%H%M%SZ')}-{safe_token.lower()}"


def validate_upload(
    filename: str,
    payload: bytes | bytearray | memoryview,
    *,
    allowed_extensions: set[str],
    max_bytes: int = MAX_UPLOAD_BYTES,
) -> str:
    """Validate upload metadata and return a normalized safe extension."""
    extension = Path(filename).suffix.lower()
    if extension not in allowed_extensions:
        expected = ", ".join(sorted(allowed_extensions))
        raise UploadValidationError(f"Unsupported file type. Expected one of: {expected}")
    size = len(payload)
    if size == 0:
        raise UploadValidationError("Uploaded file is empty")
    if size > max_bytes:
        raise UploadValidationError(
            f"Uploaded file exceeds the {max_bytes / (1024 * 1024):.0f} MB limit"
        )
    return extension


def probe_video(
    path: Path,
    *,
    max_duration_s: float = MAX_VIDEO_DURATION_S,
) -> VideoMetadata:
    """Decode the first frame and inspect basic video metadata with OpenCV."""
    try:
        import cv2
    except ImportError as exc:
        raise UploadValidationError(
            "OpenCV is required to validate uploaded videos"
        ) from exc

    capture = cv2.VideoCapture(str(path))
    try:
        if not capture.isOpened():
            raise UploadValidationError("Uploaded video could not be opened")
        ok, frame = capture.read()
        if not ok or frame is None:
            raise UploadValidationError("Uploaded file does not contain a decodable video frame")
        fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        height, width = frame.shape[:2]
        if fps <= 0 or width <= 0 or height <= 0:
            raise UploadValidationError("Uploaded video has invalid frame metadata")
        duration_s = frame_count / fps if frame_count > 0 else 0.0
        if duration_s > max_duration_s:
            raise UploadValidationError(
                f"Video is {duration_s / 60:.1f} minutes; the demo limit is "
                f"{max_duration_s / 60:.0f} minutes"
            )
        return VideoMetadata(
            frame_count=frame_count,
            fps=round(fps, 3),
            duration_s=round(duration_s, 3),
            width=width,
            height=height,
        )
    finally:
        capture.release()


def _write_manifest(run_dir: Path, manifest: Mapping[str, Any]) -> None:
    path = run_dir / "manifest.json"
    temporary = run_dir / "manifest.json.tmp"
    temporary.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    temporary.replace(path)


def load_manifest(run_dir: Path) -> dict[str, Any]:
    path = run_dir / "manifest.json"
    if not path.is_file():
        return {}
    value = json.loads(path.read_text(encoding="utf-8"))
    return value if isinstance(value, dict) else {}


def list_completed_runs(root: Path) -> list[Path]:
    """Return newest manifest-backed runs without following arbitrary paths."""
    if not root.is_dir():
        return []
    runs = [
        path
        for path in root.iterdir()
        if path.is_dir()
        and not path.is_symlink()
        and (path / "manifest.json").is_file()
    ]
    return sorted(runs, key=lambda path: path.stat().st_mtime, reverse=True)


def prepare_run(
    *,
    runs_root: Path,
    original_video_name: str,
    video_payload: bytes | bytearray | memoryview,
    gps_source_type: str,
    modules: tuple[str, ...],
    scan_profile: str,
    gps_payload: bytes | bytearray | memoryview | None = None,
    original_gps_name: str = "gps.csv",
    synthetic_gps_path: Path = Path("gps_data.csv"),
    road_model: str = "models/road_hazards.pt",
    traffic_model: str = "yolov8n.pt",
    asset_model: str = "models/urban_assets.pt",
    asset_inventory_payload: bytes | bytearray | memoryview | None = None,
    original_asset_inventory_name: str = "assets.json",
    school_zones_payload: bytes | bytearray | memoryview | None = None,
    original_school_zones_name: str = "school_zones.json",
    run_id: str | None = None,
    video_probe: Callable[[Path], VideoMetadata] = probe_video,
) -> AnalysisRequest:
    """Persist validated dashboard inputs and initialize a portable run manifest."""
    invalid_modules = set(modules) - set(SUPPORTED_MODULES)
    if invalid_modules or not modules:
        raise ValueError("modules must contain one or more supported analysis modules")
    if gps_source_type not in {"synthetic_demo", "real_telemetry"}:
        raise ValueError("gps_source_type must be synthetic_demo or real_telemetry")

    video_extension = validate_upload(
        original_video_name,
        video_payload,
        allowed_extensions=VIDEO_EXTENSIONS,
    )
    selected_run_id = run_id or generate_run_id()
    if not SAFE_RUN_ID_PATTERN.fullmatch(selected_run_id):
        raise ValueError(
            "run_id must start with 'run-' and contain only letters, digits, or hyphens"
        )
    run_dir = runs_root / selected_run_id
    if run_dir.exists():
        raise FileExistsError(f"Run already exists: {selected_run_id}")

    input_dir = run_dir / "input"
    input_dir.mkdir(parents=True)
    video_path = input_dir / f"video{video_extension}"
    try:
        video_path.write_bytes(video_payload)
        metadata = video_probe(video_path)

        gps_path = input_dir / "gps.csv"
        if gps_source_type == "real_telemetry":
            if gps_payload is None:
                raise UploadValidationError("Upload a GPS CSV for real telemetry mode")
            validate_upload(
                original_gps_name,
                gps_payload,
                allowed_extensions={".csv"},
                max_bytes=25 * 1024 * 1024,
            )
            gps_path.write_bytes(gps_payload)
        else:
            if not synthetic_gps_path.is_file():
                raise UploadValidationError(
                    f"Synthetic demo GPS file is missing: {synthetic_gps_path}"
                )
            shutil.copy2(synthetic_gps_path, gps_path)
        gps_track = load_gps_csv(gps_path)
        gps_start_s = gps_track.points[0].timestamp_s
        gps_end_s = gps_track.points[-1].timestamp_s
        warnings: list[str] = []
        if metadata.duration_s and metadata.duration_s > gps_end_s:
            warnings.append(
                "GPS coverage ends before the video; coordinates clamp to the last "
                "telemetry point for the remaining frames."
            )

        asset_inventory_path: Path | None = None
        if asset_inventory_payload is not None:
            validate_upload(
                original_asset_inventory_name,
                asset_inventory_payload,
                allowed_extensions={".json"},
                max_bytes=5 * 1024 * 1024,
            )
            asset_inventory_path = input_dir / "assets.json"
            asset_inventory_path.write_bytes(asset_inventory_payload)

        school_zones_path: Path | None = None
        if school_zones_payload is not None:
            validate_upload(
                original_school_zones_name,
                school_zones_payload,
                allowed_extensions={".json"},
                max_bytes=5 * 1024 * 1024,
            )
            school_zones_path = input_dir / "school_zones.json"
            school_zones_path.write_bytes(school_zones_payload)

        request = AnalysisRequest(
            run_id=selected_run_id,
            run_dir=run_dir,
            input_path=video_path,
            gps_path=gps_path,
            gps_source_type=gps_source_type,
            modules=tuple(module for module in SUPPORTED_MODULES if module in modules),
            scan_profile=scan_profile,
            road_model=road_model,
            traffic_model=traffic_model,
            asset_model=asset_model,
            asset_inventory_path=asset_inventory_path,
            school_zones_path=school_zones_path,
        )
        manifest = {
            "schema_version": 1,
            "run_id": request.run_id,
            "created_at": datetime.now(UTC).isoformat(),
            "status": "ready",
            "scan_profile": scan_profile,
            "requested_modules": list(request.modules),
            "gps_source_type": gps_source_type,
            "warnings": warnings,
            "input": {
                "video": {
                    "original_name": Path(original_video_name).name,
                    "stored_path": str(video_path.relative_to(run_dir)),
                    "bytes": len(video_payload),
                    **asdict(metadata),
                },
                "gps": {
                    "original_name": (
                        Path(original_gps_name).name
                        if gps_source_type == "real_telemetry"
                        else synthetic_gps_path.name
                    ),
                    "stored_path": str(gps_path.relative_to(run_dir)),
                    "source_type": gps_source_type,
                    "start_time_s": gps_start_s,
                    "end_time_s": gps_end_s,
                    "covers_video": not warnings,
                },
                "asset_inventory": (
                    None
                    if asset_inventory_path is None
                    else {
                        "original_name": Path(original_asset_inventory_name).name,
                        "stored_path": str(asset_inventory_path.relative_to(run_dir)),
                        "bytes": len(asset_inventory_payload or b""),
                    }
                ),
                "school_zones": (
                    None
                    if school_zones_path is None
                    else {
                        "original_name": Path(original_school_zones_name).name,
                        "stored_path": str(school_zones_path.relative_to(run_dir)),
                        "bytes": len(school_zones_payload or b""),
                    }
                ),
            },
            "stages": {
                module: {"status": "queued", "message": "Waiting to run"}
                for module in request.modules
            },
            "outputs": {},
        }
        _write_manifest(run_dir, manifest)
        return request
    except BaseException:
        # This directory was created in this call and is always a child of runs_root.
        shutil.rmtree(run_dir, ignore_errors=True)
        raise


def preflight_checks(
    modules: tuple[str, ...],
    *,
    road_model: str,
    traffic_model: str = "yolov8n.pt",
    asset_model: str = "models/urban_assets.pt",
) -> dict[str, tuple[bool, str]]:
    checks: dict[str, tuple[bool, str]] = {}
    if "road" in modules:
        model_available = Path(road_model).is_file()
        runtime_available = importlib.util.find_spec("ultralytics") is not None
        available = model_available and runtime_available
        checks["Road model"] = (
            available,
            (
                road_model
                if available
                else (
                    f"Missing: {road_model}"
                    if not model_available
                    else "Install requirements.txt"
                )
            ),
        )
    if "traffic" in modules:
        runtime_available = importlib.util.find_spec("ultralytics") is not None
        model_path = Path(traffic_model)
        local_model_required = model_path.parent != Path(".")
        model_available = not local_model_required or model_path.is_file()
        available = runtime_available and model_available
        checks["Traffic runtime"] = (
            available,
            (
                f"Ultralytics ready · {traffic_model}"
                if available
                else (
                    f"Missing: {traffic_model}"
                    if not model_available
                    else "Install requirements.txt"
                )
            ),
        )
    if "anpr" in modules:
        available = importlib.util.find_spec("fast_alpr") is not None
        checks["ANPR runtime"] = (
            available,
            (
                "FastALPR ready · warm models before demo day"
                if available
                else "Install requirements-anpr.txt"
            ),
        )
    if "assets" in modules:
        runtime_available = importlib.util.find_spec("ultralytics") is not None
        model_available = Path(asset_model).is_file()
        available = runtime_available and model_available
        checks["Urban-assets model"] = (
            available,
            (
                asset_model
                if available
                else (
                    f"Missing: {asset_model}"
                    if not model_available
                    else "Install requirements.txt"
                )
            ),
        )
    return checks


def _road_runner(request: AnalysisRequest) -> Mapping[str, Any]:
    from orchestrator import run_pipeline

    return run_pipeline(
        input_path=request.input_path,
        gps_path=request.gps_path,
        output_dir=request.run_dir / "road",
        model_path=request.road_model,
        confidence=0.10,
        frame_skip=1,
        window_size=5,
        min_hits=3,
        dedupe_radius_m=12.0,
        classes="pothole,longitudinal_crack,transverse_crack,alligator_crack",
    )


def _traffic_runner(request: AnalysisRequest) -> Mapping[str, Any]:
    from traffic_analytics import load_school_zones, run_pipeline
    from urban_intelligence.traffic import parse_roi

    return run_pipeline(
        input_path=request.input_path,
        gps_path=request.gps_path,
        output_dir=request.run_dir / "traffic",
        model_path=request.traffic_model,
        confidence=0.25,
        frame_skip=3,
        window_seconds=5.0,
        roi=parse_roi("0.0,0.30,1.0,1.0"),
        congestion_min_vehicles=6.0,
        congestion_min_occupancy=0.18,
        congestion_consecutive_windows=3,
        gps_source_type=request.gps_source_type,
        gps_label=str(request.gps_path),
        school_zones=load_school_zones(request.school_zones_path),
    )


def _anpr_runner(request: AnalysisRequest) -> Mapping[str, Any]:
    from anpr_pipeline import (
        DEFAULT_CENTER_DISTANCE_THRESHOLD,
        DEFAULT_DETECTOR_MODEL,
        DEFAULT_EXECUTION_PROVIDER,
        DEFAULT_OCR_MODEL,
        create_alpr_engine,
        run_pipeline,
    )
    detector_confidence = 0.20
    predict_fn = create_alpr_engine(
        detector_model=DEFAULT_DETECTOR_MODEL,
        ocr_model=DEFAULT_OCR_MODEL,
        detector_confidence=detector_confidence,
        execution_provider=DEFAULT_EXECUTION_PROVIDER,
    )
    return run_pipeline(
        input_path=request.input_path,
        gps_path=request.gps_path,
        output_dir=request.run_dir / "anpr",
        predict_fn=predict_fn,
        detector_confidence=detector_confidence,
        sample_fps=5.0,
        min_observations=3,
        min_winning_votes=2,
        min_mean_ocr_confidence=0.80,
        iou_threshold=0.30,
        center_distance_threshold=DEFAULT_CENTER_DISTANCE_THRESHOLD,
        track_gap_s=2.0,
        show_plate_text=False,
        gps_label=str(request.gps_path),
        gps_source_type=request.gps_source_type,
        detector_model=DEFAULT_DETECTOR_MODEL,
        ocr_model=DEFAULT_OCR_MODEL,
        execution_provider=DEFAULT_EXECUTION_PROVIDER,
    )


def _assets_runner(request: AnalysisRequest) -> Mapping[str, Any]:
    from asset_inspection import run_pipeline

    return run_pipeline(
        input_path=request.input_path,
        gps_path=request.gps_path,
        output_dir=request.run_dir / "assets",
        model_path=request.asset_model,
        inventory_path=request.asset_inventory_path,
        confidence=0.25,
        frame_skip=3,
        window_size=5,
        min_hits=3,
        gps_source_type=request.gps_source_type,
    )


DEFAULT_STAGE_RUNNERS: dict[str, StageRunner] = {
    "road": _road_runner,
    "traffic": _traffic_runner,
    "assets": _assets_runner,
    "anpr": _anpr_runner,
}


def correlate_incidents(run_dir: Path) -> dict[str, Any]:
    """Link traffic safety candidates to masked ANPR evidence after both stages finish."""
    from urban_intelligence.incidents import link_anpr_evidence

    safety_path = run_dir / "traffic" / "safety_events.json"
    anpr_path = run_dir / "anpr" / "anpr_events.json"
    if not safety_path.is_file() or not anpr_path.is_file():
        return {"status": "not_run", "reason": "traffic and ANPR outputs are both required"}
    safety = json.loads(safety_path.read_text(encoding="utf-8"))
    anpr = json.loads(anpr_path.read_text(encoding="utf-8"))
    if not isinstance(safety, list) or not isinstance(anpr, list):
        raise ValueError("safety and ANPR event files must contain lists")
    incidents = link_anpr_evidence(safety, anpr)
    output_dir = run_dir / "incidents"
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / "incidents.json"
    output_path.write_text(json.dumps(incidents, indent=2), encoding="utf-8")
    return {
        "status": "completed",
        "event_count": len(incidents),
        "anpr_candidates_linked": sum(
            item.get("anpr_link_status") == "candidate_pending_review"
            for item in incidents
        ),
        "directory": "incidents",
    }


def run_analysis(
    request: AnalysisRequest,
    *,
    progress: ProgressCallback | None = None,
    stage_runners: Mapping[str, StageRunner] | None = None,
) -> dict[str, Any]:
    """Run requested stages sequentially while preserving successful outputs."""
    manifest = load_manifest(request.run_dir)
    if not manifest:
        raise FileNotFoundError(f"Run manifest not found: {request.run_dir}")

    runners = dict(DEFAULT_STAGE_RUNNERS if stage_runners is None else stage_runners)
    total = len(request.modules)
    manifest["status"] = "running"
    manifest["started_at"] = datetime.now(UTC).isoformat()
    _write_manifest(request.run_dir, manifest)

    failures = 0
    for index, module in enumerate(request.modules, start=1):
        runner = runners.get(module)
        stage = manifest["stages"][module]
        stage["status"] = "running"
        stage["started_at"] = datetime.now(UTC).isoformat()
        stage["message"] = f"Running {module} analysis"
        _write_manifest(request.run_dir, manifest)
        if progress:
            progress(module, "running", stage["message"], index - 1, total)

        try:
            if runner is None:
                raise RuntimeError(f"No runner configured for {module}")
            result = dict(runner(request))
        except (Exception, SystemExit) as exc:
            failures += 1
            stage["status"] = "failed"
            stage["message"] = str(exc) or type(exc).__name__
            stage["error_type"] = type(exc).__name__
        else:
            stage["status"] = "completed"
            stage["message"] = f"{module.title()} analysis completed"
            manifest["outputs"][module] = {
                "directory": module,
                "summary": result,
            }
        stage["finished_at"] = datetime.now(UTC).isoformat()
        _write_manifest(request.run_dir, manifest)
        if progress:
            progress(module, stage["status"], stage["message"], index, total)

    stage_failures = failures
    auxiliary_failures = 0
    try:
        correlation = correlate_incidents(request.run_dir)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        auxiliary_failures = 1
        correlation = {
            "status": "failed",
            "message": str(exc) or type(exc).__name__,
            "error_type": type(exc).__name__,
        }
    manifest["outputs"]["incident_correlation"] = correlation

    manifest["status"] = (
        "completed_with_errors" if stage_failures or auxiliary_failures else "completed"
    )
    manifest["finished_at"] = datetime.now(UTC).isoformat()
    manifest["successful_stages"] = total - stage_failures
    manifest["failed_stages"] = stage_failures
    manifest["auxiliary_failures"] = auxiliary_failures
    _write_manifest(request.run_dir, manifest)
    return manifest
