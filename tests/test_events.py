import pytest

from orchestrator import merge_or_add_event, road_detection_geometry_is_plausible, run_pipeline


def event(event_id: str, lat: float, lon: float) -> dict:
    return {
        "event_id": event_id,
        "class": "pothole",
        "confidence": 0.8,
        "lat": lat,
        "lon": lon,
        "observation_count": 1,
    }


def test_merges_same_class_inside_radius() -> None:
    events = [event("one", 28.6, 77.2)]
    added = merge_or_add_event(events, event("two", 28.60001, 77.2), 12)
    assert not added
    assert len(events) == 1
    assert events[0]["observation_count"] == 2


def test_keeps_spatially_distinct_events() -> None:
    events = [event("one", 28.6, 77.2)]
    added = merge_or_add_event(events, event("two", 28.61, 77.2), 12)
    assert added
    assert len(events) == 2


def test_road_pipeline_validates_settings_before_loading_runtime() -> None:
    with pytest.raises(ValueError, match="frame_skip"):
        run_pipeline(
            input_path=None,  # type: ignore[arg-type]
            gps_path=None,  # type: ignore[arg-type]
            output_dir=None,  # type: ignore[arg-type]
            model_path="model.pt",
            frame_skip=0,
        )


def test_road_geometry_gate_rejects_sky_and_degenerate_proposals() -> None:
    assert road_detection_geometry_is_plausible(
        (300, 400, 520, 650),
        frame_width=1280,
        frame_height=720,
    )
    assert not road_detection_geometry_is_plausible(
        (300, 20, 520, 120),
        frame_width=1280,
        frame_height=720,
    )
    assert not road_detection_geometry_is_plausible(
        (300, 400, 300, 650),
        frame_width=1280,
        frame_height=720,
    )
