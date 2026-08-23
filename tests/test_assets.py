import json
from pathlib import Path

import pytest

from urban_intelligence.assets import (
    AssetInventoryInspector,
    AssetObservation,
    ExpectedAsset,
    load_asset_inventory,
    normalize_asset_class,
)
from urban_intelligence.gps import GPSPoint


def observation(frame: int, class_name: str) -> AssetObservation:
    return AssetObservation(
        frame_index=frame,
        video_time_s=float(frame),
        class_name=class_name,
        confidence=0.8,
        latitude=28.6,
        longitude=77.2,
    )


def inventory_asset(**overrides: object) -> ExpectedAsset:
    values = {
        "asset_id": "crossing-1",
        "asset_type": "zebra_crossing",
        "latitude": 28.6,
        "longitude": 77.2,
        "inspection_radius_m": 30.0,
        "camera_visible": True,
        "min_sampled_frames": 3,
        "min_visual_hits": 2,
    }
    values.update(overrides)
    return ExpectedAsset(**values)  # type: ignore[arg-type]


def test_asset_class_aliases_are_normalized() -> None:
    assert normalize_asset_class("Crosswalk") == "zebra_crossing"
    assert normalize_asset_class("Standing Water") == "waterlogging"
    assert normalize_asset_class("Road Median") == "road_divider"


def test_visible_expected_asset_with_no_hits_becomes_review_candidate() -> None:
    inspector = AssetInventoryInspector([inventory_asset()])
    for frame in range(3):
        assert not inspector.observe(
            frame_index=frame,
            video_time_s=float(frame),
            location=GPSPoint(float(frame), 28.6, 77.2),
            observations=[],
        )
    events = inspector.observe(
        frame_index=3,
        video_time_s=3.0,
        location=GPSPoint(3.0, 28.61, 77.21),
        observations=[],
    )
    assert len(events) == 1
    event = events[0]
    assert event.class_name == "missing_zebra_crossing"
    assert event.method == "inventory_absence_heuristic"
    assert event.requires_human_review is True


def test_inventory_never_claims_missing_without_camera_visibility_contract() -> None:
    inspector = AssetInventoryInspector([inventory_asset(camera_visible=False)])
    for frame in range(5):
        inspector.observe(
            frame_index=frame,
            video_time_s=float(frame),
            location=GPSPoint(float(frame), 28.6, 77.2),
            observations=[],
        )
    assert inspector.finalize() == []


def test_present_asset_suppresses_missing_event() -> None:
    inspector = AssetInventoryInspector([inventory_asset()])
    for frame in range(3):
        inspector.observe(
            frame_index=frame,
            video_time_s=float(frame),
            location=GPSPoint(float(frame), 28.6, 77.2),
            observations=[observation(frame, "crosswalk")],
        )
    assert inspector.finalize() == []


def test_damaged_asset_wins_over_healthy_observations() -> None:
    inspector = AssetInventoryInspector([inventory_asset()])
    classes = ["zebra_crossing", "damaged_zebra_crossing", "damaged_zebra_crossing"]
    for frame, class_name in enumerate(classes):
        inspector.observe(
            frame_index=frame,
            video_time_s=float(frame),
            location=GPSPoint(float(frame), 28.6, 77.2),
            observations=[observation(frame, class_name)],
        )
    event = inspector.finalize()[0]
    assert event.event_type == "damaged_asset"
    assert event.class_name == "damaged_zebra_crossing"
    assert event.visual_hits == 2
    assert event.evidence_strength == pytest.approx(2 / 3)


def test_load_inventory_validates_shape_and_duplicate_ids(tmp_path: Path) -> None:
    path = tmp_path / "inventory.json"
    path.write_text(
        json.dumps(
            [
                {
                    "asset_id": "divider-1",
                    "asset_type": "divider",
                    "latitude": 28.6,
                    "longitude": 77.2,
                    "camera_visible": True,
                }
            ]
        ),
        encoding="utf-8",
    )
    loaded = load_asset_inventory(path)
    assert loaded[0].asset_type == "divider"

    with pytest.raises(ValueError, match="duplicate"):
        AssetInventoryInspector([loaded[0], loaded[0]])
