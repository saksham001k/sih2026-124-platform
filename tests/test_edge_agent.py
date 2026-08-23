import argparse
import json
import time
from pathlib import Path

import pytest

from edge_agent import (
    build_runner_specs,
    load_geofences,
    model_schedules,
    parse_fixed_gps,
    parse_source,
    prepare_output_dir,
    run_agent,
    validate_gps_configuration,
    validate_mission_identifiers,
)


def test_parse_source_distinguishes_camera_index_from_paths_and_urls() -> None:
    assert parse_source("0") == 0
    assert parse_source(" 12 ") == 12
    assert parse_source("clips/route.mp4") == "clips/route.mp4"
    assert parse_source("rtsp://camera/stream") == "rtsp://camera/stream"
    with pytest.raises(ValueError, match="empty"):
        parse_source("   ")


def test_fixed_gps_is_validated() -> None:
    point = parse_fixed_gps("28.6139,77.2090")
    assert point.latitude == 28.6139
    assert point.longitude == 77.209
    with pytest.raises(ValueError, match="LAT,LON"):
        parse_fixed_gps("Delhi")
    with pytest.raises(ValueError, match="invalid"):
        parse_fixed_gps("200,77")


def test_load_geofences_requires_a_list_of_valid_objects(tmp_path: Path) -> None:
    path = tmp_path / "geofences.json"
    path.write_text(
        json.dumps(
            [
                {
                    "key": "school-zone-7",
                    "latitude": 28.6139,
                    "longitude": 77.209,
                    "radius_m": 150,
                }
            ]
        ),
        encoding="utf-8",
    )
    geofences = load_geofences(path)
    assert len(geofences) == 1
    assert geofences[0].key == "school-zone-7"

    path.write_text(json.dumps({"key": "not-a-list"}), encoding="utf-8")
    with pytest.raises(ValueError, match="list"):
        load_geofences(path)


def test_pi4_profile_registers_only_models_that_are_configured() -> None:
    schedules = model_schedules("pi4", {"road_damage", "traffic"})
    assert [item.name for item in schedules] == ["traffic", "road_damage"]
    assert [item.target_fps for item in schedules] == [3.0, 2.0]
    with pytest.raises(ValueError, match="profile"):
        model_schedules("imaginary", {"traffic"})
    with pytest.raises(ValueError, match="unknown configured"):
        model_schedules("pi4", {"telepathy"})


def test_output_directory_never_overwrites_previous_mission(tmp_path: Path) -> None:
    output = tmp_path / "mission"
    prepare_output_dir(output)
    assert output.is_dir()
    (output / "metrics.json").write_text("{}", encoding="utf-8")
    with pytest.raises(FileExistsError, match="not empty"):
        prepare_output_dir(output)


def test_runner_specs_keep_asset_model_optional() -> None:
    arguments = argparse.Namespace(
        traffic_model="traffic.pt",
        traffic_confidence=0.25,
        road_model="road.pt",
        road_confidence=0.20,
        asset_model="",
        asset_confidence=0.25,
        image_size=416,
    )
    specs = build_runner_specs(arguments)
    assert [item.name for item in specs] == ["traffic", "road_damage"]
    assert specs[0].tracking is True
    assert "car" in specs[0].allowed_classes

    arguments.asset_model = "assets.pt"
    assert [item.name for item in build_runner_specs(arguments)] == [
        "traffic",
        "road_damage",
        "urban_assets",
    ]


def test_fixed_demo_coordinate_cannot_be_labelled_real_telemetry() -> None:
    with pytest.raises(ValueError, match="cannot be labelled"):
        validate_gps_configuration(
            gps_csv=None,
            fixed_gps="28.6,77.2",
            gps_source_type="real_telemetry",
        )
    validate_gps_configuration(
        gps_csv=None,
        fixed_gps="28.6,77.2",
        gps_source_type="synthetic_demo",
    )


def test_live_nmea_requires_real_telemetry_and_is_mutually_exclusive() -> None:
    validate_gps_configuration(
        gps_csv=None,
        fixed_gps=None,
        gps_nmea_device="/dev/ttyUSB0",
        gps_source_type="real_telemetry",
    )
    with pytest.raises(ValueError, match="must be labelled"):
        validate_gps_configuration(
            gps_csv=None,
            fixed_gps=None,
            gps_nmea_device="/dev/ttyUSB0",
            gps_source_type="synthetic_demo",
        )


def test_mission_identifiers_are_safe_for_outbox_and_fleet_storage() -> None:
    validate_mission_identifiers("bus-42", "mission-2026-08-23", "route.blue")
    with pytest.raises(ValueError, match="vehicle_id"):
        validate_mission_identifiers("../../bus", "mission-1", "route-1")
    with pytest.raises(ValueError, match="only one"):
        validate_gps_configuration(
            gps_csv="route.csv",
            fixed_gps=None,
            gps_nmea_device="/dev/ttyUSB0",
            gps_source_type="real_telemetry",
        )


def test_live_agent_loop_writes_truthful_metrics_with_injected_runtimes(tmp_path: Path) -> None:
    class FakeCapture:
        def __init__(self) -> None:
            self.index = 0
            self.released = False

        def isOpened(self) -> bool:
            return True

        def get(self, _: int) -> float:
            return 30.0

        def read(self) -> tuple[bool, object | None]:
            if self.index >= 8:
                return False, None
            self.index += 1
            time.sleep(0.01)
            return True, object()

        def release(self) -> None:
            self.released = True

    capture = FakeCapture()

    class FakeCV2:
        CAP_PROP_FPS = 5

        @staticmethod
        def VideoCapture(_: object) -> FakeCapture:
            return capture

    class FakeRunner:
        def infer(self, _: object) -> list[object]:
            return []

    arguments = argparse.Namespace(
        output_dir=str(tmp_path / "mission"),
        source="0",
        gps_csv=None,
        fixed_gps="28.6139,77.2090",
        gps_source_type="synthetic_demo",
        geofences=None,
        traffic_model="traffic.pt",
        traffic_confidence=0.25,
        road_model="road.pt",
        road_confidence=0.20,
        asset_model="",
        asset_confidence=0.25,
        image_size=416,
        device="cpu",
        profile="desktop",
        buffer_frames=2,
        duration_s=0,
        health_interval_s=0.01,
    )
    report = run_agent(
        arguments,
        cv2_module=FakeCV2,
        runner_factory=lambda _spec, _device: FakeRunner(),
        health_sampler=lambda: {
            "raspberry_pi": True,
            "device_model": "Raspberry Pi test double",
        },
    )

    assert capture.released is True
    assert report["capture"]["captured_frames"] == 8
    assert report["analytics_attempts"] >= 2
    assert report["gps_source_type"] == "synthetic_demo"
    assert report["claim_policy"].startswith("Profile rates are configuration targets")
    assert (tmp_path / "mission/metrics.json").is_file()
    status = json.loads((tmp_path / "mission/device_status.json").read_text(encoding="utf-8"))
    assert status["mission_status"] == "completed"
