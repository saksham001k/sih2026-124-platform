import json

from urban_intelligence.command_center import build_operational_scene, filter_scene_at


def test_builds_all_operational_layers_and_centres_on_route() -> None:
    scene = build_operational_scene(
        route_records=[
            {"timestamp": 1, "lat": 28.61, "lon": 77.21},
            {"timestamp": 0, "lat": 28.60, "lon": 77.20},
        ],
        road_events=[
            {
                "event_id": "evt-1",
                "class": "D40",
                "confidence": 0.8,
                "lat": 28.605,
                "lon": 77.205,
            }
        ],
        traffic_windows=[
            {
                "window_index": 1,
                "latitude": 28.606,
                "longitude": 77.206,
                "mean_vehicle_count": 6,
                "mean_occupancy": 0.25,
                "congested": "False",
            }
        ],
        bottleneck_events=[
            {
                "event_id": "traffic-1",
                "latitude": 28.607,
                "longitude": 77.207,
            }
        ],
        anpr_events=[
            {
                "event_id": "anpr-1",
                "masked_plate": "KA*****65",
                "latitude": 28.608,
                "longitude": 77.208,
            }
        ],
    )

    assert scene["route"] == [[77.2, 28.6], [77.21, 28.61]]
    assert scene["counts"] == {
        "hazards": 1,
        "traffic_windows": 1,
        "bottlenecks": 1,
        "anpr": 1,
    }
    assert scene["hazards"][0]["label"] == "Pothole"
    assert scene["traffic"][0]["status"] == "observed"
    assert scene["center"]["latitude"] == 28.605
    assert scene["center"]["longitude"] == 77.205


def test_discards_invalid_coordinates_and_deduplicates_route() -> None:
    scene = build_operational_scene(
        route_records=[
            {"timestamp": 0, "lat": 28.6, "lon": 77.2},
            {"timestamp": 1, "lat": 28.6, "lon": 77.2},
            {"timestamp": 2, "lat": 200, "lon": 77.2},
        ],
        road_events=[{"lat": "not-a-number", "lon": 77.2}],
    )
    assert scene["route"] == [[77.2, 28.6]]
    assert scene["hazards"] == []


def test_never_copies_complete_plate_text_into_scene() -> None:
    complete_plate = "KA01AB1234"
    scene = build_operational_scene(
        anpr_events=[
            {
                "event_id": "anpr-1",
                "normalized_plate": complete_plate,
                "masked_plate": "KA******34",
                "latitude": 28.6,
                "longitude": 77.2,
            }
        ]
    )
    serialized = json.dumps(scene)
    assert complete_plate not in serialized
    assert "KA******34" in serialized


def test_timeline_is_sorted_and_preserves_evidence_references() -> None:
    scene = build_operational_scene(
        road_events=[
            {
                "event_id": "road-late",
                "class": "pothole",
                "video_time_s": 12,
                "lat": 28.6,
                "lon": 77.2,
                "evidence_frame": "road/evidence/frame.jpg",
            }
        ],
        anpr_events=[
            {
                "event_id": "anpr-early",
                "first_video_time_s": 3,
                "masked_plate": "DL*****01",
                "latitude": 28.6,
                "longitude": 77.2,
                "evidence_crop": "anpr/evidence/crop.jpg",
            }
        ],
    )
    assert [item["id"] for item in scene["timeline"]] == [
        "anpr-early",
        "road-late",
    ]
    assert scene["timeline"][0]["evidence_crop"].endswith("crop.jpg")
    assert scene["timeline"][1]["evidence_frame"].endswith("frame.jpg")


def test_scene_replay_filters_route_and_events_by_time() -> None:
    scene = build_operational_scene(
        route_records=[
            {"timestamp_s": 0, "lat": 28.60, "lon": 77.20},
            {"timestamp_s": 5, "lat": 28.61, "lon": 77.21},
            {"timestamp_s": 10, "lat": 28.62, "lon": 77.22},
        ],
        road_events=[
            {
                "event_id": "early",
                "video_time_s": 3,
                "lat": 28.605,
                "lon": 77.205,
            },
            {
                "event_id": "late",
                "video_time_s": 9,
                "lat": 28.615,
                "lon": 77.215,
            },
        ],
    )

    replay = filter_scene_at(scene, 5)
    assert replay["route"] == [[77.2, 28.6], [77.21, 28.61]]
    assert [item["id"] for item in replay["hazards"]] == ["early"]
    assert replay["counts"]["hazards"] == 1
