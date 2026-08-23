from urban_intelligence.safety import RoadSafetyAnalyzer, TrackedRoadUser
from urban_intelligence.traffic import parse_roi


def tracked(
    frame: int,
    time_s: float,
    track_id: int,
    class_name: str,
    bbox: tuple[float, float, float, float],
) -> TrackedRoadUser:
    return TrackedRoadUser(
        frame_index=frame,
        video_time_s=time_s,
        track_id=track_id,
        class_name=class_name,
        confidence=0.9,
        bbox=bbox,
        latitude=28.6,
        longitude=77.2,
    )


def analyzer() -> RoadSafetyAnalyzer:
    return RoadSafetyAnalyzer(
        crossing_roi=parse_roi("0.0,0.30,1.0,1.0"),
        conflict_distance=0.20,
        collision_distance=0.08,
        lateral_speed_threshold=0.40,
        rapid_approach_threshold=0.80,
        hit_and_run_gap_s=1.0,
    )


def test_school_zone_conflict_never_claims_child_identity() -> None:
    engine = analyzer()
    engine.observe(
        [
            tracked(0, 0.0, 1, "person", (100, 300, 150, 450)),
            tracked(0, 0.0, 2, "car", (500, 300, 650, 480)),
        ],
        frame_width=1000,
        frame_height=600,
        school_zone_active=True,
    )
    events = engine.observe(
        [
            tracked(1, 1.0, 1, "person", (130, 300, 180, 450)),
            tracked(1, 1.0, 2, "car", (260, 300, 410, 480)),
        ],
        frame_width=1000,
        frame_height=600,
        school_zone_active=True,
    )
    conflict = next(item for item in events if item.event_type == "vulnerable_pedestrian_conflict")
    assert conflict.school_zone_context is True
    assert conflict.child_identity_inferred is False
    assert conflict.requires_human_review is True


def test_conflict_emits_only_once_for_same_tracks() -> None:
    engine = analyzer()
    for frame, car_x in ((0, 500), (1, 300), (2, 220)):
        engine.observe(
            [
                tracked(frame, float(frame), 1, "person", (100, 300, 150, 450)),
                tracked(frame, float(frame), 2, "car", (car_x, 300, car_x + 150, 480)),
            ],
            frame_width=1000,
            frame_height=600,
        )
    repeated = engine.observe(
        [
            tracked(3, 3.0, 1, "person", (100, 300, 150, 450)),
            tracked(3, 3.0, 2, "car", (210, 300, 360, 480)),
        ],
        frame_width=1000,
        frame_height=600,
    )
    assert sum(item.event_type == "vulnerable_pedestrian_conflict" for item in engine.events) == 1
    assert all(item.event_type != "vulnerable_pedestrian_conflict" for item in repeated)


def test_abrupt_lateral_motion_is_a_candidate_not_speed_claim() -> None:
    engine = analyzer()
    engine.observe(
        [tracked(0, 0.0, 7, "car", (100, 300, 200, 400))],
        frame_width=1000,
        frame_height=600,
    )
    events = engine.observe(
        [tracked(1, 0.5, 7, "car", (500, 300, 600, 400))],
        frame_width=1000,
        frame_height=600,
    )
    event = next(item for item in events if item.subtype == "abrupt_lateral_motion")
    assert event.event_type == "rash_driving_candidate"
    assert event.method == "image_space_trajectory_heuristic"


def test_collision_proximity_then_departure_emits_hit_and_run_candidate() -> None:
    engine = analyzer()
    engine.observe(
        [
            tracked(0, 0.0, 3, "person", (200, 300, 260, 460)),
            tracked(0, 0.0, 9, "car", (210, 320, 360, 470)),
        ],
        frame_width=1000,
        frame_height=600,
    )
    events = engine.observe(
        [tracked(2, 1.2, 3, "person", (200, 320, 270, 470))],
        frame_width=1000,
        frame_height=600,
    )
    event = next(item for item in events if item.event_type == "suspected_hit_and_run")
    assert event.vehicle_track_id == 9
    assert event.person_track_id == 3
    assert event.status == "pending_review"


def test_untracked_objects_are_not_accepted() -> None:
    try:
        tracked(0, 0.0, -1, "person", (0, 0, 1, 1))
    except ValueError as exc:
        assert "non-negative" in str(exc)
    else:
        raise AssertionError("negative track IDs must be rejected")
