import json

import pytest

from urban_intelligence.incidents import link_anpr_evidence


def test_links_nearest_masked_anpr_without_leaking_complete_plate() -> None:
    complete_plate = "KA01AB1234"
    safety = [
        {
            "event_id": "safety-hitrun-v1-p2",
            "event_type": "suspected_hit_and_run",
            "video_time_s": 10.0,
            "lat": 28.6,
            "lon": 77.2,
        }
    ]
    anpr = [
        {
            "event_id": "anpr-1",
            "first_video_time_s": 12.0,
            "latitude": 28.6001,
            "longitude": 77.2001,
            "masked_plate": "KA******34",
            "normalized_plate": complete_plate,
        }
    ]
    linked = link_anpr_evidence(safety, anpr)
    assert linked[0]["anpr_event_id"] == "anpr-1"
    assert linked[0]["masked_plate"] == "KA******34"
    assert complete_plate not in json.dumps(linked)


def test_rejects_distant_or_stale_plate_candidates() -> None:
    safety = [
        {
            "event_id": "rash-1",
            "event_type": "rash_driving_candidate",
            "video_time_s": 10,
            "lat": 28.6,
            "lon": 77.2,
        }
    ]
    anpr = [
        {
            "event_id": "anpr-old",
            "first_video_time_s": 30,
            "latitude": 28.6,
            "longitude": 77.2,
            "masked_plate": "XX****00",
        },
        {
            "event_id": "anpr-far",
            "first_video_time_s": 11,
            "latitude": 29.6,
            "longitude": 78.2,
            "masked_plate": "YY****11",
        },
    ]
    linked = link_anpr_evidence(safety, anpr)
    assert linked[0]["anpr_link_status"] == "no_candidate"


def test_non_incident_events_are_not_linked() -> None:
    linked = link_anpr_evidence(
        [{"event_id": "traffic-1", "event_type": "bottleneck"}],
        [{"event_id": "anpr-1", "masked_plate": "AA****00"}],
    )
    assert linked[0]["anpr_link_status"] == "not_applicable"


def test_thresholds_must_be_positive() -> None:
    with pytest.raises(ValueError, match="positive"):
        link_anpr_evidence([], [], max_time_delta_s=0)
