import json
from pathlib import Path

from urban_intelligence.fleet import FleetIngestionService, FleetStore


def envelope(
    event_id: str,
    *,
    vehicle: str = "bus-1",
    mission: str = "mission-1",
    time_s: float = 0.0,
    latitude: float = 28.6,
    longitude: float = 77.2,
    class_name: str = "pothole",
) -> dict:
    return {
        "event_id": event_id,
        "payload": {
            "event_id": event_id,
            "vehicle_id": vehicle,
            "mission_id": mission,
            "route_id": "route-7",
            "event_type": "confirmed_edge_detection",
            "class": class_name,
            "video_time_s": time_s,
            "lat": latitude,
            "lon": longitude,
            "gps_source_type": "real_telemetry",
            "status": "pending_review",
        },
    }


def test_ingestion_is_idempotent_and_updates_mission_bounds(tmp_path: Path) -> None:
    store = FleetStore(tmp_path / "fleet.db")
    first = store.ingest_envelope(envelope("event-1", time_s=2, latitude=28.6))
    duplicate = store.ingest_envelope(envelope("event-1", time_s=2, latitude=28.6))
    store.ingest_envelope(envelope("event-2", time_s=12, latitude=28.61))
    assert first.accepted is True
    assert duplicate.duplicate is True
    mission = store.missions()[0]
    assert mission["event_count"] == 2
    assert mission["start_time_s"] == 2
    assert mission["end_time_s"] == 12
    assert mission["end_latitude"] == 28.61


def test_deficiency_clusters_deduplicate_repeat_fleet_sightings(tmp_path: Path) -> None:
    store = FleetStore(tmp_path / "fleet.db")
    store.ingest_envelope(envelope("event-1", vehicle="bus-1", latitude=28.6))
    store.ingest_envelope(
        envelope("event-2", vehicle="bus-2", latitude=28.60005, longitude=77.20005)
    )
    clusters = store.deficiency_clusters(radius_m=20)
    assert len(clusters) == 1
    assert clusters[0]["sightings"] == 2
    assert clusters[0]["unique_vehicles"] == 2


def test_od_matrix_aggregates_multiple_vehicle_missions(tmp_path: Path) -> None:
    store = FleetStore(tmp_path / "fleet.db")
    for vehicle, mission in (("bus-1", "m1"), ("bus-2", "m2")):
        store.ingest_envelope(
            envelope("start", vehicle=vehicle, mission=mission, time_s=0, latitude=28.6)
        )
        store.ingest_envelope(
            envelope("end", vehicle=vehicle, mission=mission, time_s=60, latitude=28.61)
        )
    od = store.od_matrix(precision=3)
    assert od[0]["trip_count"] == 2
    assert od[0]["unique_vehicles"] == 2
    assert od[0]["mean_observed_duration_s"] == 60


def test_authenticated_service_rejects_bad_token_and_accepts_duplicate(tmp_path: Path) -> None:
    service = FleetIngestionService(FleetStore(tmp_path / "fleet.db"), "secret")
    packet = envelope("event-1")
    body = json.dumps(packet).encode()
    assert service.handle(
        body, authorization="Bearer wrong", idempotency_key="event-1"
    )[0] == 401
    first_status, first = service.handle(
        body, authorization="Bearer secret", idempotency_key="event-1"
    )
    duplicate_status, duplicate = service.handle(
        body, authorization="Bearer secret", idempotency_key="event-1"
    )
    assert first_status == 202 and first["accepted"] is True
    assert duplicate_status == 200 and duplicate["duplicate"] is True
    assert service.authorized("Bearer secret") is True
    assert service.authorized("Bearer wrong") is False


def test_export_writes_summary_geojson_clusters_and_od(tmp_path: Path) -> None:
    store = FleetStore(tmp_path / "fleet.db")
    store.ingest_envelope(envelope("event-1"))
    outputs = store.export(tmp_path / "export")
    assert all(path.is_file() for path in outputs.values())
    geojson = json.loads(outputs["geojson"].read_text(encoding="utf-8"))
    assert geojson["type"] == "FeatureCollection"
    assert geojson["features"][0]["geometry"]["type"] == "Point"
