from pathlib import Path

from urban_intelligence.gps import GPSPoint, GPSTrack, haversine_m, load_gps_csv


def test_interpolates_using_video_time() -> None:
    track = GPSTrack([GPSPoint(0, 10.0, 20.0), GPSPoint(1, 12.0, 24.0)])
    point = track.for_frame(frame_index=15, fps=30)
    assert point.timestamp_s == 0.5
    assert point.latitude == 11.0
    assert point.longitude == 22.0


def test_clamps_outside_track() -> None:
    track = GPSTrack([GPSPoint(2, 10.0, 20.0), GPSPoint(3, 11.0, 21.0)])
    assert track.at(0).latitude == 10.0
    assert track.at(9).longitude == 21.0


def test_loads_repository_gps() -> None:
    track = load_gps_csv(Path("gps_data.csv"))
    assert len(track.points) == 121
    assert track.points[0].timestamp_s == 0


def test_haversine_zero_and_known_scale() -> None:
    assert haversine_m(28.6, 77.2, 28.6, 77.2) == 0
    assert 100 < haversine_m(28.6, 77.2, 28.601, 77.2) < 120
