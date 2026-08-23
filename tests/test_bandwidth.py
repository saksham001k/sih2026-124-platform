from bandwidth_demo import calculate_bandwidth


def test_calculates_measured_reduction() -> None:
    report = calculate_bandwidth(raw_video_bytes=1000, payload_bytes=100)
    assert report["measurement_valid"]
    assert report["bytes_avoided"] == 900
    assert report["reduction_percent"] == 90.0
    assert report["compression_ratio"] == 10.0


def test_handles_empty_inputs() -> None:
    report = calculate_bandwidth(raw_video_bytes=0, payload_bytes=0)
    assert not report["measurement_valid"]
    assert report["reduction_percent"] == 0.0
    assert report["compression_ratio"] == 0.0


def test_does_not_claim_savings_without_an_edge_payload() -> None:
    report = calculate_bandwidth(raw_video_bytes=1000, payload_bytes=0)
    assert not report["measurement_valid"]
    assert report["reduction_percent"] == 0.0
