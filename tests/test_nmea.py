from collections import deque

import pytest

from urban_intelligence.nmea import SerialNMEAGPS, nmea_checksum, parse_rmc_sentence


def sentence(payload: str) -> str:
    return f"${payload}*{nmea_checksum(payload):02X}"


def test_parses_valid_gnrmc_fix_and_speed() -> None:
    fix = parse_rmc_sentence(
        sentence("GNRMC,123519.00,A,2836.8340,N,07712.5400,E,10.0,0.0,230826,,,A"),
        timestamp_s=2.5,
    )
    assert fix.point.timestamp_s == 2.5
    assert fix.point.latitude == pytest.approx(28.6139, abs=0.0001)
    assert fix.point.longitude == pytest.approx(77.209, abs=0.0001)
    assert fix.speed_mps == pytest.approx(5.14444)
    assert fix.captured_at is not None


def test_rejects_bad_checksum_void_fix_and_non_rmc_sentence() -> None:
    with pytest.raises(ValueError, match="checksum mismatch"):
        parse_rmc_sentence("$GPRMC,1,A,1,N,1,E,0,0,010101*00", timestamp_s=0)
    with pytest.raises(ValueError, match="not active"):
        parse_rmc_sentence(
            sentence("GPRMC,123519,V,2836.8340,N,07712.5400,E,0,0,230826"),
            timestamp_s=0,
        )
    with pytest.raises(ValueError, match="not a supported RMC"):
        parse_rmc_sentence(sentence("GPGGA,123519,0,0,0,0"), timestamp_s=0)


def test_serial_reader_keeps_latest_valid_fix_and_closes() -> None:
    payloads = deque(
        [
            b"garbage\n",
            (
                sentence("GPRMC,123519,A,2836.8340,N,07712.5400,E,0,0,230826")
                + "\n"
            ).encode(),
        ]
    )

    class FakeSerial:
        def __init__(self) -> None:
            self.closed = False

        def readline(self) -> bytes:
            return payloads.popleft() if payloads else b""

        def close(self) -> None:
            self.closed = True

    fake = FakeSerial()
    clock_value = [100.0]

    def clock() -> float:
        clock_value[0] += 0.01
        return clock_value[0]

    reader = SerialNMEAGPS(
        "/dev/ttyUSB0",
        serial_factory=lambda *_args, **_kwargs: fake,
        clock=clock,
    )
    reader.start()
    fix = reader.wait_for_fix(timeout_s=1.0)
    assert fix.point.latitude == pytest.approx(28.6139, abs=0.0001)
    assert reader.snapshot()["invalid_sentences"] >= 1
    reader.close()
    assert fake.closed is True
