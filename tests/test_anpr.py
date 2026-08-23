import pytest

from anpr_pipeline import consensus, is_plausible_indian_plate, normalize_plate


def test_normalizes_plate_text() -> None:
    assert normalize_plate("dl-01 ab 1234") == "DL01AB1234"


def test_validates_common_indian_plate_format() -> None:
    assert is_plausible_indian_plate("DL01AB1234")
    assert not is_plausible_indian_plate("NOTAPLATE")


def test_consensus_prefers_multi_frame_vote() -> None:
    plate, confidence, votes = consensus(
        [("DL01AB1234", 0.8), ("dl-01-ab-1234", 0.9), ("DL01A81234", 0.95)]
    )
    assert plate == "DL01AB1234"
    assert confidence == pytest.approx(0.85)
    assert votes == 2
