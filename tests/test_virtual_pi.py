import json
from pathlib import Path

from virtual_pi_test import run_virtual_pi_test


def test_virtual_pi_exercises_stack_without_claiming_physical_hardware(tmp_path: Path) -> None:
    output = tmp_path / "virtual-pi"
    report = run_virtual_pi_test(output, frame_count=140, frame_delay_s=0.035)

    assert report["passed"] is True
    assert report["arm_instruction_emulation"] is False
    assert report["physical_hardware_tested"] is False
    assert report["physical_field_gate"]["field_verified"] is False
    assert report["application_checks"]["physical_claim_blocked"] is True
    assert report["edge_metrics"]["gps_provider"] == "serial_nmea"
    assert report["edge_metrics"]["confirmed_events"] >= 1
    assert report["delivery"]["pending_after"] == 0
    assert report["fleet"]["event_count"] == report["edge_metrics"]["confirmed_events"]
    assert json.loads((output / "virtual_pi_report.json").read_text())["passed"] is True
