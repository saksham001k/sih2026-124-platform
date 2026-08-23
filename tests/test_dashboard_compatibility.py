import ast
from pathlib import Path


def test_deprecated_container_width_is_not_used() -> None:
    tree = ast.parse(Path("dashboard.py").read_text(encoding="utf-8"))
    deprecated_calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and any(keyword.arg == "use_container_width" for keyword in node.keywords)
    ]

    assert deprecated_calls == []


def test_dashboard_does_not_present_raw_plate_proposals_as_verified_detections() -> None:
    source = Path("dashboard.py").read_text(encoding="utf-8")

    assert "Plate detections" not in source
    assert "Plate-like proposals" in source
    assert "No verified plate evidence" in source


def test_dashboard_exposes_presentation_navigation_and_replay() -> None:
    source = Path("dashboard.py").read_text(encoding="utf-8")

    assert "Presentation mode" in source
    assert "Mission control" in source
    assert "Evidence review" in source
    assert "Mission replay" in source
