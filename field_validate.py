"""Generate a field-verification report from one live edge mission directory."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from urban_intelligence.field_validation import evaluate_field_run


def _read_object(path: Path) -> dict:
    if not path.is_file():
        raise ValueError(f"Required mission file not found: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Mission file must contain an object: {path}")
    return value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mission-dir", required=True)
    parser.add_argument("--minimum-runtime-s", type=float, default=60.0)
    parser.add_argument("--minimum-captured-frames", type=int, default=300)
    parser.add_argument("--output", help="Defaults to MISSION_DIR/field_validation.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    mission_dir = Path(args.mission_dir)
    try:
        report = evaluate_field_run(
            _read_object(mission_dir / "metrics.json"),
            _read_object(mission_dir / "device_status.json"),
            minimum_runtime_s=args.minimum_runtime_s,
            minimum_captured_frames=args.minimum_captured_frames,
        )
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        raise SystemExit(str(exc)) from exc
    output = Path(args.output) if args.output else mission_dir / "field_validation.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"Field validation written to: {output}")


if __name__ == "__main__":
    main()
