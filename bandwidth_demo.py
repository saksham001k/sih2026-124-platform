"""Measure full-video transfer against a metadata/evidence edge payload."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def path_size(path: Path) -> int:
    if not path.exists():
        return 0
    if path.is_file():
        return path.stat().st_size
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def calculate_bandwidth(raw_video_bytes: int, payload_bytes: int) -> dict:
    measurement_valid = raw_video_bytes > 0 and payload_bytes > 0
    reduction = (
        (1 - (payload_bytes / raw_video_bytes)) * 100 if measurement_valid else 0.0
    )
    ratio = raw_video_bytes / payload_bytes if measurement_valid else 0.0
    return {
        "measurement_valid": measurement_valid,
        "raw_video_bytes": raw_video_bytes,
        "edge_payload_bytes": payload_bytes,
        "bytes_avoided": max(raw_video_bytes - payload_bytes, 0),
        "reduction_percent": round(reduction, 3),
        "compression_ratio": round(ratio, 3),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", default="input_video.mp4")
    parser.add_argument("--artifacts", default="artifacts/latest")
    parser.add_argument(
        "--payload",
        action="append",
        help="Payload file/directory relative to artifacts; may be repeated",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    video_path = Path(args.video)
    artifacts_dir = Path(args.artifacts)
    if not video_path.is_file():
        raise SystemExit(f"Raw video not found: {video_path}")

    payload_names = args.payload or ["events.csv", "metrics.json", "evidence"]
    payload_parts = {name: path_size(artifacts_dir / name) for name in payload_names}
    report = calculate_bandwidth(video_path.stat().st_size, sum(payload_parts.values()))
    report["payload_parts"] = payload_parts
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    report_path = artifacts_dir / "bandwidth_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
