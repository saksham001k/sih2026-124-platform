"""Export fleet GIS, deficiency clusters, summary, and OD matrix from SQLite."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from urban_intelligence.fleet import FleetStore


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database", default="artifacts/fleet/fleet.db")
    parser.add_argument("--output-dir", default="artifacts/fleet/export")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    database = Path(args.database)
    if not database.is_file():
        raise SystemExit(f"Fleet database not found: {database}")
    store = FleetStore(database)
    outputs = store.export(Path(args.output_dir))
    print(json.dumps({key: str(value) for key, value in outputs.items()}, indent=2))


if __name__ == "__main__":
    main()
