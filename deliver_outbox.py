"""Deliver pending DrishtiPath evidence packets to the central fleet endpoint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from urban_intelligence.delivery import DeliveryConfig, EvidenceDeliveryClient
from urban_intelligence.edge_runtime import EvidenceOutbox


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outbox", required=True, help="Mission outbox directory")
    parser.add_argument("--endpoint", required=True, help="Central /v1/evidence endpoint")
    parser.add_argument("--token-env", default="DRISHTIPATH_INGEST_TOKEN")
    parser.add_argument("--timeout-s", type=float, default=10.0)
    parser.add_argument("--attempts", type=int, default=3)
    parser.add_argument("--limit", type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    outbox_root = Path(args.outbox)
    if not outbox_root.is_dir():
        raise SystemExit(f"Outbox not found: {outbox_root}")
    try:
        report = EvidenceDeliveryClient(
            DeliveryConfig(
                endpoint=args.endpoint,
                token_env=args.token_env,
                timeout_s=args.timeout_s,
                max_attempts_per_run=args.attempts,
            )
        ).deliver(EvidenceOutbox(outbox_root), limit=args.limit)
    except (RuntimeError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
