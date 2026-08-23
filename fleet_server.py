"""Minimal authenticated DrishtiPath central ingestion server."""

from __future__ import annotations

import argparse
import json
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from urban_intelligence.fleet import FleetIngestionService, FleetStore


def build_handler(service: FleetIngestionService):
    class Handler(BaseHTTPRequestHandler):
        server_version = "DrishtiPathFleet/1"

        def do_POST(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
            if self.path != "/v1/evidence":
                self._respond(404, {"error": "not_found"})
                return
            try:
                content_length = int(self.headers.get("Content-Length", "0"))
            except ValueError:
                self._respond(400, {"error": "invalid_content_length"})
                return
            body = self.rfile.read(min(content_length, 1024 * 1024 + 1))
            status, payload = service.handle(
                body,
                authorization=self.headers.get("Authorization", ""),
                idempotency_key=self.headers.get("Idempotency-Key", ""),
            )
            self._respond(status, payload)

        def do_GET(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
            if self.path == "/healthz":
                self._respond(200, {"status": "ok"})
            elif self.path == "/v1/summary":
                if not service.authorized(self.headers.get("Authorization", "")):
                    self._respond(401, {"error": "unauthorized"})
                else:
                    self._respond(200, service.store.summary())
            else:
                self._respond(404, {"error": "not_found"})

        def _respond(self, status: int, payload: dict[str, Any]) -> None:
            body = json.dumps(payload).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, format: str, *args: object) -> None:
            # Keep standard access logs but never include authorization headers or body.
            super().log_message(format, *args)

    return Handler


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--database", default="artifacts/fleet/fleet.db")
    parser.add_argument("--token-env", default="DRISHTIPATH_INGEST_TOKEN")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    token = os.environ.get(args.token_env, "").strip()
    if not token:
        raise SystemExit(f"Set {args.token_env} before starting fleet ingestion")
    service = FleetIngestionService(FleetStore(Path(args.database)), token)
    server = ThreadingHTTPServer((args.host, args.port), build_handler(service))
    print(f"Fleet ingestion listening on http://{args.host}:{args.port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
