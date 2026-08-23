"""Authenticated delivery of disk-backed edge evidence over intermittent networks."""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from urban_intelligence.edge_runtime import EvidenceOutbox


@dataclass(frozen=True, slots=True)
class DeliveryResponse:
    status_code: int
    body: str = ""


@dataclass(frozen=True, slots=True)
class DeliveryConfig:
    endpoint: str
    token_env: str = "DRISHTIPATH_INGEST_TOKEN"
    timeout_s: float = 10.0
    max_attempts_per_run: int = 3
    base_backoff_s: float = 0.5

    def __post_init__(self) -> None:
        if not self.endpoint.startswith(("http://", "https://")):
            raise ValueError("delivery endpoint must use http or https")
        if self.timeout_s <= 0 or self.max_attempts_per_run < 1 or self.base_backoff_s < 0:
            raise ValueError("delivery timeout, attempts, and backoff are invalid")


Transport = Callable[[str, bytes, Mapping[str, str], float], DeliveryResponse]


def urllib_transport(
    endpoint: str,
    body: bytes,
    headers: Mapping[str, str],
    timeout_s: float,
) -> DeliveryResponse:
    request = urllib.request.Request(
        endpoint,
        data=body,
        headers=dict(headers),
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout_s) as response:
            return DeliveryResponse(
                status_code=int(response.status),
                body=response.read(4096).decode("utf-8", errors="replace"),
            )
    except urllib.error.HTTPError as exc:
        return DeliveryResponse(
            status_code=int(exc.code),
            body=exc.read(4096).decode("utf-8", errors="replace"),
        )


class EvidenceDeliveryClient:
    """Drain pending JSON packets only after a successful central acknowledgement."""

    def __init__(
        self,
        config: DeliveryConfig,
        *,
        transport: Transport = urllib_transport,
        environment: Mapping[str, str] | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.config = config
        self.transport = transport
        self.environment = os.environ if environment is None else environment
        self.sleep = sleep

    def deliver(self, outbox: EvidenceOutbox, *, limit: int | None = None) -> dict[str, Any]:
        if limit is not None and limit < 1:
            raise ValueError("delivery limit must be positive")
        token = self.environment.get(self.config.token_env, "").strip()
        if not token:
            raise RuntimeError(f"Set {self.config.token_env} before delivering evidence")

        delivered = 0
        failed = 0
        attempted = 0
        paths = outbox.pending()
        if limit is not None:
            paths = paths[:limit]
        for path in paths:
            envelope = self._read_envelope(path)
            event_id = str(envelope["event_id"])
            body = json.dumps(envelope, separators=(",", ":")).encode("utf-8")
            headers = {
                "Authorization": f"Bearer {token}",
                "Content-Type": "application/json",
                "Idempotency-Key": event_id,
                "User-Agent": "DrishtiPath-Edge/1",
            }
            acknowledged = False
            last_error = ""
            for retry in range(self.config.max_attempts_per_run):
                attempted += 1
                try:
                    response = self.transport(
                        self.config.endpoint,
                        body,
                        headers,
                        self.config.timeout_s,
                    )
                except (OSError, TimeoutError) as exc:
                    last_error = f"{type(exc).__name__}: {exc}"
                else:
                    if 200 <= response.status_code < 300:
                        acknowledged = True
                        break
                    last_error = f"HTTP {response.status_code}: {response.body[:200]}"
                if retry + 1 < self.config.max_attempts_per_run:
                    self.sleep(self.config.base_backoff_s * (2**retry))
            if acknowledged:
                outbox.acknowledge(event_id)
                delivered += 1
            else:
                outbox.record_failure(event_id, last_error or "delivery failed")
                failed += 1
        return {
            "pending_before": len(paths),
            "transport_attempts": attempted,
            "delivered": delivered,
            "failed": failed,
            "pending_after": len(outbox.pending()),
            "endpoint": self.config.endpoint,
        }

    @staticmethod
    def _read_envelope(path: Path) -> dict[str, Any]:
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"Invalid outbox packet: {path.name}") from exc
        if not isinstance(value, dict) or not value.get("event_id"):
            raise ValueError(f"Outbox packet has no event_id: {path.name}")
        return value
