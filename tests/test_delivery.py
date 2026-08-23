import json
from pathlib import Path

import pytest

from urban_intelligence.delivery import (
    DeliveryConfig,
    DeliveryResponse,
    EvidenceDeliveryClient,
)
from urban_intelligence.edge_runtime import EvidenceOutbox


def make_outbox(tmp_path: Path) -> EvidenceOutbox:
    outbox = EvidenceOutbox(tmp_path / "outbox")
    outbox.enqueue("event-1", {"class": "pothole", "vehicle_id": "bus-7"})
    return outbox


def test_successful_delivery_acknowledges_packet_and_uses_idempotency_key(
    tmp_path: Path,
) -> None:
    outbox = make_outbox(tmp_path)
    requests: list[tuple[str, bytes, dict[str, str]]] = []

    def transport(endpoint: str, body: bytes, headers, _timeout: float) -> DeliveryResponse:
        requests.append((endpoint, body, dict(headers)))
        return DeliveryResponse(202, "accepted")

    report = EvidenceDeliveryClient(
        DeliveryConfig("https://command.example/v1/evidence"),
        transport=transport,
        environment={"DRISHTIPATH_INGEST_TOKEN": "secret-token"},
    ).deliver(outbox)

    assert report["delivered"] == 1
    assert outbox.pending() == []
    assert (outbox.sent_dir / "event-1.json").is_file()
    assert requests[0][2]["Idempotency-Key"] == "event-1"
    assert json.loads(requests[0][1])["payload"]["vehicle_id"] == "bus-7"
    assert "secret-token" not in json.dumps(report)


def test_failed_delivery_stays_pending_and_records_attempt(tmp_path: Path) -> None:
    outbox = make_outbox(tmp_path)
    sleeps: list[float] = []
    client = EvidenceDeliveryClient(
        DeliveryConfig(
            "https://command.example/v1/evidence",
            max_attempts_per_run=2,
            base_backoff_s=0.25,
        ),
        transport=lambda *_args: DeliveryResponse(503, "offline"),
        environment={"DRISHTIPATH_INGEST_TOKEN": "token"},
        sleep=sleeps.append,
    )
    report = client.deliver(outbox)
    assert report["failed"] == 1
    assert report["transport_attempts"] == 2
    assert sleeps == [0.25]
    envelope = json.loads(outbox.pending()[0].read_text(encoding="utf-8"))
    assert envelope["attempts"] == 1
    assert "HTTP 503" in envelope["last_error"]


def test_delivery_requires_token_and_valid_limit(tmp_path: Path) -> None:
    outbox = make_outbox(tmp_path)
    client = EvidenceDeliveryClient(
        DeliveryConfig("http://127.0.0.1:8080/v1/evidence"),
        environment={},
    )
    with pytest.raises(RuntimeError, match="DRISHTIPATH_INGEST_TOKEN"):
        client.deliver(outbox)
    with pytest.raises(ValueError, match="positive"):
        client.deliver(outbox, limit=0)
