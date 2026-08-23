import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from urban_intelligence.review import load_reviews, review_summary, save_review


def test_review_is_atomic_updatable_and_active_learning_ready(tmp_path: Path) -> None:
    path = tmp_path / "review_feedback.json"
    first = save_review(
        path,
        module="road",
        event_id="evt-001",
        decision="rejected_false_positive",
        note="zebra crossing paint",
        reviewed_at=datetime(2026, 8, 23, tzinfo=UTC),
    )
    assert first["active_learning_label"] == "hard_negative"
    assert not list(tmp_path.glob("*.tmp"))

    updated = save_review(
        path,
        module="road",
        event_id="evt-001",
        decision="confirmed",
    )
    assert updated["active_learning_label"] == "positive"
    reviews = load_reviews(path)
    assert len(reviews) == 1
    assert review_summary(reviews)["confirmed"] == 1
    assert json.loads(path.read_text())["road:evt-001"]["decision"] == "confirmed"


def test_review_rejects_unsafe_or_invalid_input(tmp_path: Path) -> None:
    path = tmp_path / "reviews.json"
    with pytest.raises(ValueError, match="unsafe"):
        save_review(
            path,
            module="road",
            event_id="../../escape",
            decision="confirmed",
        )
    with pytest.raises(ValueError, match="decision"):
        save_review(path, module="road", event_id="evt-1", decision="maybe")
    with pytest.raises(ValueError, match="500"):
        save_review(
            path,
            module="road",
            event_id="evt-1",
            decision="confirmed",
            note="x" * 501,
        )
