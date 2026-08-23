"""Atomic human-review decisions for presentation and active-learning workflows."""

from __future__ import annotations

import json
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

SAFE_KEY = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,255}$")
REVIEW_DECISIONS = frozenset(
    {"confirmed", "rejected_false_positive", "needs_field_inspection"}
)


def load_reviews(path: Path) -> dict[str, dict[str, Any]]:
    if not path.is_file():
        return {}
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("review file must contain an object")
    return {
        str(key): dict(record)
        for key, record in value.items()
        if isinstance(record, dict)
    }


def save_review(
    path: Path,
    *,
    module: str,
    event_id: str,
    decision: str,
    note: str = "",
    reviewed_at: datetime | None = None,
) -> dict[str, Any]:
    review_key = f"{module}:{event_id}"
    if not SAFE_KEY.fullmatch(review_key):
        raise ValueError("review module/event identifier is unsafe")
    if decision not in REVIEW_DECISIONS:
        raise ValueError("review decision is invalid")
    if len(note) > 500:
        raise ValueError("review note must not exceed 500 characters")
    if path.exists() and path.is_symlink():
        raise ValueError("review file must not be a symbolic link")

    reviews = load_reviews(path)
    record = {
        "review_key": review_key,
        "module": module,
        "event_id": event_id,
        "decision": decision,
        "note": note.strip(),
        "reviewed_at": (reviewed_at or datetime.now(UTC)).isoformat(),
        "active_learning_label": (
            "positive" if decision == "confirmed" else "hard_negative"
            if decision == "rejected_false_positive"
            else "unresolved"
        ),
    }
    reviews[review_key] = record
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(reviews, indent=2), encoding="utf-8")
    temporary.replace(path)
    return record


def review_summary(reviews: dict[str, dict[str, Any]]) -> dict[str, int]:
    return {
        decision: sum(record.get("decision") == decision for record in reviews.values())
        for decision in sorted(REVIEW_DECISIONS)
    }
