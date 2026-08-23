"""Confidence-aware ANPR for a selected incident clip.

This module deliberately requires a dedicated plate detector. A generic COCO
model must not be presented as a number-plate detector.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter
from pathlib import Path

INDIAN_PLATE_PATTERN = re.compile(r"^[A-Z]{2}[0-9]{1,2}[A-Z]{1,3}[0-9]{4}$")


def normalize_plate(text: str) -> str:
    return "".join(character for character in text.upper() if character.isalnum())


def is_plausible_indian_plate(text: str) -> bool:
    return bool(INDIAN_PLATE_PATTERN.fullmatch(normalize_plate(text)))


def consensus(candidates: list[tuple[str, float]]) -> tuple[str, float, int]:
    """Choose a multi-frame OCR result using vote count and mean confidence."""
    normalized = [(normalize_plate(text), confidence) for text, confidence in candidates]
    normalized = [(text, confidence) for text, confidence in normalized if text]
    if not normalized:
        return "", 0.0, 0
    counts = Counter(text for text, _ in normalized)
    winner = max(
        counts,
        key=lambda text: (
            counts[text],
            sum(score for value, score in normalized if value == text) / counts[text],
        ),
    )
    winner_scores = [score for text, score in normalized if text == winner]
    return winner, sum(winner_scores) / len(winner_scores), len(winner_scores)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Incident video clip")
    parser.add_argument("--plate-model", required=True, help="YOLO plate detector weights")
    parser.add_argument("--output-dir", default="artifacts/anpr")
    parser.add_argument("--confidence", type=float, default=0.25)
    parser.add_argument("--frame-skip", type=int, default=2)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        import cv2
        import easyocr
        from ultralytics import YOLO
    except ImportError as exc:
        raise SystemExit(
            "Install ANPR dependencies with: pip install -r requirements-anpr.txt"
        ) from exc

    input_path = Path(args.input)
    model_path = Path(args.plate_model)
    if not input_path.is_file():
        raise SystemExit(f"Incident clip not found: {input_path}")
    if not model_path.is_file():
        raise SystemExit(f"Plate detector not found: {model_path}")

    output_dir = Path(args.output_dir)
    crops_dir = output_dir / "plate_crops"
    crops_dir.mkdir(parents=True, exist_ok=True)
    model = YOLO(str(model_path))
    reader = easyocr.Reader(["en"], gpu=False)
    capture = cv2.VideoCapture(str(input_path))
    candidates: list[tuple[str, float]] = []
    detailed_candidates: list[dict] = []
    frame_index = 0

    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            if frame_index % args.frame_skip:
                frame_index += 1
                continue

            result = model.predict(frame, conf=args.confidence, verbose=False)[0]
            if result.boxes is not None:
                for box_index, box in enumerate(result.boxes):
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    detection_confidence = float(box.conf[0])
                    crop = frame[max(0, y1) : max(0, y2), max(0, x1) : max(0, x2)]
                    if not crop.size:
                        continue
                    crop_path = crops_dir / f"frame-{frame_index:06d}-{box_index}.jpg"
                    cv2.imwrite(str(crop_path), crop)
                    for _, text, ocr_confidence in reader.readtext(crop):
                        combined = math.sqrt(detection_confidence * float(ocr_confidence))
                        normalized = normalize_plate(text)
                        candidates.append((normalized, combined))
                        detailed_candidates.append(
                            {
                                "frame_index": frame_index,
                                "plate": normalized,
                                "combined_confidence": round(combined, 4),
                                "plausible_indian_format": is_plausible_indian_plate(normalized),
                                "crop": str(crop_path),
                            }
                        )
            frame_index += 1
    finally:
        capture.release()

    plate, confidence, votes = consensus(candidates)
    report = {
        "plate": plate,
        "confidence": round(confidence, 4),
        "votes": votes,
        "plausible_indian_format": is_plausible_indian_plate(plate),
        "requires_human_review": confidence < 0.8 or votes < 2,
        "candidates": detailed_candidates,
    }
    report_path = output_dir / "anpr_result.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
