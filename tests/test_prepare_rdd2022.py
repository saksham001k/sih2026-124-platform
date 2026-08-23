"""Tests for RDD2022 dataset preparation helpers."""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import pytest

from scripts.prepare_rdd2022 import (
    AnnotationRecord,
    deterministic_split,
    parse_voc_annotation,
    prepare_dataset,
    voc_bbox_to_yolo,
)


def write_voc_xml(
    path: Path,
    *,
    filename: str,
    width: int,
    height: int,
    objects: list[tuple[str, tuple[int, int, int, int]]],
) -> None:
    lines = [
        "<annotation>",
        f"  <filename>{filename}</filename>",
        "  <size>",
        f"    <width>{width}</width>",
        f"    <height>{height}</height>",
        "  </size>",
    ]
    for name, (xmin, ymin, xmax, ymax) in objects:
        lines.extend(
            [
                "  <object>",
                f"    <name>{name}</name>",
                "    <bndbox>",
                f"      <xmin>{xmin}</xmin>",
                f"      <ymin>{ymin}</ymin>",
                f"      <xmax>{xmax}</xmax>",
                f"      <ymax>{ymax}</ymax>",
                "    </bndbox>",
                "  </object>",
            ]
        )
    lines.append("</annotation>")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_voc_bbox_to_yolo_converts_center_format() -> None:
    result = voc_bbox_to_yolo(10, 20, 110, 120, 200, 200)
    assert result is not None
    x_center, y_center, width, height = result
    assert x_center == pytest.approx(0.3)
    assert y_center == pytest.approx(0.35)
    assert width == pytest.approx(0.5)
    assert height == pytest.approx(0.5)


def test_voc_bbox_to_yolo_rejects_invalid_boxes() -> None:
    assert voc_bbox_to_yolo(50, 50, 50, 80, 100, 100) is None
    assert voc_bbox_to_yolo(10, 20, 110, 120, 0, 100) is None


def test_parse_voc_annotation_reads_supported_classes(tmp_path: Path) -> None:
    image_path = tmp_path / "sample.jpg"
    image_path.write_bytes(b"fake-image")
    xml_path = tmp_path / "sample.xml"
    write_voc_xml(
        xml_path,
        filename="sample.jpg",
        width=100,
        height=100,
        objects=[("D40", (10, 10, 40, 40)), ("D00", (50, 50, 80, 80))],
    )

    record = parse_voc_annotation(xml_path)
    assert record is not None
    assert record.image_path == image_path.resolve()
    assert len(record.label_lines) == 2
    assert record.class_counts["pothole"] == 1
    assert record.class_counts["longitudinal_crack"] == 1


def test_deterministic_split_is_reproducible() -> None:
    records = [
        AnnotationRecord(Path(f"{index}.jpg"), ["0 0.5 0.5 0.1 0.1"], Counter({"pothole": 1}))
        for index in range(10)
    ]

    train_a, val_a = deterministic_split(records, validation_ratio=0.2, seed=26124)
    train_b, val_b = deterministic_split(records, validation_ratio=0.2, seed=26124)

    assert [item.image_path.name for item in train_a] == [item.image_path.name for item in train_b]
    assert [item.image_path.name for item in val_a] == [item.image_path.name for item in val_b]
    assert len(train_a) == 8
    assert len(val_a) == 2


def test_prepare_dataset_skips_malformed_annotations(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    for index in range(3):
        image_path = source / f"img_{index}.jpg"
        image_path.write_bytes(b"image")
        write_voc_xml(
            source / f"img_{index}.xml",
            filename=image_path.name,
            width=100,
            height=100,
            objects=[("D40", (10, 10, 40, 40))],
        )

    write_voc_xml(
        source / "broken.xml",
        filename="missing.jpg",
        width=100,
        height=100,
        objects=[("D40", (80, 80, 60, 60))],
    )

    output = tmp_path / "output"
    train_images, val_images, train_counts, val_counts, skipped = prepare_dataset(
        source_dir=source,
        output_dir=output,
        validation_ratio=0.34,
        seed=26124,
    )

    assert train_images == 2
    assert val_images == 1
    assert skipped == 1
    assert train_counts["pothole"] == 2
    assert val_counts["pothole"] == 1
    assert (output / "data.yaml").is_file()


def test_prepare_dataset_fails_without_valid_annotations(tmp_path: Path) -> None:
    source = tmp_path / "empty_source"
    source.mkdir()
    with pytest.raises(FileNotFoundError, match="No XML annotations found"):
        prepare_dataset(source, tmp_path / "output", validation_ratio=0.2, seed=26124)

    invalid_source = tmp_path / "invalid_source"
    invalid_source.mkdir()
    write_voc_xml(
        invalid_source / "broken.xml",
        filename="missing.jpg",
        width=100,
        height=100,
        objects=[("D40", (80, 80, 60, 60))],
    )
    with pytest.raises(ValueError, match="No valid RDD2022 annotations"):
        prepare_dataset(invalid_source, tmp_path / "output2", validation_ratio=0.2, seed=26124)
