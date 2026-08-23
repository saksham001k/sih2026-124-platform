"""Tests for RDD2022 dataset preparation helpers."""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import pytest

from scripts.prepare_rdd2022 import (
    MARKER_FILE,
    AnnotationRecord,
    ParseOutcome,
    build_image_index,
    deterministic_split,
    parse_voc_annotation,
    prepare_dataset,
    prepare_output_directory,
    validate_output_path,
    voc_bbox_to_yolo,
)


def write_voc_xml(
    path: Path,
    *,
    filename: str,
    width: int,
    height: int,
    objects: list[tuple[str, tuple[int, int, int, int]]] | None = None,
) -> None:
    lines = [
        "<annotation>",
        f"  <filename>{filename}</filename>",
        "  <size>",
        f"    <width>{width}</width>",
        f"    <height>{height}</height>",
        "  </size>",
    ]
    if objects is not None:
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

    image_index = build_image_index(tmp_path)
    parsed = parse_voc_annotation(xml_path, image_index)
    assert parsed.outcome is ParseOutcome.POSITIVE
    assert parsed.record is not None
    assert parsed.record.image_path == image_path.resolve()
    assert len(parsed.record.label_lines) == 2
    assert parsed.record.class_counts["pothole"] == 1
    assert parsed.record.class_counts["longitudinal_crack"] == 1


def test_parse_voc_annotation_keeps_empty_negative_sample(tmp_path: Path) -> None:
    image_path = tmp_path / "negative.jpg"
    image_path.write_bytes(b"negative-image")
    xml_path = tmp_path / "negative.xml"
    write_voc_xml(xml_path, filename="negative.jpg", width=100, height=100, objects=[])

    parsed = parse_voc_annotation(xml_path, build_image_index(tmp_path))
    assert parsed.outcome is ParseOutcome.NEGATIVE
    assert parsed.record is not None
    assert parsed.record.label_lines == []


def test_parse_voc_annotation_skips_unsupported_only_sample(tmp_path: Path) -> None:
    image_path = tmp_path / "unsupported.jpg"
    image_path.write_bytes(b"unsupported-image")
    xml_path = tmp_path / "unsupported.xml"
    write_voc_xml(
        xml_path,
        filename="unsupported.jpg",
        width=100,
        height=100,
        objects=[("D50", (10, 10, 40, 40))],
    )

    parsed = parse_voc_annotation(xml_path, build_image_index(tmp_path))
    assert parsed.outcome is ParseOutcome.UNSUPPORTED
    assert parsed.record is None


def test_official_rdd2022_layout_finds_and_copies_image(tmp_path: Path) -> None:
    source = tmp_path / "India"
    image_path = source / "train" / "images" / "sample.jpg"
    xml_path = source / "train" / "annotations" / "xmls" / "sample.xml"
    image_path.parent.mkdir(parents=True)
    xml_path.parent.mkdir(parents=True)
    image_path.write_bytes(b"official-layout-image")
    write_voc_xml(
        xml_path,
        filename="sample.jpg",
        width=100,
        height=100,
        objects=[("D40", (10, 10, 40, 40))],
    )

    output = tmp_path / "output"
    stats = prepare_dataset(source, output, validation_ratio=0.2, seed=26124)

    assert stats.positive_images == 1
    assert stats.train_images == 1
    assert stats.val_images == 0
    copied_image = output / "images" / "train" / "sample.jpg"
    copied_label = output / "labels" / "train" / "sample.txt"
    assert copied_image.is_file()
    assert copied_image.read_bytes() == b"official-layout-image"
    assert copied_label.read_text(encoding="utf-8").startswith("3 ")
    assert (output / MARKER_FILE).is_file()


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


def test_prepare_dataset_skips_malformed_and_keeps_negative(tmp_path: Path) -> None:
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

    negative_image = source / "negative.jpg"
    negative_image.write_bytes(b"negative")
    write_voc_xml(
        source / "negative.xml",
        filename=negative_image.name,
        width=100,
        height=100,
        objects=[],
    )

    write_voc_xml(
        source / "broken.xml",
        filename="missing.jpg",
        width=100,
        height=100,
        objects=[("D40", (80, 80, 60, 60))],
    )

    output = tmp_path / "output"
    stats = prepare_dataset(source, output, validation_ratio=0.34, seed=26124)

    assert stats.positive_images == 3
    assert stats.negative_images == 1
    assert stats.malformed_or_missing == 1
    assert stats.train_counts["pothole"] + stats.val_counts["pothole"] == 3
    assert stats.train_images + stats.val_images == 4
    negative_labels = list((output / "labels").rglob("negative.txt"))
    assert len(negative_labels) == 1
    assert negative_labels[0].read_text(encoding="utf-8") == ""
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


def test_validate_output_path_rejects_unsafe_locations(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()

    with pytest.raises(ValueError, match="cannot equal the source directory"):
        validate_output_path(source, source)

    nested_output = source / "output"
    nested_output.mkdir()
    with pytest.raises(ValueError, match="cannot be a parent of the source directory"):
        validate_output_path(source, tmp_path)

    with pytest.raises(ValueError, match="unsafe output directory"):
        validate_output_path(source, Path.cwd())


def test_prepare_output_directory_refuses_unmarked_nonempty_dir(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    output = tmp_path / "output"
    output.mkdir()
    (output / "keep-me.txt").write_text("user data", encoding="utf-8")

    with pytest.raises(ValueError, match="missing marker"):
        prepare_output_directory(source, output, force=True)

    assert (output / "keep-me.txt").is_file()


def test_prepare_output_directory_allows_force_with_marker(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    output = tmp_path / "output"
    output.mkdir()
    (output / MARKER_FILE).write_text("generated", encoding="utf-8")
    (output / "old.txt").write_text("old", encoding="utf-8")

    prepare_output_directory(source, output, force=True)
    assert output.is_dir()
    assert not (output / "old.txt").exists()


def test_prepare_dataset_refuses_nonempty_output_without_force(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    image_path = source / "sample.jpg"
    image_path.write_bytes(b"image")
    write_voc_xml(
        source / "sample.xml",
        filename="sample.jpg",
        width=100,
        height=100,
        objects=[("D40", (10, 10, 40, 40))],
    )

    output = tmp_path / "output"
    output.mkdir()
    (output / MARKER_FILE).write_text("generated", encoding="utf-8")
    (output / "existing.txt").write_text("existing", encoding="utf-8")

    with pytest.raises(ValueError, match="Use --force"):
        prepare_dataset(source, output, validation_ratio=0.2, seed=26124, force=False)
