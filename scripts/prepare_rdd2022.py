"""Convert extracted RDD2022 India Pascal VOC annotations to Ultralytics YOLO format."""

from __future__ import annotations

import argparse
import random
import shutil
import sys
import xml.etree.ElementTree as ET
from collections import Counter
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
MARKER_FILE = ".drishtipath-rdd2022-generated"

# Official RDD2022 damage codes mapped to friendly YOLO class names.
CLASS_CODES: dict[str, str] = {
    "D00": "longitudinal_crack",
    "D10": "transverse_crack",
    "D20": "alligator_crack",
    "D40": "pothole",
}

CLASS_NAMES = list(CLASS_CODES.values())


class ParseOutcome(str, Enum):
    POSITIVE = "positive"
    NEGATIVE = "negative"
    UNSUPPORTED = "unsupported"
    SKIPPED = "skipped"


@dataclass(frozen=True, slots=True)
class AnnotationRecord:
    """One image and its validated YOLO label lines."""

    image_path: Path
    label_lines: list[str]
    class_counts: Counter[str]


@dataclass(frozen=True, slots=True)
class ParsedAnnotation:
    outcome: ParseOutcome
    record: AnnotationRecord | None = None


@dataclass(frozen=True, slots=True)
class PreparationStats:
    train_images: int
    val_images: int
    train_counts: Counter[str]
    val_counts: Counter[str]
    positive_images: int
    negative_images: int
    unsupported_only: int
    malformed_or_missing: int


@dataclass(frozen=True, slots=True)
class ImageIndex:
    by_filename: dict[str, Path]
    by_stem: dict[str, Path]


def voc_bbox_to_yolo(
    xmin: float,
    ymin: float,
    xmax: float,
    ymax: float,
    image_width: int,
    image_height: int,
) -> tuple[float, float, float, float] | None:
    """Convert Pascal VOC box to normalized YOLO ``x_center y_center width height``."""
    if image_width <= 0 or image_height <= 0:
        return None

    xmin = max(0.0, min(float(xmin), float(image_width)))
    ymin = max(0.0, min(float(ymin), float(image_height)))
    xmax = max(0.0, min(float(xmax), float(image_width)))
    ymax = max(0.0, min(float(ymax), float(image_height)))

    if xmax <= xmin or ymax <= ymin:
        return None

    box_width = xmax - xmin
    box_height = ymax - ymin
    if box_width < 1.0 or box_height < 1.0:
        return None

    x_center = ((xmin + xmax) / 2.0) / image_width
    y_center = ((ymin + ymax) / 2.0) / image_height
    width = box_width / image_width
    height = box_height / image_height

    if not (0.0 <= x_center <= 1.0 and 0.0 <= y_center <= 1.0):
        return None
    if width <= 0.0 or height <= 0.0 or width > 1.0 or height > 1.0:
        return None

    return x_center, y_center, width, height


def build_image_index(source_dir: Path) -> ImageIndex:
    """Build a one-pass image lookup index from the source directory."""
    by_filename: dict[str, Path] = {}
    by_stem: dict[str, Path] = {}

    for path in source_dir.rglob("*"):
        if not path.is_file():
            continue
        if path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        resolved = path.resolve()
        by_filename.setdefault(path.name.lower(), resolved)
        by_stem.setdefault(path.stem.lower(), resolved)

    return ImageIndex(by_filename=by_filename, by_stem=by_stem)


def find_image_for_xml(xml_path: Path, root: ET.Element, image_index: ImageIndex) -> Path | None:
    """Locate the image referenced by a VOC annotation using the prebuilt index."""
    filename = root.findtext("filename")
    if filename:
        indexed = image_index.by_filename.get(filename.lower())
        if indexed is not None:
            return indexed
        sibling = xml_path.parent / filename
        if sibling.is_file():
            return sibling.resolve()
        nested = xml_path.parent.parent / "images" / filename
        if nested.is_file():
            return nested.resolve()

    indexed_stem = image_index.by_stem.get(xml_path.stem.lower())
    if indexed_stem is not None:
        return indexed_stem

    for directory in (xml_path.parent, xml_path.parent.parent / "images"):
        if not directory.is_dir():
            continue
        for ext in IMAGE_EXTENSIONS:
            candidate = directory / f"{xml_path.stem}{ext}"
            if candidate.is_file():
                return candidate.resolve()
    return None


def parse_voc_annotation(xml_path: Path, image_index: ImageIndex) -> ParsedAnnotation:
    """Parse one Pascal VOC XML file into a validated or classified outcome."""
    try:
        root = ET.parse(xml_path).getroot()
    except ET.ParseError:
        return ParsedAnnotation(ParseOutcome.SKIPPED)

    size = root.find("size")
    if size is None:
        return ParsedAnnotation(ParseOutcome.SKIPPED)

    width_text = size.findtext("width")
    height_text = size.findtext("height")
    if width_text is None or height_text is None:
        return ParsedAnnotation(ParseOutcome.SKIPPED)

    try:
        image_width = int(float(width_text))
        image_height = int(float(height_text))
    except ValueError:
        return ParsedAnnotation(ParseOutcome.SKIPPED)

    if image_width <= 0 or image_height <= 0:
        return ParsedAnnotation(ParseOutcome.SKIPPED)

    image_path = find_image_for_xml(xml_path, root, image_index)
    if image_path is None:
        return ParsedAnnotation(ParseOutcome.SKIPPED)

    objects = root.findall("object")
    if not objects:
        return ParsedAnnotation(
            ParseOutcome.NEGATIVE,
            AnnotationRecord(image_path=image_path, label_lines=[], class_counts=Counter()),
        )

    label_lines: list[str] = []
    class_counts: Counter[str] = Counter()
    supported_found = False
    unsupported_found = False

    for obj in objects:
        raw_name = obj.findtext("name")
        if raw_name is None:
            continue
        normalized_code = raw_name.strip().upper()
        class_name = CLASS_CODES.get(normalized_code)
        if class_name is None:
            unsupported_found = True
            continue

        supported_found = True
        bbox = obj.find("bndbox")
        if bbox is None:
            continue

        try:
            xmin = float(bbox.findtext("xmin", default="nan"))
            ymin = float(bbox.findtext("ymin", default="nan"))
            xmax = float(bbox.findtext("xmax", default="nan"))
            ymax = float(bbox.findtext("ymax", default="nan"))
        except ValueError:
            continue

        yolo_box = voc_bbox_to_yolo(xmin, ymin, xmax, ymax, image_width, image_height)
        if yolo_box is None:
            continue

        class_id = CLASS_NAMES.index(class_name)
        x_center, y_center, box_width, box_height = yolo_box
        label_lines.append(
            f"{class_id} {x_center:.6f} {y_center:.6f} {box_width:.6f} {box_height:.6f}"
        )
        class_counts[class_name] += 1

    if unsupported_found and not supported_found:
        return ParsedAnnotation(ParseOutcome.UNSUPPORTED)

    if not supported_found:
        return ParsedAnnotation(
            ParseOutcome.NEGATIVE,
            AnnotationRecord(image_path=image_path, label_lines=[], class_counts=Counter()),
        )

    return ParsedAnnotation(
        ParseOutcome.POSITIVE,
        AnnotationRecord(
            image_path=image_path,
            label_lines=label_lines,
            class_counts=class_counts,
        ),
    )


def deterministic_split(
    records: list[AnnotationRecord],
    validation_ratio: float,
    seed: int,
) -> tuple[list[AnnotationRecord], list[AnnotationRecord]]:
    """Split records reproducibly into train and validation sets."""
    if not 0.0 < validation_ratio < 1.0:
        raise ValueError("validation_ratio must be between 0 and 1")

    ordered = sorted(records, key=lambda item: item.image_path.name)
    rng = random.Random(seed)
    shuffled = ordered[:]
    rng.shuffle(shuffled)

    if len(shuffled) == 1:
        return shuffled, []

    val_count = max(1, int(round(len(shuffled) * validation_ratio)))
    if val_count >= len(shuffled):
        val_count = len(shuffled) - 1

    val_records = shuffled[:val_count]
    train_records = shuffled[val_count:]
    return train_records, val_records


def validate_output_path(source_dir: Path, output_dir: Path) -> None:
    """Reject unsafe output locations before any write or delete occurs."""
    source_resolved = source_dir.resolve()
    output_resolved = output_dir.resolve()

    if output_resolved == source_resolved:
        raise ValueError("Output directory cannot equal the source directory.")

    try:
        source_resolved.relative_to(output_resolved)
    except ValueError:
        pass
    else:
        raise ValueError("Output directory cannot be a parent of the source directory.")

    forbidden = {Path("/").resolve(), Path.home().resolve(), Path.cwd().resolve()}
    if output_resolved in forbidden:
        raise ValueError(
            f"Refusing to use unsafe output directory: {output_resolved}. "
            "Choose a dedicated dataset path."
        )


def prepare_output_directory(source_dir: Path, output_dir: Path, force: bool) -> None:
    """Validate and optionally clean a generated output directory."""
    validate_output_path(source_dir, output_dir)

    if not output_dir.exists():
        output_dir.mkdir(parents=True, exist_ok=True)
        return

    contents = list(output_dir.iterdir())
    if not contents:
        return

    marker_path = output_dir / MARKER_FILE
    if not force:
        raise ValueError(
            f"Output directory is nonempty: {output_dir}. "
            f"Use --force to regenerate a previously prepared dataset."
        )

    if not marker_path.is_file():
        raise ValueError(
            f"Output directory is nonempty and missing marker {MARKER_FILE}. "
            f"Refusing to delete arbitrary directory: {output_dir}"
        )

    shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)


def write_data_yaml(output_dir: Path) -> None:
    """Write Ultralytics dataset config for the prepared YOLO dataset."""
    yaml_path = output_dir / "data.yaml"
    content = (
        f"path: {output_dir.resolve()}\n"
        "train: images/train\n"
        "val: images/val\n"
        "names:\n"
    )
    for index, name in enumerate(CLASS_NAMES):
        content += f"  {index}: {name}\n"
    yaml_path.write_text(content, encoding="utf-8")


def write_marker(output_dir: Path) -> None:
    output_dir.joinpath(MARKER_FILE).write_text(
        "Generated by scripts/prepare_rdd2022.py\n",
        encoding="utf-8",
    )


def copy_split(
    records: list[AnnotationRecord],
    output_dir: Path,
    split_name: str,
) -> Counter[str]:
    """Copy images and labels into a split directory."""
    images_dir = output_dir / "images" / split_name
    labels_dir = output_dir / "labels" / split_name
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    class_counts: Counter[str] = Counter()
    for record in records:
        destination_image = images_dir / record.image_path.name
        destination_label = labels_dir / f"{record.image_path.stem}.txt"
        shutil.copy2(record.image_path, destination_image)
        label_text = "\n".join(record.label_lines)
        if label_text:
            label_text += "\n"
        destination_label.write_text(label_text, encoding="utf-8")
        class_counts.update(record.class_counts)
    return class_counts


def discover_xml_files(source_dir: Path) -> list[Path]:
    """Find Pascal VOC XML files under the source directory."""
    return sorted(path for path in source_dir.rglob("*.xml") if path.is_file())


def prepare_dataset(
    source_dir: Path,
    output_dir: Path,
    validation_ratio: float,
    seed: int,
    force: bool = False,
) -> PreparationStats:
    """Prepare YOLO dataset and return summary statistics."""
    xml_files = discover_xml_files(source_dir)
    if not xml_files:
        raise FileNotFoundError(f"No XML annotations found under: {source_dir}")

    image_index = build_image_index(source_dir)
    records: list[AnnotationRecord] = []
    positive_images = 0
    negative_images = 0
    unsupported_only = 0
    malformed_or_missing = 0

    for xml_path in xml_files:
        parsed = parse_voc_annotation(xml_path, image_index)
        if parsed.outcome is ParseOutcome.POSITIVE:
            assert parsed.record is not None
            records.append(parsed.record)
            positive_images += 1
        elif parsed.outcome is ParseOutcome.NEGATIVE:
            assert parsed.record is not None
            records.append(parsed.record)
            negative_images += 1
        elif parsed.outcome is ParseOutcome.UNSUPPORTED:
            unsupported_only += 1
        else:
            malformed_or_missing += 1

    if not records:
        raise ValueError(
            f"No valid RDD2022 annotations found under: {source_dir}. "
            f"Malformed or missing-image records: {malformed_or_missing}, "
            f"unsupported-only records: {unsupported_only}."
        )

    prepare_output_directory(source_dir, output_dir, force=force)
    train_records, val_records = deterministic_split(records, validation_ratio, seed)
    train_counts = copy_split(train_records, output_dir, "train")
    val_counts = copy_split(val_records, output_dir, "val")
    write_data_yaml(output_dir)
    write_marker(output_dir)

    return PreparationStats(
        train_images=len(train_records),
        val_images=len(val_records),
        train_counts=train_counts,
        val_counts=val_counts,
        positive_images=positive_images,
        negative_images=negative_images,
        unsupported_only=unsupported_only,
        malformed_or_missing=malformed_or_missing,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        required=True,
        help="Extracted RDD2022 India dataset directory containing Pascal VOC XML files",
    )
    parser.add_argument(
        "--output",
        default="datasets/rdd2022_india_yolo",
        help="Destination YOLO dataset directory",
    )
    parser.add_argument(
        "--validation-ratio",
        type=float,
        default=0.20,
        help="Fraction of images reserved for validation",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=26124,
        help="Deterministic split seed",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help=(
            "Regenerate output only when it already contains "
            f"{MARKER_FILE} from a previous preparation run"
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source_dir = Path(args.source)
    output_dir = Path(args.output)

    if not source_dir.is_dir():
        print(f"Error: source directory not found: {source_dir}", file=sys.stderr)
        raise SystemExit(1)

    try:
        stats = prepare_dataset(
            source_dir=source_dir,
            output_dir=output_dir,
            validation_ratio=args.validation_ratio,
            seed=args.seed,
            force=args.force,
        )
    except (FileNotFoundError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc

    print(f"Prepared dataset at: {output_dir.resolve()}")
    print(f"Training images: {stats.train_images}")
    print(f"Validation images: {stats.val_images}")
    print(f"Valid positive images: {stats.positive_images}")
    print(f"Valid negative images: {stats.negative_images}")
    print(f"Unsupported-only images: {stats.unsupported_only}")
    print(f"Malformed or missing-image records: {stats.malformed_or_missing}")
    print("Training annotations per class:")
    for class_name in CLASS_NAMES:
        print(f"  {class_name}: {stats.train_counts.get(class_name, 0)}")
    print("Validation annotations per class:")
    for class_name in CLASS_NAMES:
        print(f"  {class_name}: {stats.val_counts.get(class_name, 0)}")
    print(f"Dataset config: {(output_dir / 'data.yaml').resolve()}")


if __name__ == "__main__":
    main()
