"""Convert extracted RDD2022 India Pascal VOC annotations to Ultralytics YOLO format."""

from __future__ import annotations

import argparse
import random
import shutil
import sys
import xml.etree.ElementTree as ET
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

# Official RDD2022 damage codes mapped to friendly YOLO class names.
CLASS_CODES: dict[str, str] = {
    "D00": "longitudinal_crack",
    "D10": "transverse_crack",
    "D20": "alligator_crack",
    "D40": "pothole",
}

CLASS_NAMES = list(CLASS_CODES.values())


@dataclass(frozen=True, slots=True)
class AnnotationRecord:
    """One image and its validated YOLO label lines."""

    image_path: Path
    label_lines: list[str]
    class_counts: Counter[str]


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


def parse_voc_annotation(xml_path: Path) -> AnnotationRecord | None:
    """Parse one Pascal VOC XML file into validated YOLO label lines."""
    try:
        root = ET.parse(xml_path).getroot()
    except ET.ParseError:
        return None

    size = root.find("size")
    if size is None:
        return None

    width_text = size.findtext("width")
    height_text = size.findtext("height")
    if width_text is None or height_text is None:
        return None

    try:
        image_width = int(float(width_text))
        image_height = int(float(height_text))
    except ValueError:
        return None

    if image_width <= 0 or image_height <= 0:
        return None

    image_path = find_image_for_xml(xml_path, root)
    if image_path is None:
        return None

    label_lines: list[str] = []
    class_counts: Counter[str] = Counter()

    for obj in root.findall("object"):
        raw_name = obj.findtext("name")
        if raw_name is None:
            continue
        class_name = CLASS_CODES.get(raw_name.strip().upper())
        if class_name is None:
            continue

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

    if not label_lines:
        return None

    return AnnotationRecord(
        image_path=image_path,
        label_lines=label_lines,
        class_counts=class_counts,
    )


def find_image_for_xml(xml_path: Path, root: ET.Element) -> Path | None:
    """Locate the image referenced by a VOC annotation."""
    candidates: list[Path] = []
    filename = root.findtext("filename")
    if filename:
        candidates.append(xml_path.parent / filename)
        candidates.append(xml_path.parent.parent / "images" / filename)

    stem = xml_path.stem
    search_dirs = {xml_path.parent, xml_path.parent.parent / "images"}
    for directory in search_dirs:
        if not directory.is_dir():
            continue
        for ext in IMAGE_EXTENSIONS:
            candidate = directory / f"{stem}{ext}"
            if candidate.is_file():
                candidates.append(candidate)

    seen: set[Path] = set()
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        if resolved.is_file():
            return resolved
    return None


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
        destination_label.write_text("\n".join(record.label_lines) + "\n", encoding="utf-8")
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
) -> tuple[int, int, Counter[str], Counter[str], int]:
    """Prepare YOLO dataset and return summary statistics."""
    xml_files = discover_xml_files(source_dir)
    if not xml_files:
        raise FileNotFoundError(f"No XML annotations found under: {source_dir}")

    records: list[AnnotationRecord] = []
    skipped = 0
    for xml_path in xml_files:
        record = parse_voc_annotation(xml_path)
        if record is None:
            skipped += 1
            continue
        records.append(record)

    if not records:
        raise ValueError(
            f"No valid RDD2022 annotations found under: {source_dir}. "
            f"Skipped {skipped} malformed or unsupported files."
        )

    train_records, val_records = deterministic_split(records, validation_ratio, seed)

    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_counts = copy_split(train_records, output_dir, "train")
    val_counts = copy_split(val_records, output_dir, "val")
    write_data_yaml(output_dir)

    return len(train_records), len(val_records), train_counts, val_counts, skipped


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
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source_dir = Path(args.source)
    output_dir = Path(args.output)

    if not source_dir.is_dir():
        print(f"Error: source directory not found: {source_dir}", file=sys.stderr)
        raise SystemExit(1)

    try:
        train_images, val_images, train_counts, val_counts, skipped = prepare_dataset(
            source_dir=source_dir,
            output_dir=output_dir,
            validation_ratio=args.validation_ratio,
            seed=args.seed,
        )
    except (FileNotFoundError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc

    print(f"Prepared dataset at: {output_dir.resolve()}")
    print(f"Training images: {train_images}")
    print(f"Validation images: {val_images}")
    print(f"Skipped malformed annotations: {skipped}")
    print("Training annotations per class:")
    for class_name in CLASS_NAMES:
        print(f"  {class_name}: {train_counts.get(class_name, 0)}")
    print("Validation annotations per class:")
    for class_name in CLASS_NAMES:
        print(f"  {class_name}: {val_counts.get(class_name, 0)}")
    print(f"Dataset config: {(output_dir / 'data.yaml').resolve()}")


if __name__ == "__main__":
    main()
