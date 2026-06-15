from __future__ import annotations

import argparse
import math
import os
import shutil
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path

import yaml
from sklearn.model_selection import StratifiedShuffleSplit


ONLY_VISIBLE_DIR = Path(__file__).resolve().parents[1]
PROJECT_DIR = ONLY_VISIBLE_DIR.parent
SOURCE_DIR = PROJECT_DIR / "ATR-UMOD" / "train"
DATA_DIR = ONLY_VISIBLE_DIR / "data"
IMAGES_DIR = DATA_DIR / "images"
LABELS_DIR = DATA_DIR / "labels"
SPLITS_DIR = ONLY_VISIBLE_DIR / "data_splits"
CONFIG_DIR = ONLY_VISIBLE_DIR / "config"

CLASSES = (
    "car",
    "suv",
    "van",
    "bus",
    "freight_car",
    "truck",
    "motorcycle",
    "trailer",
    "excavator",
    "crane",
    "tank_truck",
)
CLASS_MAP = {name: index for index, name in enumerate(CLASSES)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare ATR-UMOD visible-light images and a fixed 90/10 split."
    )
    parser.add_argument(
        "--copy-images",
        action="store_true",
        help="Copy RGB images instead of creating hard links.",
    )
    return parser.parse_args()


def ensure_directories() -> None:
    for directory in (IMAGES_DIR, LABELS_DIR, SPLITS_DIR, CONFIG_DIR):
        directory.mkdir(parents=True, exist_ok=True)


def link_or_copy_image(source: Path, destination: Path, copy_images: bool) -> str:
    if destination.exists():
        return "existing"
    if copy_images:
        shutil.copy2(source, destination)
        return "copied"
    try:
        os.link(source, destination)
        return "linked"
    except OSError:
        shutil.copy2(source, destination)
        return "copied"


def parse_visible_label(xml_path: Path) -> tuple[str, list[str], Counter[str]]:
    root = ET.parse(xml_path).getroot()
    width = float(root.findtext("size/width", default="0"))
    height = float(root.findtext("size/height", default="0"))
    if width <= 0 or height <= 0:
        raise ValueError(f"Invalid image size in {xml_path}")

    location = root.findtext("location", default="unknown")
    label_lines: list[str] = []
    stats: Counter[str] = Counter()
    for obj in root.findall("object"):
        name = obj.findtext("name", default="")
        if name not in CLASS_MAP:
            stats["unknown_class_objects"] += 1
            continue

        polygon = obj.find("polygon")
        if polygon is None:
            stats["missing_polygon_objects"] += 1
            continue

        try:
            coords = [
                float(polygon.findtext(f"{axis}{index}", default="nan"))
                for index in range(1, 5)
                for axis in ("x", "y")
            ]
        except ValueError:
            stats["invalid_coordinate_objects"] += 1
            continue

        if not all(math.isfinite(value) for value in coords):
            stats["invalid_coordinate_objects"] += 1
            continue

        normalized: list[float] = []
        clipped = False
        for index, value in enumerate(coords):
            limit = width if index % 2 == 0 else height
            normalized_value = value / limit
            bounded_value = min(max(normalized_value, 0.0), 1.0)
            clipped = clipped or bounded_value != normalized_value
            normalized.append(bounded_value)
        if clipped:
            stats["clipped_objects"] += 1

        points = " ".join(f"{value:.6f}" for value in normalized)
        label_lines.append(f"{CLASS_MAP[name]} {points}")
        stats["written_objects"] += 1

    return location, label_lines, stats


def prepare_samples(copy_images: bool) -> tuple[list[str], dict[str, str], Counter[str]]:
    source_images = sorted((SOURCE_DIR / "images").glob("*.jpg"))
    if not source_images:
        raise FileNotFoundError(f"No RGB images found under {SOURCE_DIR / 'images'}")

    sample_ids: list[str] = []
    locations: dict[str, str] = {}
    stats: Counter[str] = Counter()
    for index, source_image in enumerate(source_images, start=1):
        sample_id = source_image.stem
        xml_path = SOURCE_DIR / "labels" / f"{sample_id}.xml"
        if not xml_path.exists():
            raise FileNotFoundError(f"Missing visible-light label: {xml_path}")

        destination_image = IMAGES_DIR / source_image.name
        stats[link_or_copy_image(source_image, destination_image, copy_images)] += 1

        location, label_lines, label_stats = parse_visible_label(xml_path)
        (LABELS_DIR / f"{sample_id}.txt").write_text(
            "\n".join(label_lines) + ("\n" if label_lines else ""),
            encoding="utf-8",
        )
        stats.update(label_stats)
        if not label_lines:
            stats["empty_label_files"] += 1

        sample_ids.append(sample_id)
        locations[sample_id] = location
        if index % 1000 == 0 or index == len(source_images):
            print(f"Prepared {index}/{len(source_images)} RGB samples")

    return sample_ids, locations, stats


def generate_split_ids(
    sample_ids: list[str], locations: dict[str, str]
) -> tuple[list[str], list[str]]:
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.1, random_state=42)
    strata = [locations[sample_id] for sample_id in sample_ids]
    train_indices, val_indices = next(splitter.split(sample_ids, strata))
    return (
        [sample_ids[index] for index in train_indices],
        [sample_ids[index] for index in val_indices],
    )


def write_split(path: Path, sample_ids: list[str]) -> None:
    lines = [(IMAGES_DIR / f"{sample_id}.jpg").as_posix() for sample_id in sample_ids]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_config() -> None:
    config = {
        "path": DATA_DIR.as_posix(),
        "train": (SPLITS_DIR / "train.txt").as_posix(),
        "val": (SPLITS_DIR / "val.txt").as_posix(),
        "nc": len(CLASSES),
        "names": {index: name for index, name in enumerate(CLASSES)},
    }
    (CONFIG_DIR / "visible_90_10.yaml").write_text(
        yaml.safe_dump(config, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )


def write_splits(sample_ids: list[str], locations: dict[str, str]) -> None:
    train_ids, val_ids = generate_split_ids(sample_ids, locations)
    expected_ids = set(sample_ids)
    overlap = set(train_ids) & set(val_ids)
    covered_ids = set(train_ids) | set(val_ids)
    if overlap:
        raise ValueError(f"Train/validation overlap: {sorted(overlap)[:5]}")
    if covered_ids != expected_ids:
        raise ValueError("The 90/10 split does not cover every sample exactly once.")

    write_split(SPLITS_DIR / "train.txt", train_ids)
    write_split(SPLITS_DIR / "val.txt", val_ids)
    write_config()
    print(f"Fixed stratified 90/10 split: train={len(train_ids)}, val={len(val_ids)}")


def main() -> None:
    args = parse_args()
    ensure_directories()
    sample_ids, locations, stats = prepare_samples(args.copy_images)
    write_splits(sample_ids, locations)

    print("\nVisible-light dataset preparation complete.")
    for key, value in sorted(stats.items()):
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
