from __future__ import annotations

import csv
import json
import math
from collections import Counter
from dataclasses import dataclass
from pathlib import Path


DATASET_DIR = Path("AAA_data")
TRAIN_LABELS_DIR = DATASET_DIR / "labels" / "train"
TRAIN_IMAGES_DIR = DATASET_DIR / "images" / "train"

TRAIN_CLEAN_LIST = DATASET_DIR / "train_clean.txt"
TRAIN_BALANCED_CLEAN_LIST = DATASET_DIR / "train_balanced_clean.txt"
DATA_CLEAN_YAML = DATASET_DIR / "data_clean.yaml"
DATA_BALANCED_CLEAN_YAML = DATASET_DIR / "data_balanced_clean.yaml"
SUMMARY_PATH = DATASET_DIR / "clean_summary.json"
NOISY_CSV_PATH = DATASET_DIR / "noisy_train_images.csv"


@dataclass(frozen=True)
class ImageStats:
    image_path: Path
    class_ids: tuple[int, ...]
    box_count: int
    distinct_class_count: int
    large_box_count: int
    noisy: bool


def load_class_names(data_yaml: Path) -> list[str]:
    names: list[str] = []
    for raw_line in data_yaml.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = raw_line.strip()
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        if key.strip().isdigit():
            names.append(value.strip())
    if not names:
        raise RuntimeError(f"Failed to parse class names from {data_yaml}")
    return names


def quantile(values: list[int], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return float(ordered[0])
    position = (len(ordered) - 1) * q
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(ordered[lower])
    ratio = position - lower
    return ordered[lower] * (1.0 - ratio) + ordered[upper] * ratio


def resolve_image_path(label_path: Path) -> Path:
    relative = label_path.relative_to(TRAIN_LABELS_DIR)
    candidate = TRAIN_IMAGES_DIR / relative.with_suffix(".jpg")
    if candidate.exists():
        return candidate.resolve()
    matches = list((TRAIN_IMAGES_DIR / relative.parent).glob(f"{relative.stem}.*"))
    if not matches:
        raise FileNotFoundError(f"Missing image for label: {label_path}")
    return matches[0].resolve()


def load_train_stats() -> tuple[list[ImageStats], dict[str, int]]:
    draft_rows: list[dict[str, object]] = []
    box_counts: list[int] = []
    class_counts: list[int] = []
    large_box_counts: list[int] = []

    for label_path in sorted(TRAIN_LABELS_DIR.rglob("*.txt")):
        class_ids: list[int] = []
        large_box_count = 0
        for line in label_path.read_text(encoding="utf-8", errors="ignore").splitlines():
            parts = line.split()
            if len(parts) != 5:
                continue
            cid = int(float(parts[0]))
            width = float(parts[3])
            height = float(parts[4])
            if width * height >= 0.5:
                large_box_count += 1
            class_ids.append(cid)

        distinct_ids = sorted(set(class_ids))
        box_count = len(class_ids)
        distinct_class_count = len(distinct_ids)
        box_counts.append(box_count)
        class_counts.append(distinct_class_count)
        large_box_counts.append(large_box_count)
        draft_rows.append(
            {
                "image_path": resolve_image_path(label_path),
                "class_ids": tuple(distinct_ids),
                "box_count": box_count,
                "distinct_class_count": distinct_class_count,
                "large_box_count": large_box_count,
            }
        )

    box_threshold = math.ceil(quantile(box_counts, 0.95))
    class_threshold = math.ceil(quantile(class_counts, 0.95))
    large_box_threshold = max(2, math.ceil(quantile(large_box_counts, 0.99)))

    stats: list[ImageStats] = []
    for row in draft_rows:
        noisy = (
            row["box_count"] > box_threshold
            or row["distinct_class_count"] > class_threshold
            or row["large_box_count"] > large_box_threshold
        )
        stats.append(
            ImageStats(
                image_path=row["image_path"],
                class_ids=row["class_ids"],
                box_count=row["box_count"],
                distinct_class_count=row["distinct_class_count"],
                large_box_count=row["large_box_count"],
                noisy=noisy,
            )
        )
    thresholds = {
        "box_count_gt": box_threshold,
        "distinct_classes_gt": class_threshold,
        "large_boxes_gt": large_box_threshold,
    }
    return stats, thresholds


def build_balanced_lines(records: list[ImageStats], class_names: list[str]) -> tuple[list[str], dict[str, object]]:
    image_presence = Counter()
    for record in records:
        for class_id in record.class_ids:
            image_presence[class_id] += 1

    sorted_presence = sorted(image_presence.get(class_id, 0) for class_id in range(len(class_names)))
    q25 = quantile(sorted_presence, 0.25)
    q40 = quantile(sorted_presence, 0.40)

    severe_classes = {
        class_id for class_id in range(len(class_names))
        if image_presence.get(class_id, 0) <= q25
    }
    moderate_classes = {
        class_id for class_id in range(len(class_names))
        if q25 < image_presence.get(class_id, 0) <= q40
    }

    repeated_lines: list[str] = []
    repeat_histogram = Counter()
    effective_presence = Counter()
    for record in records:
        repeat_factor = 1
        if severe_classes.intersection(record.class_ids):
            repeat_factor = 3
        elif moderate_classes.intersection(record.class_ids):
            repeat_factor = 2
        repeat_histogram[repeat_factor] += 1
        repeated_lines.extend([record.image_path.as_posix()] * repeat_factor)
        for class_id in record.class_ids:
            effective_presence[class_id] += repeat_factor

    meta = {
        "q25": q25,
        "q40": q40,
        "severe_classes": [class_names[class_id] for class_id in sorted(severe_classes)],
        "moderate_classes": [class_names[class_id] for class_id in sorted(moderate_classes)],
        "repeat_histogram": dict(sorted(repeat_histogram.items())),
        "effective_image_presence": {
            class_names[class_id]: effective_presence.get(class_id, 0)
            for class_id in range(len(class_names))
        },
        "original_image_presence": {
            class_names[class_id]: image_presence.get(class_id, 0)
            for class_id in range(len(class_names))
        },
    }
    return repeated_lines, meta


def write_yaml(path: Path, train_value: str, class_names: list[str]) -> None:
    lines = [
        f"path: {DATASET_DIR.resolve()}",
        f"train: {train_value}",
        "val: images/val",
        "test: images/test",
        "",
        f"nc: {len(class_names)}",
        "names:",
    ]
    for class_id, class_name in enumerate(class_names):
        lines.append(f"  {class_id}: {class_name}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    class_names = load_class_names(DATASET_DIR / "data.yaml")
    stats, thresholds = load_train_stats()

    clean_records = [record for record in stats if not record.noisy]
    removed_records = [record for record in stats if record.noisy]

    TRAIN_CLEAN_LIST.write_text(
        "\n".join(record.image_path.as_posix() for record in clean_records) + "\n",
        encoding="utf-8",
    )
    write_yaml(DATA_CLEAN_YAML, TRAIN_CLEAN_LIST.name, class_names)

    balanced_clean_lines, balance_meta = build_balanced_lines(clean_records, class_names)
    TRAIN_BALANCED_CLEAN_LIST.write_text("\n".join(balanced_clean_lines) + "\n", encoding="utf-8")
    write_yaml(DATA_BALANCED_CLEAN_YAML, TRAIN_BALANCED_CLEAN_LIST.name, class_names)

    with NOISY_CSV_PATH.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "image",
                "box_count",
                "distinct_class_count",
                "large_box_count",
                "classes",
            ],
        )
        writer.writeheader()
        for record in removed_records:
            writer.writerow(
                {
                    "image": record.image_path.as_posix(),
                    "box_count": record.box_count,
                    "distinct_class_count": record.distinct_class_count,
                    "large_box_count": record.large_box_count,
                    "classes": ";".join(class_names[class_id] for class_id in record.class_ids),
                }
            )

    clean_presence = Counter()
    removed_presence = Counter()
    removal_reasons = Counter()
    for record in clean_records:
        for class_id in record.class_ids:
            clean_presence[class_id] += 1
    for record in removed_records:
        if record.box_count > thresholds["box_count_gt"]:
            removal_reasons[f"box_count_gt_{thresholds['box_count_gt']}"] += 1
        if record.distinct_class_count > thresholds["distinct_classes_gt"]:
            removal_reasons[f"distinct_classes_gt_{thresholds['distinct_classes_gt']}"] += 1
        if record.large_box_count > thresholds["large_boxes_gt"]:
            removal_reasons[f"large_boxes_gt_{thresholds['large_boxes_gt']}"] += 1
        for class_id in record.class_ids:
            removed_presence[class_id] += 1

    summary = {
        "dataset_dir": str(DATASET_DIR.resolve()),
        "train_images_total": len(stats),
        "clean_train_images": len(clean_records),
        "removed_noisy_images": len(removed_records),
        "thresholds": thresholds,
        "removal_reasons": dict(sorted(removal_reasons.items())),
        "clean_image_presence": {
            class_names[class_id]: clean_presence.get(class_id, 0)
            for class_id in range(len(class_names))
        },
        "removed_image_presence": {
            class_names[class_id]: removed_presence.get(class_id, 0)
            for class_id in range(len(class_names))
        },
        "balanced_clean": balance_meta,
    }
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print(f"[done] Clean train list: {TRAIN_CLEAN_LIST.resolve()}")
    print(f"[done] Balanced clean train list: {TRAIN_BALANCED_CLEAN_LIST.resolve()}")
    print(f"[done] Noisy train csv: {NOISY_CSV_PATH.resolve()}")
    print(
        "[summary] "
        f"train_total={len(stats)} clean={len(clean_records)} removed={len(removed_records)} "
        f"balanced_clean_lines={len(balanced_clean_lines)}"
    )


if __name__ == "__main__":
    main()
