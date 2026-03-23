from __future__ import annotations

import json
import math
from collections import Counter
from dataclasses import dataclass
from pathlib import Path


DATASET_DIR = Path("AAA_data")
TRAIN_LABELS_DIR = DATASET_DIR / "labels" / "train"
TRAIN_IMAGES_DIR = DATASET_DIR / "images" / "train"
BALANCED_LIST_PATH = DATASET_DIR / "train_balanced.txt"
BALANCED_YAML_PATH = DATASET_DIR / "data_balanced.yaml"
SUMMARY_PATH = DATASET_DIR / "balance_summary.json"


@dataclass(frozen=True)
class ImageRecord:
    image_path: Path
    class_ids: tuple[int, ...]


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


def quantile(sorted_values: list[int], q: float) -> float:
    if not sorted_values:
        return 0.0
    if len(sorted_values) == 1:
        return float(sorted_values[0])
    position = (len(sorted_values) - 1) * q
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(sorted_values[lower])
    ratio = position - lower
    return sorted_values[lower] * (1.0 - ratio) + sorted_values[upper] * ratio


def load_train_records() -> list[ImageRecord]:
    records: list[ImageRecord] = []
    for label_path in sorted(TRAIN_LABELS_DIR.rglob("*.txt")):
        relative = label_path.relative_to(TRAIN_LABELS_DIR).with_suffix(".jpg")
        image_path = TRAIN_IMAGES_DIR / relative
        if not image_path.exists():
            matches = list((TRAIN_IMAGES_DIR / relative.parent).glob(f"{relative.stem}.*"))
            if not matches:
                continue
            image_path = matches[0]

        class_ids: list[int] = []
        for line in label_path.read_text(encoding="utf-8", errors="ignore").splitlines():
            parts = line.split()
            if len(parts) != 5:
                continue
            class_ids.append(int(float(parts[0])))
        records.append(ImageRecord(image_path=image_path.resolve(), class_ids=tuple(sorted(set(class_ids)))))
    return records


def main() -> None:
    class_names = load_class_names(DATASET_DIR / "data.yaml")
    records = load_train_records()

    image_presence = Counter()
    for record in records:
        for class_id in record.class_ids:
            image_presence[class_id] += 1

    sorted_presence = sorted(image_presence.get(class_id, 0) for class_id in range(len(class_names)))
    q25 = quantile(sorted_presence, 0.25)
    q40 = quantile(sorted_presence, 0.40)
    median = quantile(sorted_presence, 0.50)

    severe_classes = {
        class_id for class_id in range(len(class_names))
        if image_presence.get(class_id, 0) <= q25
    }
    moderate_classes = {
        class_id for class_id in range(len(class_names))
        if q25 < image_presence.get(class_id, 0) <= q40
    }

    repeated_lines: list[str] = []
    effective_presence = Counter()
    repeat_histogram = Counter()
    image_repeat_rows: list[dict[str, object]] = []

    for record in records:
        repeat_factor = 1
        class_name_list = [class_names[class_id] for class_id in record.class_ids]

        if severe_classes.intersection(record.class_ids):
            repeat_factor = 3
        elif moderate_classes.intersection(record.class_ids):
            repeat_factor = 2

        repeat_histogram[repeat_factor] += 1
        repeated_lines.extend([record.image_path.as_posix()] * repeat_factor)

        for class_id in record.class_ids:
            effective_presence[class_id] += repeat_factor

        image_repeat_rows.append(
            {
                "image": record.image_path.as_posix(),
                "repeat_factor": repeat_factor,
                "classes": class_name_list,
            }
        )

    BALANCED_LIST_PATH.write_text("\n".join(repeated_lines) + "\n", encoding="utf-8")

    balanced_yaml = [
        f"path: {DATASET_DIR.resolve()}",
        f"train: {BALANCED_LIST_PATH.name}",
        "val: images/val",
        "test: images/test",
        "",
        f"nc: {len(class_names)}",
        "names:",
    ]
    for class_id, class_name in enumerate(class_names):
        balanced_yaml.append(f"  {class_id}: {class_name}")
    BALANCED_YAML_PATH.write_text("\n".join(balanced_yaml) + "\n", encoding="utf-8")

    summary = {
        "dataset_dir": str(DATASET_DIR.resolve()),
        "train_images": len(records),
        "balanced_train_lines": len(repeated_lines),
        "repeat_histogram": dict(sorted(repeat_histogram.items())),
        "presence_quantiles": {
            "q25": q25,
            "q40": q40,
            "median": median,
        },
        "severe_classes": [class_names[class_id] for class_id in sorted(severe_classes)],
        "moderate_classes": [class_names[class_id] for class_id in sorted(moderate_classes)],
        "original_image_presence": {
            class_names[class_id]: image_presence.get(class_id, 0)
            for class_id in range(len(class_names))
        },
        "effective_image_presence": {
            class_names[class_id]: effective_presence.get(class_id, 0)
            for class_id in range(len(class_names))
        },
    }
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print(f"[done] Balanced train manifest: {BALANCED_LIST_PATH.resolve()}")
    print(f"[done] Balanced yaml: {BALANCED_YAML_PATH.resolve()}")
    print(
        "[summary] "
        f"train_images={len(records)} "
        f"balanced_train_lines={len(repeated_lines)} "
        f"severe={len(severe_classes)} "
        f"moderate={len(moderate_classes)}"
    )


if __name__ == "__main__":
    main()
