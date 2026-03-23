from __future__ import annotations

import argparse
import csv
import json
import os
import random
import shutil
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".avif"}
SOURCE_LABEL_DIRNAME = "labels_prelabel"
REVIEW_DIRNAME = "review_conflicts"
OVERLAP_THRESHOLD = 0.98

CLASS_MAPPING = {
    "beam": "beam",
    "bed": "bed",
    "broad_leaf_live": "plant",
    "cabinet": "cabinet",
    "coffee_table": "table",
    "desk": "table",
    "dining_table": "table",
    "fake_plant": "plant",
    "floor_window": "window",
    "main_door": "door",
    "mirror": "mirror",
    "normal_window": "window",
    "room_door": "door",
    "sharp_leaf_live": "plant",
    "sink": "sink",
    "sofa": "sofa",
    "stairs": "stairs",
    "stove": "stove",
    "toilet": "toilet",
    "water_feature": "water_feature",
}

OUTPUT_CLASS_ORDER = [
    "beam",
    "bed",
    "plant",
    "cabinet",
    "table",
    "window",
    "door",
    "mirror",
    "sink",
    "sofa",
    "stairs",
    "stove",
    "toilet",
    "water_feature",
]


@dataclass(frozen=True)
class Box:
    original_index: int
    source_class: str
    target_class: str
    xc: float
    yc: float
    w: float
    h: float

    @property
    def area(self) -> float:
        return self.w * self.h

    def corners(self) -> tuple[float, float, float, float]:
        half_w = self.w / 2.0
        half_h = self.h / 2.0
        return (
            self.xc - half_w,
            self.yc - half_h,
            self.xc + half_w,
            self.yc + half_h,
        )


class DisjointSet:
    def __init__(self, size: int) -> None:
        self.parent = list(range(size))

    def find(self, value: int) -> int:
        if self.parent[value] != value:
            self.parent[value] = self.find(self.parent[value])
        return self.parent[value]

    def union(self, a: int, b: int) -> None:
        root_a = self.find(a)
        root_b = self.find(b)
        if root_a != root_b:
            self.parent[root_b] = root_a


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build AAA_data RT-DETR dataset with merged classes and conflict review export."
    )
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=Path("fengshui_dataset/ai_full_review"),
        help="Source review workspace root.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("AAA_data"),
        help="Output dataset root.",
    )
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--copy-mode", choices=("hardlink", "copy"), default="hardlink")
    parser.add_argument("--overlap-threshold", type=float, default=OVERLAP_THRESHOLD)
    parser.add_argument("--clean", action="store_true", help="Delete output directory before writing.")
    return parser.parse_args()


def iter_image_files(images_dir: Path) -> list[Path]:
    return sorted(
        path
        for path in images_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def split_names(paths: list[Path], train_ratio: float, val_ratio: float) -> list[str]:
    count = len(paths)
    if count == 1:
        return ["train"]
    if count == 2:
        return ["train", "val"]
    train_count = max(1, int(count * train_ratio))
    val_count = max(1, int(count * val_ratio))
    if train_count + val_count >= count:
        val_count = max(1, count - train_count)
    test_count = count - train_count - val_count
    if test_count <= 0:
        if train_count > val_count:
            train_count -= 1
        else:
            val_count -= 1
        test_count = 1
    return ["train"] * train_count + ["val"] * val_count + ["test"] * test_count


def link_or_copy(source: Path, target: Path, mode: str) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        return
    if mode == "hardlink":
        try:
            os.link(source, target)
            return
        except OSError:
            pass
    shutil.copy2(source, target)


def write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def load_boxes(label_path: Path, class_names: dict[int, str]) -> list[Box]:
    boxes: list[Box] = []
    if not label_path.exists():
        return boxes
    lines = label_path.read_text(encoding="utf-8", errors="ignore").splitlines()
    for index, line in enumerate(lines):
        parts = line.strip().split()
        if len(parts) != 5:
            continue
        class_id = int(float(parts[0]))
        source_class = class_names[class_id]
        target_class = CLASS_MAPPING[source_class]
        boxes.append(
            Box(
                original_index=index,
                source_class=source_class,
                target_class=target_class,
                xc=float(parts[1]),
                yc=float(parts[2]),
                w=float(parts[3]),
                h=float(parts[4]),
            )
        )
    return boxes


def overlap_ratio(a: Box, b: Box) -> float:
    ax1, ay1, ax2, ay2 = a.corners()
    bx1, by1, bx2, by2 = b.corners()
    inter_w = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    inter_h = max(0.0, min(ay2, by2) - max(ay1, by1))
    inter_area = inter_w * inter_h
    union_area = a.area + b.area - inter_area
    if union_area <= 0:
        return 0.0
    return inter_area / union_area


def canonical_box(boxes: list[Box]) -> Box:
    return max(
        boxes,
        key=lambda box: (
            box.area,
            -box.original_index,
        ),
    )


def merge_boxes(
    boxes: list[Box],
    overlap_threshold: float,
) -> tuple[list[Box], list[dict[str, object]]]:
    if not boxes:
        return [], []

    dsu = DisjointSet(len(boxes))
    conflicts: list[dict[str, object]] = []
    for i in range(len(boxes)):
        for j in range(i + 1, len(boxes)):
            score = overlap_ratio(boxes[i], boxes[j])
            if score < overlap_threshold:
                continue
            if boxes[i].target_class == boxes[j].target_class:
                dsu.union(i, j)
            else:
                conflicts.append(
                    {
                        "box_a_index": boxes[i].original_index,
                        "box_a_source_class": boxes[i].source_class,
                        "box_a_target_class": boxes[i].target_class,
                        "box_b_index": boxes[j].original_index,
                        "box_b_source_class": boxes[j].source_class,
                        "box_b_target_class": boxes[j].target_class,
                        "overlap_ratio": round(score, 6),
                    }
                )

    grouped: dict[int, list[Box]] = defaultdict(list)
    for index, box in enumerate(boxes):
        grouped[dsu.find(index)].append(box)

    merged = [canonical_box(group) for group in grouped.values()]
    merged.sort(key=lambda box: (OUTPUT_CLASS_ORDER.index(box.target_class), box.original_index))
    return merged, conflicts


def serialize_labels(boxes: list[Box], class_to_id: dict[str, int]) -> str:
    lines = []
    for box in boxes:
        class_id = class_to_id[box.target_class]
        lines.append(
            f"{class_id} {box.xc:.6f} {box.yc:.6f} {box.w:.6f} {box.h:.6f}"
        )
    return "\n".join(lines) + ("\n" if lines else "")


def load_source_names(data_yaml: Path) -> dict[int, str]:
    names: dict[int, str] = {}
    for raw_line in data_yaml.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = raw_line.strip()
        if not line or ":" not in line:
            continue
        key, value = line.split(":", 1)
        key = key.strip()
        value = value.strip()
        if key.isdigit():
            names[int(key)] = value
    if not names:
        raise RuntimeError(f"Unable to parse class names from {data_yaml}")
    return names


def build_split_plan(
    image_paths: list[Path],
    source_images_dir: Path,
    train_ratio: float,
    val_ratio: float,
    seed: int,
) -> dict[Path, str]:
    by_folder: dict[str, list[Path]] = defaultdict(list)
    for path in image_paths:
        relative = path.relative_to(source_images_dir)
        folder = relative.parts[0] if len(relative.parts) > 1 else "_root"
        by_folder[folder].append(path)

    randomizer = random.Random(seed)
    split_plan: dict[Path, str] = {}
    for folder, paths in sorted(by_folder.items()):
        shuffled = list(paths)
        randomizer.shuffle(shuffled)
        splits = split_names(shuffled, train_ratio=train_ratio, val_ratio=val_ratio)
        for path, split in zip(shuffled, splits):
            split_plan[path] = split
    return split_plan


def write_data_yaml(output_dir: Path) -> None:
    lines = [
        f"path: {output_dir.resolve()}",
        "train: images/train",
        "val: images/val",
        "test: images/test",
        "",
        f"nc: {len(OUTPUT_CLASS_ORDER)}",
        "names:",
    ]
    for index, name in enumerate(OUTPUT_CLASS_ORDER):
        lines.append(f"  {index}: {name}")
    write_text(output_dir / "data.yaml", "\n".join(lines) + "\n")


def main() -> None:
    args = parse_args()
    source_dir = args.source_dir.resolve()
    output_dir = args.output_dir.resolve()
    source_images_dir = source_dir / "images"
    source_labels_dir = source_dir / SOURCE_LABEL_DIRNAME
    source_names = load_source_names(source_dir / "data.yaml")
    target_name_to_id = {name: index for index, name in enumerate(OUTPUT_CLASS_ORDER)}

    if args.clean and output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    image_paths = iter_image_files(source_images_dir)
    split_plan = build_split_plan(
        image_paths=image_paths,
        source_images_dir=source_images_dir,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )

    split_counts: Counter[str] = Counter()
    review_folder_counts: Counter[str] = Counter()
    class_counts: Counter[str] = Counter()
    source_to_target_counts: Counter[str] = Counter()
    conflict_image_count = 0
    merged_box_drop_count = 0
    missing_label_count = 0
    split_rows: list[dict[str, object]] = []
    conflict_rows: list[dict[str, object]] = []

    for image_path in image_paths:
        relative_image = image_path.relative_to(source_images_dir)
        source_folder = relative_image.parts[0] if len(relative_image.parts) > 1 else "_root"
        label_path = source_labels_dir / relative_image.with_suffix(".txt")
        if not label_path.exists():
            missing_label_count += 1

        source_boxes = load_boxes(label_path, source_names)
        merged_boxes, conflicts = merge_boxes(source_boxes, args.overlap_threshold)
        merged_box_drop_count += max(0, len(source_boxes) - len(merged_boxes))

        target_labels_text = serialize_labels(merged_boxes, target_name_to_id)

        if conflicts:
            conflict_image_count += 1
            review_folder_counts[source_folder] += 1
            review_image_target = output_dir / REVIEW_DIRNAME / "images" / relative_image
            review_label_target = output_dir / REVIEW_DIRNAME / "labels" / relative_image.with_suffix(".txt")
            link_or_copy(image_path, review_image_target, args.copy_mode)
            write_text(review_label_target, target_labels_text)
            for conflict in conflicts:
                conflict_rows.append(
                    {
                        "image": str(relative_image).replace("\\", "/"),
                        "source_folder": source_folder,
                        "box_a_index": conflict["box_a_index"],
                        "box_a_source_class": conflict["box_a_source_class"],
                        "box_a_target_class": conflict["box_a_target_class"],
                        "box_b_index": conflict["box_b_index"],
                        "box_b_source_class": conflict["box_b_source_class"],
                        "box_b_target_class": conflict["box_b_target_class"],
                        "overlap_ratio": conflict["overlap_ratio"],
                    }
                )
            split_rows.append(
                {
                    "image": str(relative_image).replace("\\", "/"),
                    "source_folder": source_folder,
                    "status": REVIEW_DIRNAME,
                    "source_box_count": len(source_boxes),
                    "export_box_count": len(merged_boxes),
                }
            )
            continue

        split = split_plan[image_path]
        image_target = output_dir / "images" / split / relative_image
        label_target = output_dir / "labels" / split / relative_image.with_suffix(".txt")
        link_or_copy(image_path, image_target, args.copy_mode)
        write_text(label_target, target_labels_text)

        split_counts[split] += 1
        split_rows.append(
            {
                "image": str(relative_image).replace("\\", "/"),
                "source_folder": source_folder,
                "status": split,
                "source_box_count": len(source_boxes),
                "export_box_count": len(merged_boxes),
            }
        )

        seen_source_classes = {box.source_class for box in source_boxes}
        for name in seen_source_classes:
            source_to_target_counts[f"{name}->{CLASS_MAPPING[name]}"] += 1
        for box in merged_boxes:
            class_counts[box.target_class] += 1

    write_data_yaml(output_dir)
    write_text(
        output_dir / "labels.json",
        json.dumps(OUTPUT_CLASS_ORDER, indent=2, ensure_ascii=False) + "\n",
    )

    split_manifest_path = output_dir / "split_manifest.csv"
    with split_manifest_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["image", "source_folder", "status", "source_box_count", "export_box_count"],
        )
        writer.writeheader()
        writer.writerows(split_rows)

    conflicts_csv_path = output_dir / REVIEW_DIRNAME / "conflicts.csv"
    conflicts_csv_path.parent.mkdir(parents=True, exist_ok=True)
    with conflicts_csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "image",
                "source_folder",
                "box_a_index",
                "box_a_source_class",
                "box_a_target_class",
                "box_b_index",
                "box_b_source_class",
                "box_b_target_class",
                "overlap_ratio",
            ],
        )
        writer.writeheader()
        writer.writerows(conflict_rows)

    summary = {
        "source_dir": str(source_dir),
        "output_dir": str(output_dir),
        "image_total": len(image_paths),
        "exported_split_images": dict(split_counts),
        "review_conflict_images": conflict_image_count,
        "review_conflict_pairs": len(conflict_rows),
        "review_conflict_source_folders": dict(sorted(review_folder_counts.items())),
        "merged_box_drop_count": merged_box_drop_count,
        "missing_label_count": missing_label_count,
        "target_class_order": OUTPUT_CLASS_ORDER,
        "target_instance_counts": dict(sorted(class_counts.items())),
        "source_to_target_image_presence": dict(sorted(source_to_target_counts.items())),
        "overlap_threshold": args.overlap_threshold,
        "copy_mode": args.copy_mode,
        "split_seed": args.seed,
        "split_ratios": {
            "train": args.train_ratio,
            "val": args.val_ratio,
            "test": max(0.0, 1.0 - args.train_ratio - args.val_ratio),
        },
        "review_policy": "exclude_images_with_cross_class_overlap_at_or_above_threshold_from_train_val_test",
    }
    write_text(output_dir / "summary.json", json.dumps(summary, indent=2, ensure_ascii=False) + "\n")

    print(f"[done] AAA_data exported to: {output_dir}")
    print(
        "[summary] "
        f"images={len(image_paths)} "
        f"train={split_counts.get('train', 0)} "
        f"val={split_counts.get('val', 0)} "
        f"test={split_counts.get('test', 0)} "
        f"review={conflict_image_count}"
    )
    print(
        "[summary] "
        f"merged_box_drop_count={merged_box_drop_count} "
        f"conflict_pairs={len(conflict_rows)} "
        f"missing_label_count={missing_label_count}"
    )


if __name__ == "__main__":
    main()
