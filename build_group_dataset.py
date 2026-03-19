"""
Build grouped classification datasets directly from the existing labeled folders.

Example:
    python build_group_dataset.py --all-groups --clean
"""

from __future__ import annotations

import argparse
import json
import os
import random
import shutil
from collections import Counter
from pathlib import Path

from auto_label import IMAGE_SUFFIXES
from config import OUTPUT_DIR
from group_classifier_config import DEFAULT_MASKED_CLS_OUTPUT, GROUPS


DEFAULT_OUTPUT_DIR = "FengShui_GroupCls_Raw"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build grouped raw-image classification datasets.")
    parser.add_argument("--dataset-dir", default=OUTPUT_DIR, help="Source dataset root.")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR, help="Grouped classification dataset root.")
    parser.add_argument("--group", choices=sorted(GROUPS), help="Single group to export.")
    parser.add_argument("--all-groups", action="store_true", help="Export every configured group.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for split assignment.")
    parser.add_argument("--train-ratio", type=float, default=0.8, help="Train split ratio.")
    parser.add_argument("--val-ratio", type=float, default=0.1, help="Validation split ratio.")
    parser.add_argument("--copy-mode", choices=("hardlink", "copy"), default="hardlink", help="Export mode.")
    parser.add_argument("--clean", action="store_true", help="Delete the export directory before writing.")
    return parser.parse_args()


def selected_groups(args: argparse.Namespace) -> list[str]:
    if args.all_groups:
        return list(GROUPS)
    if args.group:
        return [args.group]
    raise SystemExit("Choose either --group <name> or --all-groups.")


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


def write_group_yaml(group_root: Path, labels: list[str]) -> None:
    lines = [
        f"path: {group_root.resolve()}",
        "train: train",
        "val: val",
        "test: test",
        "",
        f"nc: {len(labels)}",
        "names:",
    ]
    for index, label in enumerate(labels):
        lines.append(f"  {index}: {label}")
    (group_root / "data.yaml").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(args: argparse.Namespace) -> None:
    dataset_dir = Path(args.dataset_dir)
    output_dir = Path(args.output_dir)
    groups = selected_groups(args)
    randomizer = random.Random(args.seed)

    if args.clean and output_dir.exists():
        shutil.rmtree(output_dir)

    summary: dict[str, dict[str, object]] = {}
    for group in groups:
        group_root = output_dir / group
        labels = GROUPS[group]
        split_counts = Counter()
        class_counts = Counter()

        for split in ("train", "val", "test"):
            for label in labels:
                (group_root / split / label).mkdir(parents=True, exist_ok=True)

        for label in labels:
            class_dir = dataset_dir / label
            if not class_dir.exists():
                raise FileNotFoundError(f"Missing class folder: {class_dir}")
            paths = [
                path
                for path in sorted(class_dir.iterdir())
                if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
            ]
            randomizer.shuffle(paths)
            splits = split_names(paths, train_ratio=args.train_ratio, val_ratio=args.val_ratio)

            class_counts[label] = len(paths)
            for path, split in zip(paths, splits):
                target = group_root / split / label / path.name
                link_or_copy(path, target, args.copy_mode)
                split_counts[split] += 1

        write_group_yaml(group_root, labels)
        (group_root / "labels.json").write_text(
            json.dumps(labels, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        summary[group] = {
            "source_images": int(sum(class_counts.values())),
            "class_counts": dict(class_counts),
            "split_counts": dict(split_counts),
        }
        (group_root / "summary.json").write_text(
            json.dumps(summary[group], indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"[done] Grouped raw classification datasets written to: {output_dir.resolve()}")


if __name__ == "__main__":
    main(parse_args())
