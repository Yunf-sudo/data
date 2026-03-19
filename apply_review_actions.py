"""
Apply high-confidence relabel suggestions to a copied grouped dataset.

The script copies or hardlinks a grouped dataset root and then moves files
according to relabel_candidates.csv manifests.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Apply relabel suggestions to a grouped dataset copy.")
    parser.add_argument("--source-root", default="FengShui_GroupCls_Raw", help="Original grouped dataset root.")
    parser.add_argument("--review-root", default="review_manifests", help="Review manifest root.")
    parser.add_argument("--output-root", default="FengShui_GroupCls_CleanV1", help="Target cleaned dataset root.")
    parser.add_argument("--group", nargs="+", default=["plant", "door", "window"], help="Groups to apply.")
    parser.add_argument("--copy-mode", choices=("hardlink", "copy"), default="hardlink", help="How to copy the source dataset tree.")
    parser.add_argument("--clean", action="store_true", help="Delete the output root first if it exists.")
    return parser.parse_args()


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


def clone_group(source_group: Path, target_group: Path, mode: str) -> None:
    for path in source_group.rglob("*"):
        rel = path.relative_to(source_group)
        target = target_group / rel
        if path.is_dir():
            target.mkdir(parents=True, exist_ok=True)
        else:
            link_or_copy(path, target, mode)


def apply_group_actions(source_root: Path, review_root: Path, output_root: Path, group: str, mode: str) -> dict[str, object]:
    source_group = source_root / group
    target_group = output_root / group
    target_group.mkdir(parents=True, exist_ok=True)
    clone_group(source_group, target_group, mode)

    relabel_csv = review_root / group / "actions" / "relabel_candidates.csv"
    if not relabel_csv.exists():
        raise FileNotFoundError(f"Missing relabel_candidates.csv for group {group}: {relabel_csv}")

    moved = 0
    skipped = 0
    rows = []
    with relabel_csv.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            rows.append(row)

    for row in rows:
        original_path = Path(row["path"])
        try:
            rel = original_path.relative_to(source_root / group)
        except ValueError:
            skipped += 1
            continue

        parts = list(rel.parts)
        if len(parts) < 3:
            skipped += 1
            continue
        split = parts[0]
        true_label = parts[1]
        file_name = parts[-1]
        pred_label = row["pred_label"]

        src = target_group / split / true_label / file_name
        dst = target_group / split / pred_label / file_name
        dst.parent.mkdir(parents=True, exist_ok=True)
        if not src.exists():
            skipped += 1
            continue
        if dst.exists():
            skipped += 1
            continue
        shutil.move(str(src), str(dst))
        moved += 1

    summary = {
        "group": group,
        "moved": moved,
        "skipped": skipped,
        "relabel_csv": str(relabel_csv.resolve()),
    }
    (target_group / "cleaning_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return summary


def main(args: argparse.Namespace) -> None:
    source_root = Path(args.source_root).resolve()
    review_root = Path(args.review_root).resolve()
    output_root = Path(args.output_root).resolve()

    if args.clean and output_root.exists():
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    summary = {}
    for group in args.group:
        print(f"[apply] {group}")
        summary[group] = apply_group_actions(source_root, review_root, output_root, group, args.copy_mode)

    (output_root / "cleaning_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"[done] cleaned dataset written to: {output_root.resolve()}")


if __name__ == "__main__":
    main(parse_args())
