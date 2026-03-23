from __future__ import annotations

import argparse
import csv
import hashlib
import json
import uuid
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageOps


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".avif"}


@dataclass(frozen=True)
class ImageRecord:
    image_path: Path
    relative_path: Path
    leaf_folder: str
    width: int
    height: int
    file_size: int


@dataclass(frozen=True)
class RemovalDecision:
    keep_image: Path
    remove_image: Path
    reason: str
    score: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Delete exact and training-insignificant near-duplicate images, then rename images and labels in order."
    )
    parser.add_argument("--dataset-dir", type=Path, default=Path("FengShui_Dataset/ai_full_review"))
    parser.add_argument("--images-subdir", default="images")
    parser.add_argument("--labels-subdir", default="labels_prelabel")
    parser.add_argument("--extra-subdirs", nargs="*", default=["annotated_previews"])
    parser.add_argument("--manifest-name", default="review_manifest.csv")
    parser.add_argument("--prefix-mode", choices=("folder", "parent_folder"), default="folder")
    parser.add_argument("--start-index", type=int, default=1)
    parser.add_argument("--padding", type=int, default=6)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--batch-size", type=int, default=128, help="Feature extraction batch size.")
    parser.add_argument("--feature-size", type=int, default=32, help="Resize edge length for near-duplicate features.")
    parser.add_argument(
        "--near-sim-threshold",
        type=float,
        default=0.995,
        help="Cosine similarity threshold for training-insignificant near duplicates within the same folder.",
    )
    parser.add_argument(
        "--disable-near-duplicates",
        action="store_true",
        help="Only delete exact duplicates.",
    )
    return parser.parse_args()


def iter_image_files(images_dir: Path) -> list[Path]:
    return sorted(
        path for path in images_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def load_image_rgb(path: Path) -> Image.Image:
    with Image.open(path) as image:
        return ImageOps.exif_transpose(image).convert("RGB")


def image_signature(path: Path) -> tuple[str, tuple[int, int]]:
    image = load_image_rgb(path)
    digest = hashlib.sha1()
    digest.update(f"{image.width}x{image.height}".encode("utf-8"))
    digest.update(image.tobytes())
    return digest.hexdigest(), image.size


def quality_tuple(record: ImageRecord) -> tuple[int, int, int, str]:
    return (record.width * record.height, record.width, record.file_size, str(record.image_path))


def choose_keep_record(a: ImageRecord, b: ImageRecord) -> ImageRecord:
    return a if quality_tuple(a) >= quality_tuple(b) else b


def mirrored_label_path(labels_dir: Path, images_dir: Path, image_path: Path) -> Path:
    return labels_dir / image_path.relative_to(images_dir).with_suffix(".txt")


def mirrored_extra_paths(extra_dir: Path, images_dir: Path, image_path: Path) -> list[Path]:
    relative = image_path.relative_to(images_dir)
    parent = extra_dir / relative.parent
    stem = relative.stem
    if not parent.exists():
        return []
    return sorted(path for path in parent.glob(f"{stem}.*") if path.is_file())


def collect_records(images_dir: Path) -> tuple[list[ImageRecord], list[dict[str, str]]]:
    records: list[ImageRecord] = []
    unreadable: list[dict[str, str]] = []
    for image_path in iter_image_files(images_dir):
        try:
            with Image.open(image_path) as image:
                image = ImageOps.exif_transpose(image)
                width, height = image.size
        except Exception as exc:
            unreadable.append({"image_path": str(image_path), "reason": str(exc)})
            continue
        records.append(
            ImageRecord(
                image_path=image_path,
                relative_path=image_path.relative_to(images_dir),
                leaf_folder=image_path.parent.name,
                width=width,
                height=height,
                file_size=image_path.stat().st_size,
            )
        )
    return records, unreadable


def build_exact_duplicate_decisions(records: list[ImageRecord]) -> list[RemovalDecision]:
    by_signature: dict[str, ImageRecord] = {}
    decisions: list[RemovalDecision] = []
    for record in records:
        signature, _ = image_signature(record.image_path)
        kept = by_signature.get(signature)
        if kept is None:
            by_signature[signature] = record
            continue
        winner = choose_keep_record(kept, record)
        loser = record if winner == kept else kept
        by_signature[signature] = winner
        decisions.append(
            RemovalDecision(
                keep_image=winner.image_path,
                remove_image=loser.image_path,
                reason="exact_duplicate",
                score=1.0,
            )
        )
    return decisions


def resolve_device(device_arg: str) -> str:
    import torch

    if device_arg == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available in this Python environment.")
    if device_arg == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device_arg


def build_near_duplicate_decisions(
    records: list[ImageRecord],
    feature_size: int,
    batch_size: int,
    similarity_threshold: float,
    device: str,
) -> list[RemovalDecision]:
    import torch
    import torch.nn.functional as F

    active_records = sorted(records, key=lambda item: (item.leaf_folder, str(item.image_path)))
    by_folder: dict[str, list[ImageRecord]] = {}
    for record in active_records:
        by_folder.setdefault(record.leaf_folder, []).append(record)

    decisions: list[RemovalDecision] = []
    for folder_name, folder_records in by_folder.items():
        if len(folder_records) < 2:
            continue

        features: list[torch.Tensor] = []
        for start in range(0, len(folder_records), batch_size):
            batch = folder_records[start : start + batch_size]
            batch_tensors: list[torch.Tensor] = []
            for record in batch:
                image = load_image_rgb(record.image_path)
                resized = image.resize((feature_size, feature_size), Image.Resampling.BILINEAR)
                tensor = torch.tensor(list(resized.getdata()), dtype=torch.float32).reshape(feature_size, feature_size, 3)
                tensor = tensor.permute(2, 0, 1).contiguous() / 255.0
                batch_tensors.append(tensor)
            stacked = torch.stack(batch_tensors, dim=0).to(device)
            flattened = stacked.flatten(start_dim=1)
            normalized = F.normalize(flattened, dim=1)
            features.append(normalized.cpu())

        embeddings = torch.cat(features, dim=0)
        sim_matrix = embeddings @ embeddings.T
        removed: set[int] = set()

        for i in range(len(folder_records)):
            if i in removed:
                continue
            base_record = folder_records[i]
            similar_group = [i]
            for j in range(i + 1, len(folder_records)):
                if j in removed:
                    continue
                score = float(sim_matrix[i, j])
                if score >= similarity_threshold:
                    similar_group.append(j)
            if len(similar_group) == 1:
                continue

            keep_idx = similar_group[0]
            keep_record = folder_records[keep_idx]
            for idx in similar_group[1:]:
                candidate = folder_records[idx]
                if choose_keep_record(keep_record, candidate) == candidate:
                    keep_record = candidate
                    keep_idx = idx

            for idx in similar_group:
                if idx == keep_idx:
                    continue
                removed.add(idx)
                score = float(sim_matrix[min(idx, keep_idx), max(idx, keep_idx)])
                decisions.append(
                    RemovalDecision(
                        keep_image=keep_record.image_path,
                        remove_image=folder_records[idx].image_path,
                        reason=f"near_duplicate:{folder_name}",
                        score=score,
                    )
                )

    return decisions


def deduplicate_decisions(decisions: list[RemovalDecision]) -> list[RemovalDecision]:
    final_by_remove: dict[Path, RemovalDecision] = {}
    for decision in decisions:
        current = final_by_remove.get(decision.remove_image)
        if current is None or decision.score > current.score:
            final_by_remove[decision.remove_image] = decision
    return sorted(final_by_remove.values(), key=lambda item: str(item.remove_image))


def apply_removals(
    decisions: list[RemovalDecision],
    images_dir: Path,
    labels_dir: Path,
    extra_dirs: list[Path],
    dry_run: bool,
) -> tuple[list[dict[str, object]], set[Path]]:
    removed_records: list[dict[str, object]] = []
    removed_images: set[Path] = set()
    for decision in decisions:
        image_path = decision.remove_image
        if image_path in removed_images:
            continue
        label_path = mirrored_label_path(labels_dir, images_dir, image_path)
        extra_paths: list[str] = []
        for extra_dir in extra_dirs:
            extra_paths.extend(str(path) for path in mirrored_extra_paths(extra_dir, images_dir, image_path))

        removed_records.append(
            {
                "keep_image": str(decision.keep_image),
                "remove_image": str(image_path),
                "remove_label": str(label_path) if label_path.exists() else "",
                "remove_extras": extra_paths,
                "reason": decision.reason,
                "score": decision.score,
            }
        )

        if not dry_run:
            if image_path.exists():
                image_path.unlink()
            if label_path.exists():
                label_path.unlink()
            for extra_dir in extra_dirs:
                for extra_path in mirrored_extra_paths(extra_dir, images_dir, image_path):
                    if extra_path.exists():
                        extra_path.unlink()
        removed_images.add(image_path)

    return removed_records, removed_images


def build_rename_map(
    images_dir: Path,
    labels_dir: Path,
    extra_dirs: list[Path],
    prefix_mode: str,
    start_index: int,
    padding: int,
) -> dict[Path, Path]:
    rename_map: dict[Path, Path] = {}
    directories = sorted({path.parent for path in iter_image_files(images_dir)})

    for directory in directories:
        files = sorted(path for path in directory.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS)
        if not files:
            continue
        prefix = directory.name if prefix_mode == "folder" else directory.parent.name
        for offset, image_path in enumerate(files, start=start_index):
            new_stem = f"{prefix}_{offset:0{padding}d}"
            new_image = image_path.with_name(f"{new_stem}{image_path.suffix.lower()}")
            rename_map[image_path] = new_image

            label_path = mirrored_label_path(labels_dir, images_dir, image_path)
            if label_path.exists():
                rename_map[label_path] = label_path.with_name(f"{new_stem}.txt")

            for extra_dir in extra_dirs:
                for extra_path in mirrored_extra_paths(extra_dir, images_dir, image_path):
                    rename_map[extra_path] = extra_path.with_name(f"{new_stem}{extra_path.suffix.lower()}")

    return rename_map


def apply_rename_map(rename_map: dict[Path, Path], dry_run: bool) -> list[dict[str, str]]:
    planned: list[dict[str, str]] = []
    if not rename_map:
        return planned

    temp_map: dict[Path, Path] = {}
    for old_path, new_path in rename_map.items():
        if old_path == new_path:
            continue
        planned.append({"old_path": str(old_path), "new_path": str(new_path)})
        if dry_run:
            continue
        temp_path = old_path.with_name(f"{old_path.name}.tmp_{uuid.uuid4().hex}")
        old_path.rename(temp_path)
        temp_map[temp_path] = new_path

    if dry_run:
        return planned

    for temp_path, new_path in temp_map.items():
        new_path.parent.mkdir(parents=True, exist_ok=True)
        temp_path.rename(new_path)

    return planned


def update_manifest(
    dataset_dir: Path,
    manifest_name: str,
    removed_images: set[Path],
    labels_dir: Path,
    images_dir: Path,
    rename_map: dict[Path, Path],
    dry_run: bool,
) -> bool:
    manifest_path = dataset_dir / manifest_name
    if not manifest_path.exists():
        return False

    removed_rel_paths: set[str] = set()
    for image_path in removed_images:
        removed_rel_paths.add(str(image_path.relative_to(dataset_dir)).replace("\\", "/"))
        removed_rel_paths.add(str(mirrored_label_path(labels_dir, images_dir, image_path).relative_to(dataset_dir)).replace("\\", "/"))

    rel_map: dict[str, str] = {}
    for old_path, new_path in rename_map.items():
        try:
            old_rel = old_path.relative_to(dataset_dir)
            new_rel = new_path.relative_to(dataset_dir)
        except ValueError:
            continue
        rel_map[str(old_rel).replace("\\", "/")] = str(new_rel).replace("\\", "/")

    with manifest_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        fieldnames = reader.fieldnames or []

    changed = False
    kept_rows: list[dict[str, str]] = []
    for row in rows:
        export_image = row.get("export_image", "")
        export_label = row.get("export_label", "")
        if export_image in removed_rel_paths or export_label in removed_rel_paths:
            changed = True
            continue
        if export_image in rel_map:
            row["export_image"] = rel_map[export_image]
            changed = True
        if export_label in rel_map:
            row["export_label"] = rel_map[export_label]
            changed = True
        kept_rows.append(row)

    if changed and not dry_run:
        with manifest_path.open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(kept_rows)

    return changed


def write_report(
    dataset_dir: Path,
    removed: list[dict[str, object]],
    renamed: list[dict[str, str]],
    unreadable: list[dict[str, str]],
    dry_run: bool,
) -> None:
    if dry_run:
        return
    report_dir = dataset_dir / "dedupe_reports"
    report_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        "removed_total": len(removed),
        "removed_exact_duplicates": sum(1 for item in removed if str(item["reason"]).startswith("exact_duplicate")),
        "removed_near_duplicates": sum(1 for item in removed if str(item["reason"]).startswith("near_duplicate")),
        "renamed_files": len(renamed),
        "unreadable_images": len(unreadable),
    }
    (report_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (report_dir / "removed_duplicates.json").write_text(json.dumps(removed, ensure_ascii=False, indent=2), encoding="utf-8")
    (report_dir / "renamed_files.json").write_text(json.dumps(renamed, ensure_ascii=False, indent=2), encoding="utf-8")
    (report_dir / "unreadable_images.json").write_text(json.dumps(unreadable, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    args = parse_args()
    dataset_dir = args.dataset_dir.resolve()
    images_dir = (dataset_dir / args.images_subdir).resolve()
    labels_dir = (dataset_dir / args.labels_subdir).resolve()
    extra_dirs = [path.resolve() for path in ((dataset_dir / extra) for extra in args.extra_subdirs) if path.exists()]

    if not images_dir.exists():
        raise FileNotFoundError(f"Images directory not found: {images_dir}")
    if not labels_dir.exists():
        raise FileNotFoundError(f"Labels directory not found: {labels_dir}")

    records, unreadable = collect_records(images_dir)

    exact_decisions = deduplicate_decisions(build_exact_duplicate_decisions(records))
    exact_removed, exact_removed_paths = apply_removals(exact_decisions, images_dir, labels_dir, extra_dirs, dry_run=args.dry_run)

    remaining_records = [record for record in records if record.image_path not in exact_removed_paths]

    near_decisions: list[RemovalDecision] = []
    if not args.disable_near_duplicates:
        device = resolve_device(args.device)
        near_decisions = deduplicate_decisions(
            build_near_duplicate_decisions(
                records=remaining_records,
                feature_size=args.feature_size,
                batch_size=args.batch_size,
                similarity_threshold=args.near_sim_threshold,
                device=device,
            )
        )
    near_removed, near_removed_paths = apply_removals(near_decisions, images_dir, labels_dir, extra_dirs, dry_run=args.dry_run)

    removed_images = exact_removed_paths | near_removed_paths
    rename_map = build_rename_map(
        images_dir=images_dir,
        labels_dir=labels_dir,
        extra_dirs=extra_dirs,
        prefix_mode=args.prefix_mode,
        start_index=args.start_index,
        padding=args.padding,
    )
    renamed = apply_rename_map(rename_map, dry_run=args.dry_run)
    manifest_updated = update_manifest(
        dataset_dir=dataset_dir,
        manifest_name=args.manifest_name,
        removed_images=removed_images,
        labels_dir=labels_dir,
        images_dir=images_dir,
        rename_map=rename_map,
        dry_run=args.dry_run,
    )

    removed = exact_removed + near_removed
    write_report(dataset_dir, removed, renamed, unreadable, dry_run=args.dry_run)

    print(f"Exact duplicates removed: {len(exact_removed)}")
    print(f"Near duplicates removed: {len(near_removed)}")
    print(f"Unreadable images skipped: {len(unreadable)}")
    print(f"Files renamed: {len(renamed)}")
    print(f"Manifest updated: {manifest_updated}")
    if args.dry_run:
        print("Dry run only. No files were changed.")
    else:
        print(f"Report directory: {dataset_dir / 'dedupe_reports'}")


if __name__ == "__main__":
    main()
