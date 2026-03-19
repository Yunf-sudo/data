"""
Build masked classification datasets for grouped FengShui classes.

Each source image is detected with Grounding DINO, segmented with SAM,
and exported as a masked foreground crop for lightweight classification.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import urllib.request
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np
from PIL import Image
from tqdm import tqdm

from auto_label import (
    IMAGE_SUFFIXES,
    LOCAL_MODEL_DIR,
    PROMPT_SPECS,
    classwise_nms,
    load_image_rgb,
    load_model as load_grounding_dino_model,
    move_model_to_runtime,
    parse_single_class_detections,
    predict_prompt_batch,
    prepare_inference_image,
    setup_runtime,
)
from config import OUTPUT_DIR
from group_classifier_config import DEFAULT_MASKED_CLS_OUTPUT, GROUPS


DEFAULT_SAM_URL = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth"


@dataclass
class Sample:
    group: str
    label: str
    path: Path
    split: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build masked datasets for grouped classification.")
    parser.add_argument("--dataset-dir", default=OUTPUT_DIR, help="Raw dataset root.")
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_MASKED_CLS_OUTPUT,
        help="Root export directory for grouped masked classification data.",
    )
    parser.add_argument("--group", choices=sorted(GROUPS), help="Single group to export.")
    parser.add_argument("--all-groups", action="store_true", help="Export every configured group.")
    parser.add_argument("--limit", type=int, default=0, help="Optional per-class debug limit.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed used for split assignment.")
    parser.add_argument("--train-ratio", type=float, default=0.8, help="Train split ratio.")
    parser.add_argument("--val-ratio", type=float, default=0.1, help="Validation split ratio.")
    parser.add_argument("--model-id", default=str(LOCAL_MODEL_DIR), help="Grounding DINO local path or Hugging Face model id.")
    parser.add_argument("--sam-checkpoint", default="models/sam_vit_b_01ec64.pth", help="Path to the SAM checkpoint.")
    parser.add_argument("--sam-model-type", choices=("vit_b", "vit_l", "vit_h"), default="vit_b", help="SAM backbone type.")
    parser.add_argument("--sam-url", default=DEFAULT_SAM_URL, help="Checkpoint download URL.")
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto", help="Inference device.")
    parser.add_argument("--amp", choices=("auto", "bf16", "fp16", "off"), default="auto", help="AMP mode.")
    parser.add_argument("--batch-size", type=int, default=0, help="Grounding DINO batch size.")
    parser.add_argument("--inference-max-side", type=int, default=1024, help="Resize long side before detection.")
    parser.add_argument("--box-threshold", type=float, default=0.24, help="Grounding DINO box threshold.")
    parser.add_argument("--text-threshold", type=float, default=0.20, help="Grounding DINO text threshold.")
    parser.add_argument("--nms-iou", type=float, default=0.5, help="NMS IoU for duplicate boxes.")
    parser.add_argument("--min-box-area-ratio", type=float, default=0.002, help="Drop boxes smaller than this ratio.")
    parser.add_argument("--max-box-area-ratio", type=float, default=0.95, help="Drop boxes larger than this ratio.")
    parser.add_argument("--min-mask-area", type=int, default=256, help="Minimum accepted mask area in pixels.")
    parser.add_argument("--crop-padding", type=float, default=0.08, help="Extra padding around the object crop.")
    parser.add_argument("--output-size", type=int, default=320, help="Saved masked crop size.")
    parser.add_argument("--copy-review", action="store_true", help="Copy failed images into review folders.")
    parser.add_argument("--copy-mode", choices=("hardlink", "copy"), default="hardlink", help="Review copy mode.")
    parser.add_argument("--clean", action="store_true", help="Delete the export root before writing.")
    return parser.parse_args()


def ensure_dependencies() -> tuple[object, object]:
    try:
        import cv2  # type: ignore
        import torch  # type: ignore
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "Missing dependencies. Install with:\n"
            "pip install torch torchvision transformers opencv-python segment-anything tqdm"
        ) from exc
    return cv2, torch


def ensure_sam_checkpoint(checkpoint_path: Path, url: str) -> Path:
    if checkpoint_path.exists():
        return checkpoint_path

    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"[download] SAM checkpoint -> {checkpoint_path}")
    urllib.request.urlretrieve(url, checkpoint_path)
    return checkpoint_path


def selected_groups(args: argparse.Namespace) -> list[str]:
    if args.all_groups:
        return list(GROUPS)
    if args.group:
        return [args.group]
    raise SystemExit("Choose either --group <name> or --all-groups.")


def split_paths(paths: list[Path], train_ratio: float, val_ratio: float) -> list[str]:
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


def discover_samples(
    dataset_dir: Path,
    groups: list[str],
    limit: int,
    seed: int,
    train_ratio: float,
    val_ratio: float,
) -> list[Sample]:
    randomizer = random.Random(seed)
    samples: list[Sample] = []
    for group in groups:
        for label in GROUPS[group]:
            class_dir = dataset_dir / label
            if not class_dir.exists():
                raise FileNotFoundError(f"Missing class folder: {class_dir}")
            paths = [
                path
                for path in sorted(class_dir.iterdir())
                if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
            ]
            randomizer.shuffle(paths)
            if limit > 0:
                paths = paths[:limit]
            splits = split_paths(paths, train_ratio=train_ratio, val_ratio=val_ratio)
            for path, split in zip(paths, splits):
                samples.append(Sample(group=group, label=label, path=path, split=split))
    return samples


def prepare_dirs(output_root: Path, groups: list[str]) -> None:
    for group in groups:
        for split in ("train", "val", "test"):
            for label in GROUPS[group]:
                (output_root / group / split / label).mkdir(parents=True, exist_ok=True)
                (output_root / group / "masks" / split / label).mkdir(parents=True, exist_ok=True)
            (output_root / group / "review" / split).mkdir(parents=True, exist_ok=True)


def maybe_copy_review(source: Path, target: Path, mode: str, enabled: bool) -> None:
    if not enabled:
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    if mode == "hardlink":
        try:
            os.link(source, target)
            return
        except OSError:
            pass
    shutil.copy2(source, target)


def detection_area_ratio(box: tuple[float, float, float, float], width: int, height: int) -> float:
    x1, y1, x2, y2 = box
    return max(0.0, x2 - x1) * max(0.0, y2 - y1) / float(max(1, width * height))


def choose_detection(
    detections: list,
    image_size: tuple[int, int],
    min_area_ratio: float,
    max_area_ratio: float,
    nms_iou: float,
):
    width, height = image_size
    filtered = []
    for detection in detections:
        area_ratio = detection_area_ratio(detection.box, width, height)
        if area_ratio < min_area_ratio or area_ratio > max_area_ratio:
            continue
        filtered.append(detection)
    filtered = classwise_nms(filtered, iou_threshold=nms_iou)
    if not filtered:
        return None
    filtered.sort(key=lambda item: item.score, reverse=True)
    return filtered[0]


def chunked(items: list[Sample], batch_size: int) -> list[list[Sample]]:
    size = max(1, batch_size)
    return [items[index : index + size] for index in range(0, len(items), size)]


def detect_batch(model, processor, runtime, batch: list[Sample], args: argparse.Namespace) -> dict[Path, object | None]:
    images = [load_image_rgb(sample.path) for sample in batch]
    prepared = [prepare_inference_image(image, runtime.inference_max_side) for image in images]
    prompts = [str(PROMPT_SPECS[sample.label]["fallback"]) for sample in batch]
    target_sizes = [(image.size[1], image.size[0]) for image in images]
    results = predict_prompt_batch(
        model=model,
        processor=processor,
        runtime=runtime,
        images=prepared,
        prompts=prompts,
        target_sizes=target_sizes,
        box_threshold=args.box_threshold,
        text_threshold=args.text_threshold,
    )

    mapped: dict[Path, object | None] = {}
    for sample, image, result in zip(batch, images, results):
        detections = parse_single_class_detections(result, sample.label)
        mapped[sample.path] = choose_detection(
            detections=detections,
            image_size=image.size,
            min_area_ratio=args.min_box_area_ratio,
            max_area_ratio=args.max_box_area_ratio,
            nms_iou=args.nms_iou,
        )
    return mapped


def build_sam_predictor(args: argparse.Namespace, device: str):
    from segment_anything import SamPredictor, sam_model_registry  # type: ignore

    checkpoint = ensure_sam_checkpoint(Path(args.sam_checkpoint), args.sam_url)
    sam = sam_model_registry[args.sam_model_type](checkpoint=str(checkpoint))
    sam.to(device=device)
    return SamPredictor(sam)


def crop_with_padding(
    array: np.ndarray,
    box: tuple[int, int, int, int],
    padding_ratio: float,
) -> np.ndarray:
    y_max, x_max = array.shape[:2]
    x1, y1, x2, y2 = box
    width = max(1, x2 - x1)
    height = max(1, y2 - y1)
    pad_x = int(width * padding_ratio)
    pad_y = int(height * padding_ratio)
    left = max(0, x1 - pad_x)
    top = max(0, y1 - pad_y)
    right = min(x_max, x2 + pad_x)
    bottom = min(y_max, y2 + pad_y)
    return array[top:bottom, left:right]


def square_pad(image_array: np.ndarray, fill_value: int = 0) -> np.ndarray:
    height, width = image_array.shape[:2]
    side = max(height, width)
    canvas = np.full((side, side, 3), fill_value, dtype=np.uint8)
    top = (side - height) // 2
    left = (side - width) // 2
    canvas[top : top + height, left : left + width] = image_array
    return canvas


def predict_mask(predictor, image_array: np.ndarray, detection, min_mask_area: int, torch) -> np.ndarray | None:
    predictor.set_image(image_array)
    device = predictor.model.device
    boxes = torch.tensor([detection.box], dtype=torch.float32, device=device)
    transformed = predictor.transform.apply_boxes_torch(boxes, image_array.shape[:2])
    masks, scores, _ = predictor.predict_torch(
        point_coords=None,
        point_labels=None,
        boxes=transformed,
        multimask_output=True,
    )
    best_index = int(scores[0].argmax().item())
    mask = masks[0, best_index].detach().cpu().numpy().astype(np.uint8)
    predictor.reset_image()
    if int(mask.sum()) < min_mask_area:
        return None
    return mask


def export_masked_crop(
    image: Image.Image,
    detection,
    predictor,
    output_size: int,
    crop_padding: float,
    min_mask_area: int,
    torch,
) -> tuple[Image.Image, Image.Image | None, str]:
    image_array = np.asarray(image)
    mask = predict_mask(predictor, image_array, detection, min_mask_area=min_mask_area, torch=torch)

    if mask is None:
        x1, y1, x2, y2 = [int(round(value)) for value in detection.box]
        box_crop = crop_with_padding(image_array, (x1, y1, x2, y2), crop_padding)
        padded = square_pad(box_crop)
        export = Image.fromarray(padded).resize((output_size, output_size), Image.Resampling.BILINEAR)
        return export, None, "box_fallback"

    ys, xs = np.where(mask > 0)
    y1, y2 = int(ys.min()), int(ys.max()) + 1
    x1, x2 = int(xs.min()), int(xs.max()) + 1

    foreground = image_array.copy()
    foreground[mask == 0] = 0

    masked_crop = crop_with_padding(foreground, (x1, y1, x2, y2), crop_padding)
    mask_crop = crop_with_padding((mask * 255).astype(np.uint8), (x1, y1, x2, y2), crop_padding)

    padded_masked = square_pad(masked_crop)
    padded_mask = square_pad(np.stack([mask_crop] * 3, axis=-1))[:, :, 0]

    export = Image.fromarray(padded_masked).resize((output_size, output_size), Image.Resampling.BILINEAR)
    export_mask = Image.fromarray(padded_mask).resize((output_size, output_size), Image.Resampling.NEAREST)
    return export, export_mask, "sam_mask"


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


def write_metadata(group_root: Path, group: str) -> None:
    labels = GROUPS[group]
    (group_root / "labels.json").write_text(
        json.dumps({index: label for index, label in enumerate(labels)}, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    write_group_yaml(group_root, labels)


def main(args: argparse.Namespace) -> None:
    _, torch = ensure_dependencies()

    dataset_dir = Path(args.dataset_dir)
    output_root = Path(args.output_dir)
    groups = selected_groups(args)

    if args.clean and output_root.exists():
        shutil.rmtree(output_root)
    prepare_dirs(output_root, groups)

    samples = discover_samples(
        dataset_dir=dataset_dir,
        groups=groups,
        limit=args.limit,
        seed=args.seed,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
    )

    model, processor, _, local_files_only = load_grounding_dino_model(args.model_id)
    runtime_args = SimpleNamespace(
        device=args.device,
        amp=args.amp,
        batch_size=args.batch_size,
        max_batch_pixels=0,
        inference_max_side=args.inference_max_side,
    )
    runtime = setup_runtime(runtime_args, local_files_only)
    model = move_model_to_runtime(model, runtime)
    predictor = build_sam_predictor(args, runtime.device)

    summary: dict[str, dict[str, object]] = {}
    review_reasons: dict[str, Counter[str]] = {group: Counter() for group in groups}
    export_sources: dict[str, Counter[str]] = {group: Counter() for group in groups}
    split_counts: dict[str, Counter[str]] = {group: Counter() for group in groups}

    progress = tqdm(total=len(samples), desc="masked cls export")
    for group in groups:
        write_metadata(output_root / group, group)

    grouped_samples: dict[str, list[Sample]] = {group: [sample for sample in samples if sample.group == group] for group in groups}
    for group in groups:
        for sample in grouped_samples[group]:
            split_counts[group][sample.split] += 1

        for batch in chunked(grouped_samples[group], runtime.batch_size):
            detections_by_path = detect_batch(model, processor, runtime, batch, args)
            for sample in batch:
                image = load_image_rgb(sample.path)
                detection = detections_by_path.get(sample.path)
                if detection is None:
                    review_reasons[group]["empty_detection"] += 1
                    maybe_copy_review(
                        source=sample.path,
                        target=output_root / group / "review" / sample.split / sample.path.name,
                        mode=args.copy_mode,
                        enabled=args.copy_review,
                    )
                    progress.update(1)
                    continue

                crop_image, crop_mask, source_tag = export_masked_crop(
                    image=image,
                    detection=detection,
                    predictor=predictor,
                    output_size=args.output_size,
                    crop_padding=args.crop_padding,
                    min_mask_area=args.min_mask_area,
                    torch=torch,
                )
                export_sources[group][source_tag] += 1

                image_path = output_root / group / sample.split / sample.label / f"{sample.path.stem}.png"
                image_path.parent.mkdir(parents=True, exist_ok=True)
                crop_image.save(image_path)
                if crop_mask is not None:
                    mask_path = output_root / group / "masks" / sample.split / sample.label / f"{sample.path.stem}.png"
                    mask_path.parent.mkdir(parents=True, exist_ok=True)
                    crop_mask.save(mask_path)
                progress.update(1)

    progress.close()

    for group in groups:
        group_samples = grouped_samples[group]
        group_root = output_root / group
        summary[group] = {
            "source_images": len(group_samples),
            "class_counts": dict(Counter(sample.label for sample in group_samples)),
            "split_counts": dict(split_counts[group]),
            "review_reasons": dict(review_reasons[group]),
            "export_sources": dict(export_sources[group]),
            "exported_images": sum(1 for _ in group_root.glob("train/*/*.png"))
            + sum(1 for _ in group_root.glob("val/*/*.png"))
            + sum(1 for _ in group_root.glob("test/*/*.png")),
        }
        (group_root / "summary.json").write_text(
            json.dumps(summary[group], indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

    (output_root / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"[done] Grouped masked datasets written to: {output_root.resolve()}")


if __name__ == "__main__":
    main(parse_args())
