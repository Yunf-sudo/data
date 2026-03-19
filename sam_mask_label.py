"""
Build a segmentation dataset from selected FengShui classes with
Grounding DINO box prompts + SAM masks.

Outputs:
- images/{train,val,test}
- labels/{train,val,test}       # YOLO segmentation labels
- masks/{train,val,test}        # class-index PNG masks for quick review
- review/images                 # images that need manual checking
- review/reasons.json
- data.yaml
- summary.json

Example:
    python sam_mask_label.py --clean
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


DEFAULT_CLASSES = [
    "broad_leaf_live",
    "sharp_leaf_live",
    "fake_plant",
    "dining_table",
    "coffee_table",
]
DEFAULT_OUTPUT_DIR = "FengShui_SAM_5cls"
DEFAULT_SAM_URL = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth"


@dataclass
class Sample:
    label: str
    path: Path
    split: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate YOLO segmentation labels with Grounding DINO + SAM."
    )
    parser.add_argument("--dataset-dir", default=OUTPUT_DIR, help="Raw dataset root.")
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        help="Export directory for the segmentation dataset.",
    )
    parser.add_argument(
        "--classes",
        nargs="+",
        default=DEFAULT_CLASSES,
        help="Subset of classes to export.",
    )
    parser.add_argument(
        "--model-id",
        default=str(LOCAL_MODEL_DIR),
        help="Grounding DINO local path or Hugging Face model id.",
    )
    parser.add_argument(
        "--sam-checkpoint",
        default="models/sam_vit_b_01ec64.pth",
        help="Path to the SAM checkpoint.",
    )
    parser.add_argument(
        "--sam-model-type",
        choices=("vit_b", "vit_l", "vit_h"),
        default="vit_b",
        help="SAM backbone type.",
    )
    parser.add_argument(
        "--sam-url",
        default=DEFAULT_SAM_URL,
        help="Download URL used when the checkpoint is missing.",
    )
    parser.add_argument("--train-ratio", type=float, default=0.8, help="Train ratio.")
    parser.add_argument("--val-ratio", type=float, default=0.1, help="Validation ratio.")
    parser.add_argument("--seed", type=int, default=42, help="Split seed.")
    parser.add_argument("--limit", type=int, default=0, help="Optional debug limit.")
    parser.add_argument(
        "--device",
        choices=("auto", "cuda", "cpu"),
        default="auto",
        help="Inference device.",
    )
    parser.add_argument(
        "--amp",
        choices=("auto", "bf16", "fp16", "off"),
        default="auto",
        help="Mixed precision mode for Grounding DINO.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=0,
        help="Grounding DINO batch size. Default is auto-tuned.",
    )
    parser.add_argument(
        "--inference-max-side",
        type=int,
        default=1024,
        help="Resize long side for Grounding DINO inference.",
    )
    parser.add_argument(
        "--box-threshold",
        type=float,
        default=0.24,
        help="Grounding DINO box threshold.",
    )
    parser.add_argument(
        "--text-threshold",
        type=float,
        default=0.20,
        help="Grounding DINO text threshold.",
    )
    parser.add_argument(
        "--nms-iou",
        type=float,
        default=0.5,
        help="NMS IoU threshold for same-class detections.",
    )
    parser.add_argument(
        "--max-instances",
        type=int,
        default=5,
        help="Keep at most this many instances per image.",
    )
    parser.add_argument(
        "--min-box-area-ratio",
        type=float,
        default=0.002,
        help="Drop detections whose boxes are too small.",
    )
    parser.add_argument(
        "--max-box-area-ratio",
        type=float,
        default=0.95,
        help="Drop detections whose boxes are too large.",
    )
    parser.add_argument(
        "--min-mask-area",
        type=int,
        default=256,
        help="Drop SAM masks below this area in pixels.",
    )
    parser.add_argument(
        "--polygon-epsilon-ratio",
        type=float,
        default=0.002,
        help="Polygon simplification ratio based on contour perimeter.",
    )
    parser.add_argument(
        "--copy-mode",
        choices=("hardlink", "copy"),
        default="hardlink",
        help="How images are exported into the training folder.",
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Remove the output directory before exporting.",
    )
    return parser.parse_args()


def ensure_dependencies() -> tuple[object, object]:
    try:
        import cv2  # type: ignore
        import torch  # type: ignore
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "Missing dependencies. Install with:\n"
            "pip install torch torchvision transformers opencv-python segment-anything ultralytics tqdm"
        ) from exc
    return cv2, torch


def ensure_sam_checkpoint(checkpoint_path: Path, url: str) -> Path:
    if checkpoint_path.exists():
        return checkpoint_path

    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"[download] SAM checkpoint -> {checkpoint_path}")
    urllib.request.urlretrieve(url, checkpoint_path)
    return checkpoint_path


def discover_samples(
    dataset_dir: Path,
    labels: list[str],
    limit: int,
    seed: int,
    train_ratio: float,
    val_ratio: float,
) -> list[Sample]:
    randomizer = random.Random(seed)
    discovered: dict[str, list[Path]] = {}

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
        if limit > 0:
            paths = paths[:limit]
        discovered[label] = paths

    samples: list[Sample] = []
    for label in labels:
        paths = discovered[label]
        count = len(paths)
        if count == 1:
            train_cut = 1
            val_cut = 1
        elif count == 2:
            train_cut = 1
            val_cut = 2
        else:
            train_cut = max(1, int(count * train_ratio))
            val_count = max(1, int(count * val_ratio))
            if train_cut + val_count >= count:
                val_count = max(1, count - train_cut)
            val_cut = min(count, train_cut + val_count)
        for index, path in enumerate(paths):
            split = "train"
            if index >= val_cut:
                split = "test"
            elif index >= train_cut:
                split = "val"
            samples.append(Sample(label=label, path=path, split=split))
    return samples


def prepare_output_dirs(output_dir: Path) -> None:
    for group in ("images", "labels", "masks"):
        for split in ("train", "val", "test"):
            (output_dir / group / split).mkdir(parents=True, exist_ok=True)
    (output_dir / "review" / "images").mkdir(parents=True, exist_ok=True)


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


def detection_area_ratio(box: tuple[float, float, float, float], width: int, height: int) -> float:
    x1, y1, x2, y2 = box
    return max(0.0, x2 - x1) * max(0.0, y2 - y1) / float(max(1, width * height))


def filter_detections(
    detections: list,
    image_size: tuple[int, int],
    min_area_ratio: float,
    max_area_ratio: float,
    max_instances: int,
    nms_iou: float,
) -> list:
    width, height = image_size
    filtered = []
    for detection in detections:
        area_ratio = detection_area_ratio(detection.box, width, height)
        if area_ratio < min_area_ratio or area_ratio > max_area_ratio:
            continue
        filtered.append(detection)

    filtered = classwise_nms(filtered, iou_threshold=nms_iou)
    filtered.sort(key=lambda item: item.score, reverse=True)
    return filtered[:max_instances]


def chunked(items: list[Sample], batch_size: int) -> list[list[Sample]]:
    return [items[index : index + batch_size] for index in range(0, len(items), batch_size)]


def detect_batch(model, processor, runtime, batch: list[Sample], args: argparse.Namespace) -> dict[Path, list]:
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

    mapped: dict[Path, list] = {}
    for sample, image, result in zip(batch, images, results):
        detections = parse_single_class_detections(result, sample.label)
        detections = filter_detections(
            detections=detections,
            image_size=image.size,
            min_area_ratio=args.min_box_area_ratio,
            max_area_ratio=args.max_box_area_ratio,
            max_instances=args.max_instances,
            nms_iou=args.nms_iou,
        )
        mapped[sample.path] = detections
    return mapped


def box_to_polygon(box: tuple[float, float, float, float], width: int, height: int) -> list[float]:
    x1, y1, x2, y2 = box
    polygon = [
        x1 / width,
        y1 / height,
        x2 / width,
        y1 / height,
        x2 / width,
        y2 / height,
        x1 / width,
        y2 / height,
    ]
    return [max(0.0, min(1.0, value)) for value in polygon]


def mask_to_polygon(mask: np.ndarray, width: int, height: int, cv2, epsilon_ratio: float) -> list[float] | None:
    binary = (mask.astype(np.uint8) * 255).copy()
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    contour = max(contours, key=cv2.contourArea)
    perimeter = cv2.arcLength(contour, True)
    epsilon = max(1.0, perimeter * epsilon_ratio)
    approx = cv2.approxPolyDP(contour, epsilon, True)
    chosen = approx if len(approx) >= 3 else contour
    if len(chosen) < 3:
        return None

    polygon: list[float] = []
    for point in chosen.reshape(-1, 2):
        x = max(0.0, min(1.0, float(point[0]) / width))
        y = max(0.0, min(1.0, float(point[1]) / height))
        polygon.extend([x, y])

    return polygon if len(polygon) >= 6 else None


def export_mask_png(
    masks: list[np.ndarray],
    class_index: int,
    image_size: tuple[int, int],
    target_path: Path,
) -> None:
    width, height = image_size
    canvas = np.zeros((height, width), dtype=np.uint8)
    fill_value = class_index + 1
    for mask in masks:
        canvas[mask.astype(bool)] = fill_value
    Image.fromarray(canvas, mode="L").save(target_path)


def write_yolo_segmentation(label_path: Path, class_index: int, polygons: list[list[float]]) -> None:
    lines = []
    for polygon in polygons:
        coords = " ".join(f"{value:.6f}" for value in polygon)
        lines.append(f"{class_index} {coords}")
    label_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def write_data_yaml(output_dir: Path, labels: list[str]) -> None:
    lines = [
        f"path: {output_dir.resolve()}",
        "train: images/train",
        "val: images/val",
        "test: images/test",
        "",
        f"nc: {len(labels)}",
        "names:",
    ]
    for index, label in enumerate(labels):
        lines.append(f"  {index}: {label}")
    (output_dir / "data.yaml").write_text("\n".join(lines) + "\n", encoding="utf-8")


def save_review_image(sample: Sample, output_dir: Path, copy_mode: str) -> None:
    target = output_dir / "review" / "images" / sample.path.name
    link_or_copy(sample.path, target, copy_mode)


def build_sam_predictor(args: argparse.Namespace, torch, device: str):
    from segment_anything import SamPredictor, sam_model_registry  # type: ignore

    checkpoint = ensure_sam_checkpoint(Path(args.sam_checkpoint), args.sam_url)
    sam = sam_model_registry[args.sam_model_type](checkpoint=str(checkpoint))
    sam.to(device=device)
    return SamPredictor(sam)


def segment_instances(
    predictor,
    image: Image.Image,
    detections: list,
    min_mask_area: int,
    epsilon_ratio: float,
    torch,
    cv2,
) -> tuple[list[list[float]], list[np.ndarray]]:
    image_array = np.asarray(image)
    predictor.set_image(image_array)

    if not detections:
        return [], []

    device = predictor.model.device
    boxes = torch.tensor([detection.box for detection in detections], dtype=torch.float32, device=device)
    transformed_boxes = predictor.transform.apply_boxes_torch(boxes, image_array.shape[:2])
    masks, scores, _ = predictor.predict_torch(
        point_coords=None,
        point_labels=None,
        boxes=transformed_boxes,
        multimask_output=True,
    )

    polygons: list[list[float]] = []
    valid_masks: list[np.ndarray] = []
    best_indexes = scores.argmax(dim=1)
    width, height = image.size

    for index, detection in enumerate(detections):
        selected_mask = masks[index, best_indexes[index]].detach().cpu().numpy().astype(np.uint8)
        if int(selected_mask.sum()) < min_mask_area:
            polygon = box_to_polygon(detection.box, width, height)
            polygons.append(polygon)
            continue

        polygon = mask_to_polygon(selected_mask, width, height, cv2, epsilon_ratio)
        if polygon is None:
            polygon = box_to_polygon(detection.box, width, height)
        else:
            valid_masks.append(selected_mask)
        polygons.append(polygon)

    predictor.reset_image()
    return polygons, valid_masks


def build_summary(samples: list[Sample], exported_by_class: Counter, review_reasons: Counter) -> dict[str, object]:
    split_counts = Counter(sample.split for sample in samples)
    source_counts = Counter(sample.label for sample in samples)
    return {
        "source_images": len(samples),
        "source_class_counts": dict(source_counts),
        "split_counts": dict(split_counts),
        "exported_instance_images_by_class": dict(exported_by_class),
        "review_reasons": dict(review_reasons),
    }


def main(args: argparse.Namespace) -> None:
    cv2, torch = ensure_dependencies()

    dataset_dir = Path(args.dataset_dir)
    output_dir = Path(args.output_dir)
    target_labels = list(dict.fromkeys(args.classes))
    class_to_index = {label: index for index, label in enumerate(target_labels)}

    if args.clean and output_dir.exists():
        shutil.rmtree(output_dir)
    prepare_output_dirs(output_dir)

    samples = discover_samples(
        dataset_dir=dataset_dir,
        labels=target_labels,
        limit=args.limit,
        seed=args.seed,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
    )
    write_data_yaml(output_dir, target_labels)
    (output_dir / "label_map.json").write_text(
        json.dumps(class_to_index, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
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
    predictor = build_sam_predictor(args, torch, runtime.device)

    review_reasons: Counter[str] = Counter()
    exported_by_class: Counter[str] = Counter()
    progress = tqdm(total=len(samples), desc="SAM export")

    for label in target_labels:
        class_samples = [sample for sample in samples if sample.label == label]
        for batch in chunked(class_samples, runtime.batch_size):
            detections_by_path = detect_batch(model, processor, runtime, batch, args)
            for sample in batch:
                image = load_image_rgb(sample.path)
                detections = detections_by_path.get(sample.path, [])
                if not detections:
                    review_reasons["empty_detection"] += 1
                    save_review_image(sample, output_dir, args.copy_mode)
                    progress.update(1)
                    continue

                polygons, valid_masks = segment_instances(
                    predictor=predictor,
                    image=image,
                    detections=detections,
                    min_mask_area=args.min_mask_area,
                    epsilon_ratio=args.polygon_epsilon_ratio,
                    torch=torch,
                    cv2=cv2,
                )
                if not polygons:
                    review_reasons["empty_mask"] += 1
                    save_review_image(sample, output_dir, args.copy_mode)
                    progress.update(1)
                    continue

                image_target = output_dir / "images" / sample.split / sample.path.name
                label_target = output_dir / "labels" / sample.split / f"{sample.path.stem}.txt"
                mask_target = output_dir / "masks" / sample.split / f"{sample.path.stem}.png"

                link_or_copy(sample.path, image_target, args.copy_mode)
                write_yolo_segmentation(label_target, class_to_index[label], polygons)
                export_mask_png(valid_masks, class_to_index[label], image.size, mask_target)
                exported_by_class[label] += 1
                progress.update(1)

    progress.close()

    summary = build_summary(samples, exported_by_class, review_reasons)
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    (output_dir / "review" / "reasons.json").write_text(
        json.dumps(review_reasons, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    print(f"[done] Segmentation dataset written to: {output_dir.resolve()}")
    print(f"[done] data.yaml: {(output_dir / 'data.yaml').resolve()}")


if __name__ == "__main__":
    args = parse_args()
    main(args)
