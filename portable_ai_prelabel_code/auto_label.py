"""
Training-friendly auto-label pipeline for the FengShui dataset.

What this script does:
1. Runs Grounding DINO in multi-class mode on every collected image.
2. Falls back to the source folder's expected class if the main object is missed.
3. Writes YOLO labels and exports train/val/test splits.
4. Separates risky samples into a review queue instead of silently poisoning training.
5. Produces review manifests and data.yaml for RT-DETR / YOLO style training.

Recommended first run:
    pip install torch torchvision transformers pillow tqdm
    python auto_label.py --clean
"""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import io
import json
import os
import random
import shutil
import warnings
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Sequence

from PIL import Image, ImageOps
from tqdm import tqdm

from config import CATEGORIES, MIN_SHORT_SIDE, OUTPUT_DIR

warnings.filterwarnings("ignore")

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}
PROJECT_ROOT = Path(__file__).resolve().parent
LOCAL_MODEL_DIR = PROJECT_ROOT / "models" / "grounding-dino-base"
REMOTE_MODEL_ID = "IDEA-Research/grounding-dino-base"
DEFAULT_BOX_THRESHOLD = 0.28
DEFAULT_TEXT_THRESHOLD = 0.24
DEFAULT_FALLBACK_BOX_THRESHOLD = 0.22
DEFAULT_FALLBACK_TEXT_THRESHOLD = 0.18
DEFAULT_TRAIN_RATIO = 0.8
DEFAULT_VAL_RATIO = 0.1
DEFAULT_SEED = 42
DEFAULT_GPU_BATCH_SIZE = 1
DEFAULT_CPU_BATCH_SIZE = 1
DEFAULT_GPU_MAX_BATCH_PIXELS = 1_600_000
DEFAULT_INFERENCE_MAX_SIDE = 0
REVIEW_DIR_NAME = "review"
REVIEW_IMAGE_DIR = "images"
REVIEW_LABEL_DIR = "labels_prelabel"
NEEDS_REVIEW_FILE = "needs_review.txt"
DEFAULT_SCAN_WORKERS = max(2, min(8, (os.cpu_count() or 4) // 2))


PROMPT_SPECS: dict[str, dict[str, Sequence[str] | str]] = {
    "main_door": {
        "phrase": "front entry door",
        "fallback": (
            "front door . main entrance door . entry door . apartment entrance door"
        ),
        "aliases": (
            "front entry door",
            "front door",
            "entry door",
            "main entrance door",
        ),
    },
    "room_door": {
        "phrase": "interior room door",
        "fallback": (
            "interior door . room door . bedroom door . sliding door . pocket door"
        ),
        "aliases": (
            "interior room door",
            "interior door",
            "room door",
            "bedroom door",
            "sliding door",
            "pocket door",
        ),
    },
    "floor_window": {
        "phrase": "floor to ceiling window",
        "fallback": (
            "floor to ceiling window . full height window . panoramic window"
        ),
        "aliases": (
            "floor to ceiling window",
            "full height window",
            "panoramic window",
        ),
    },
    "normal_window": {
        "phrase": "standard window",
        "fallback": "window . casement window . small window . double hung window",
        "aliases": (
            "standard window",
            "window",
            "casement window",
            "small window",
            "double hung window",
        ),
    },
    "stove": {
        "phrase": "kitchen stove",
        "fallback": "stove . gas stove . cooktop . induction stove . range",
        "aliases": ("kitchen stove", "stove", "gas stove", "cooktop", "range"),
    },
    "sink": {
        "phrase": "sink",
        "fallback": "sink . kitchen sink . bathroom sink . wash basin",
        "aliases": ("sink", "kitchen sink", "bathroom sink", "wash basin"),
    },
    "stairs": {
        "phrase": "staircase",
        "fallback": "staircase . stairs . stairway . indoor steps",
        "aliases": ("staircase", "stairs", "stairway", "indoor steps"),
    },
    "water_feature": {
        "phrase": "indoor aquarium",
        "fallback": (
            "aquarium . fish tank . indoor fountain . indoor water feature"
        ),
        "aliases": (
            "indoor aquarium",
            "aquarium",
            "fish tank",
            "indoor fountain",
            "indoor water feature",
        ),
    },
    "broad_leaf_live": {
        "phrase": "broadleaf houseplant",
        "fallback": (
            "broadleaf houseplant . monstera plant . fiddle leaf fig . tropical plant"
        ),
        "aliases": (
            "broadleaf houseplant",
            "monstera plant",
            "fiddle leaf fig",
            "tropical plant",
        ),
    },
    "sharp_leaf_live": {
        "phrase": "spiky houseplant",
        "fallback": (
            "spiky houseplant . cactus . snake plant . aloe vera . succulent"
        ),
        "aliases": (
            "spiky houseplant",
            "cactus",
            "snake plant",
            "aloe vera",
            "succulent",
        ),
    },
    "fake_plant": {
        "phrase": "artificial plant",
        "fallback": (
            "artificial plant . fake plant . faux plant . plastic plant . dried plant"
        ),
        "aliases": (
            "artificial plant",
            "fake plant",
            "faux plant",
            "plastic plant",
            "dried plant",
        ),
    },
    "bed": {
        "phrase": "bed",
        "fallback": "bed . bedroom bed . mattress . headboard bed",
        "aliases": ("bed", "bedroom bed", "mattress", "headboard bed"),
    },
    "sofa": {
        "phrase": "sofa",
        "fallback": "sofa . couch . sectional sofa . loveseat",
        "aliases": ("sofa", "couch", "sectional sofa", "loveseat"),
    },
    "desk": {
        "phrase": "desk",
        "fallback": "desk . office desk . study desk . computer desk . work table",
        "aliases": ("desk", "office desk", "study desk", "computer desk"),
    },
    "dining_table": {
        "phrase": "dining table",
        "fallback": "dining table . dining room table . kitchen table",
        "aliases": ("dining table", "dining room table", "kitchen table"),
    },
    "coffee_table": {
        "phrase": "coffee table",
        "fallback": "coffee table . center table . low table",
        "aliases": ("coffee table", "center table", "low table"),
    },
    "mirror": {
        "phrase": "wall mirror",
        "fallback": "mirror . wall mirror . full length mirror . floor mirror",
        "aliases": (
            "wall mirror",
            "mirror",
            "full length mirror",
            "floor mirror",
        ),
    },
    "beam": {
        "phrase": "ceiling beam",
        "fallback": "ceiling beam . exposed beam . wooden beam . structural beam",
        "aliases": (
            "ceiling beam",
            "exposed beam",
            "wooden beam",
            "structural beam",
        ),
    },
    "toilet": {
        "phrase": "toilet",
        "fallback": "toilet . bathroom toilet . commode",
        "aliases": ("toilet", "bathroom toilet", "commode"),
    },
    "cabinet": {
        "phrase": "storage cabinet",
        "fallback": (
            "storage cabinet . cabinet . wardrobe . cupboard . bookshelf cabinet"
        ),
        "aliases": (
            "storage cabinet",
            "cabinet",
            "wardrobe",
            "cupboard",
            "bookshelf cabinet",
        ),
    },
}

FULL_PROMPT = " . ".join(PROMPT_SPECS[label]["phrase"] for label in CATEGORIES) + " ."
LABEL_TO_IDX = {label: cfg["idx"] for label, cfg in CATEGORIES.items()}
IDX_TO_LABEL = {cfg["idx"]: label for label, cfg in CATEGORIES.items()}
MANAGED_PATHS = (
    Path("images") / "train",
    Path("images") / "val",
    Path("images") / "test",
    Path("labels") / "train",
    Path("labels") / "val",
    Path("labels") / "test",
    Path(REVIEW_DIR_NAME),
    Path("data.yaml"),
    Path("split_manifest.csv"),
    Path("annotation_summary.json"),
)
CRITICAL_REVIEW_FLAGS = {
    "empty_detection",
    "expected_missing",
    "cross_class_duplicate",
}


@dataclass
class Detection:
    label: str
    score: float
    box: tuple[float, float, float, float]
    source: str = "full_prompt"


@dataclass
class ImageRecord:
    path: Path
    expected_class: str
    width: int
    height: int
    sha1: str
    ahash: str
    detections: list[Detection] = field(default_factory=list)
    flags: set[str] = field(default_factory=set)
    split: str | None = None

    @property
    def file_name(self) -> str:
        return self.path.name

    @property
    def review_required(self) -> bool:
        return any(flag in CRITICAL_REVIEW_FLAGS for flag in self.flags)

    @property
    def unique_detected_labels(self) -> list[str]:
        labels = {det.label for det in self.detections}
        return sorted(labels, key=lambda label: LABEL_TO_IDX[label])


@dataclass
class RuntimeConfig:
    device: str
    use_autocast: bool
    autocast_dtype: object | None
    local_files_only: bool
    batch_size: int
    max_batch_pixels: int
    inference_max_side: int
    gpu_memory_gb: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Auto-label FengShui_Dataset.")
    parser.add_argument("--dataset-dir", default=OUTPUT_DIR, help="Dataset root directory.")
    parser.add_argument(
        "--model-id",
        default=str(LOCAL_MODEL_DIR),
        help="Local model directory or Hugging Face model id.",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=DEFAULT_TRAIN_RATIO,
        help="Train split ratio.",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=DEFAULT_VAL_RATIO,
        help="Validation split ratio.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help="Random seed used for split assignment.",
    )
    parser.add_argument(
        "--box-threshold",
        type=float,
        default=DEFAULT_BOX_THRESHOLD,
        help="Threshold for multi-class detection boxes.",
    )
    parser.add_argument(
        "--text-threshold",
        type=float,
        default=DEFAULT_TEXT_THRESHOLD,
        help="Threshold for multi-class text matching.",
    )
    parser.add_argument(
        "--fallback-box-threshold",
        type=float,
        default=DEFAULT_FALLBACK_BOX_THRESHOLD,
        help="Threshold used when the expected class is missing.",
    )
    parser.add_argument(
        "--fallback-text-threshold",
        type=float,
        default=DEFAULT_FALLBACK_TEXT_THRESHOLD,
        help="Text threshold used when the expected class is missing.",
    )
    parser.add_argument(
        "--review-policy",
        choices=("exclude", "include"),
        default="exclude",
        help="Whether critical review samples are excluded from train/val/test.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Optional debug limit on number of source images.",
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Delete generated splits/review artifacts before exporting.",
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cuda", "cpu"),
        default="auto",
        help="Inference device. Default prefers CUDA when available.",
    )
    parser.add_argument(
        "--amp",
        choices=("auto", "bf16", "fp16", "off"),
        default="auto",
        help="Mixed precision mode for CUDA inference.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=0,
        help="Inference batch size. Default is auto-tuned by device.",
    )
    parser.add_argument(
        "--scan-workers",
        type=int,
        default=DEFAULT_SCAN_WORKERS,
        help="Worker count for source image metadata scanning.",
    )
    parser.add_argument(
        "--max-batch-pixels",
        type=int,
        default=0,
        help="Max total source pixels per inference batch. Default is auto-tuned on CUDA.",
    )
    parser.add_argument(
        "--inference-max-side",
        type=int,
        default=DEFAULT_INFERENCE_MAX_SIDE,
        help="Resize long image side to this value for inference only; labels stay in original coordinates.",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.train_ratio <= 0 or args.val_ratio < 0:
        raise ValueError("Split ratios must be positive.")
    if args.train_ratio + args.val_ratio >= 1:
        raise ValueError("train_ratio + val_ratio must be smaller than 1.")
    if args.batch_size < 0:
        raise ValueError("batch_size must be >= 0.")
    if args.scan_workers < 1:
        raise ValueError("scan_workers must be >= 1.")
    if args.max_batch_pixels < 0:
        raise ValueError("max_batch_pixels must be >= 0.")
    if args.inference_max_side < 0:
        raise ValueError("inference_max_side must be >= 0.")


def resolve_model_source(model_id: str) -> str:
    candidate = Path(model_id).expanduser()
    if candidate.exists():
        return str(candidate.resolve())

    env_model_dir = Path(
        os.environ.get("GDINO_MODEL_DIR", str(LOCAL_MODEL_DIR))
    ).expanduser()
    if env_model_dir.exists():
        return str(env_model_dir.resolve())

    looks_like_local_path = (
        candidate.is_absolute()
        or model_id.startswith(".")
        or model_id.startswith("\\")
        or "\\" in model_id
    )
    if not looks_like_local_path and model_id.count("/") == 1:
        return model_id

    raise FileNotFoundError(
        "Grounding DINO local model directory not found.\n"
        f"Expected default location: {LOCAL_MODEL_DIR}\n"
        "You can either:\n"
        f"1. Download the model into that folder, or\n"
        f"2. Pass --model-id {REMOTE_MODEL_ID}, or\n"
        "3. Set environment variable GDINO_MODEL_DIR to your local model path."
    )


def normalize_text(text: str) -> str:
    keep = []
    for char in text.lower():
        keep.append(char if char.isalnum() else " ")
    return " ".join("".join(keep).split())


def build_alias_lookup() -> list[tuple[str, tuple[str, ...]]]:
    order = [
        "floor_window",
        "main_door",
        "room_door",
        "coffee_table",
        "dining_table",
        "sharp_leaf_live",
        "broad_leaf_live",
        "fake_plant",
        "water_feature",
        "normal_window",
        "stove",
        "sink",
        "stairs",
        "bed",
        "sofa",
        "desk",
        "mirror",
        "beam",
        "toilet",
        "cabinet",
    ]
    lookup: list[tuple[str, tuple[str, ...]]] = []
    for label in order:
        aliases = tuple(normalize_text(alias) for alias in PROMPT_SPECS[label]["aliases"])
        lookup.append((label, aliases))
    return lookup


ALIAS_LOOKUP = build_alias_lookup()


def canonicalize_text_label(raw_label: str) -> str | None:
    text = normalize_text(str(raw_label))
    if not text:
        return None
    for label, aliases in ALIAS_LOOKUP:
        if any(alias and alias in text for alias in aliases):
            return label
    return None


def load_model(model_id: str):
    import torch
    from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

    resolved_model = resolve_model_source(model_id)
    local_files_only = Path(resolved_model).exists()
    print(f"[load] Grounding DINO model: {resolved_model}")
    processor = AutoProcessor.from_pretrained(resolved_model, local_files_only=local_files_only)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(
        resolved_model,
        local_files_only=local_files_only,
    )
    return model, processor, resolved_model, local_files_only


def setup_runtime(args: argparse.Namespace, local_files_only: bool) -> RuntimeConfig:
    import torch

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available in this environment.")

    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    if device == "auto":
        device = "cpu"

    gpu_memory_gb = 0.0
    if device == "cuda":
        gpu_memory_gb = (
            torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        )

    use_autocast = False
    autocast_dtype = None
    if device == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision("high")

        amp_mode = args.amp
        if amp_mode == "auto":
            amp_mode = "bf16" if torch.cuda.is_bf16_supported() else "fp16"
        if amp_mode == "bf16":
            use_autocast = True
            autocast_dtype = torch.bfloat16
        elif amp_mode == "fp16":
            use_autocast = True
            autocast_dtype = torch.float16

    if device == "cuda":
        if gpu_memory_gb >= 28:
            auto_batch_size = 8
            auto_max_batch_pixels = 18_000_000
            auto_inference_max_side = 1536
        elif gpu_memory_gb >= 20:
            auto_batch_size = 6
            auto_max_batch_pixels = 12_000_000
            auto_inference_max_side = 1408
        elif gpu_memory_gb >= 12:
            auto_batch_size = 4
            auto_max_batch_pixels = 8_000_000
            auto_inference_max_side = 1280
        else:
            auto_batch_size = DEFAULT_GPU_BATCH_SIZE
            auto_max_batch_pixels = DEFAULT_GPU_MAX_BATCH_PIXELS
            auto_inference_max_side = 1024
    else:
        auto_batch_size = DEFAULT_CPU_BATCH_SIZE
        auto_max_batch_pixels = 0
        auto_inference_max_side = 1024

    batch_size = args.batch_size or auto_batch_size
    max_batch_pixels = args.max_batch_pixels or auto_max_batch_pixels
    inference_max_side = args.inference_max_side or auto_inference_max_side
    return RuntimeConfig(
        device=device,
        use_autocast=use_autocast,
        autocast_dtype=autocast_dtype,
        local_files_only=local_files_only,
        batch_size=max(1, batch_size),
        max_batch_pixels=max(0, max_batch_pixels),
        inference_max_side=max(0, inference_max_side),
        gpu_memory_gb=gpu_memory_gb,
    )


def move_model_to_runtime(model, runtime: RuntimeConfig):
    import torch

    model = model.to(runtime.device).eval()
    if runtime.device == "cuda":
        try:
            model = model.to(memory_format=torch.channels_last)
        except Exception:
            pass
    return model


def iter_source_images(dataset_dir: Path) -> Iterable[tuple[str, Path]]:
    for label in CATEGORIES:
        class_dir = dataset_dir / label
        if not class_dir.exists():
            continue
        for path in sorted(class_dir.iterdir()):
            if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES:
                yield label, path


def file_sha1_from_bytes(data: bytes) -> str:
    return hashlib.sha1(data).hexdigest()


def image_ahash_from_image(image: Image.Image, size: int = 8) -> str:
    gray = image.convert("L").resize((size, size))
    pixels = list(gray.getdata())
    mean_value = sum(pixels) / len(pixels)
    bits = "".join("1" if pixel >= mean_value else "0" for pixel in pixels)
    return f"{int(bits, 2):016x}"


def load_image_rgb(path: Path) -> Image.Image:
    with Image.open(path) as image:
        image = ImageOps.exif_transpose(image).convert("RGB")
        return image.copy()


def prepare_inference_image(image: Image.Image, max_side: int) -> Image.Image:
    if max_side <= 0:
        return image
    width, height = image.size
    longest = max(width, height)
    if longest <= max_side:
        return image
    scale = max_side / float(longest)
    new_size = (max(1, round(width * scale)), max(1, round(height * scale)))
    return image.resize(new_size, Image.Resampling.BILINEAR)


def iou(box_a: tuple[float, float, float, float], box_b: tuple[float, float, float, float]) -> float:
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    if inter_x2 <= inter_x1 or inter_y2 <= inter_y1:
        return 0.0
    inter = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def classwise_nms(
    detections: Sequence[Detection],
    iou_threshold: float = 0.5,
) -> list[Detection]:
    kept: list[Detection] = []
    by_class: dict[str, list[Detection]] = defaultdict(list)
    for detection in detections:
        by_class[detection.label].append(detection)

    for label, items in by_class.items():
        ordered = sorted(items, key=lambda det: det.score, reverse=True)
        accepted: list[Detection] = []
        while ordered:
            current = ordered.pop(0)
            accepted.append(current)
            ordered = [
                candidate
                for candidate in ordered
                if iou(candidate.box, current.box) < iou_threshold
            ]
        kept.extend(accepted)

    return sorted(kept, key=lambda det: (LABEL_TO_IDX[det.label], -det.score))


def to_yolo(box: tuple[float, float, float, float], width: int, height: int) -> str:
    x1, y1, x2, y2 = box
    center_x = max(0.0, min(1.0, (x1 + x2) / 2 / width))
    center_y = max(0.0, min(1.0, (y1 + y2) / 2 / height))
    box_w = max(0.001, min(1.0, (x2 - x1) / width))
    box_h = max(0.001, min(1.0, (y2 - y1) / height))
    return f"{center_x:.6f} {center_y:.6f} {box_w:.6f} {box_h:.6f}"


def result_lists(result: dict) -> tuple[list, list, list]:
    boxes = result["boxes"]
    scores = result["scores"]
    text_labels = result.get("text_labels", result.get("labels", []))
    if hasattr(boxes, "tolist"):
        boxes = boxes.tolist()
    if hasattr(scores, "tolist"):
        scores = scores.tolist()
    return list(boxes), list(scores), list(text_labels)


def post_process_results(
    processor,
    outputs,
    input_ids,
    target_sizes: Sequence[tuple[int, int]],
    box_threshold: float,
    text_threshold: float,
) -> list[dict]:
    results = processor.post_process_grounded_object_detection(
        outputs,
        input_ids,
        threshold=box_threshold,
        text_threshold=text_threshold,
        target_sizes=list(target_sizes),
    )
    return list(results)


def predict_prompt_batch(
    model,
    processor,
    runtime: RuntimeConfig,
    images: Sequence[Image.Image],
    prompts: Sequence[str],
    target_sizes: Sequence[tuple[int, int]],
    box_threshold: float,
    text_threshold: float,
) -> list[dict]:
    import torch

    inputs = processor(images=list(images), text=list(prompts), return_tensors="pt").to(
        runtime.device
    )
    with torch.inference_mode():
        if runtime.use_autocast and runtime.device == "cuda":
            with torch.autocast(
                device_type="cuda",
                dtype=runtime.autocast_dtype,
            ):
                outputs = model(**inputs)
        else:
            outputs = model(**inputs)

    return post_process_results(
        processor,
        outputs,
        inputs.input_ids,
        list(target_sizes),
        box_threshold,
        text_threshold,
    )


def parse_multi_class_detections(result: dict) -> list[Detection]:
    detections: list[Detection] = []
    boxes, scores, text_labels = result_lists(result)
    for box, score, raw_label in zip(boxes, scores, text_labels):
        label = canonicalize_text_label(str(raw_label))
        if label is None:
            continue
        detections.append(
            Detection(
                label=label,
                score=float(score),
                box=tuple(float(value) for value in box),
                source="full_prompt",
            )
        )
    return detections


def parse_single_class_detections(result: dict, label: str) -> list[Detection]:
    detections: list[Detection] = []
    boxes, scores, _ = result_lists(result)
    for box, score in zip(boxes, scores):
        detections.append(
            Detection(
                label=label,
                score=float(score),
                box=tuple(float(value) for value in box),
                source="expected_fallback",
            )
        )
    return detections


def build_record(path: Path, expected_class: str) -> ImageRecord:
    data = path.read_bytes()
    with Image.open(io.BytesIO(data)) as image:
        image = ImageOps.exif_transpose(image)
        width, height = image.size
        ahash = image_ahash_from_image(image)
    return ImageRecord(
        path=path,
        expected_class=expected_class,
        width=width,
        height=height,
        sha1=file_sha1_from_bytes(data),
        ahash=ahash,
    )


def assign_quality_flags(record: ImageRecord) -> None:
    if min(record.width, record.height) < MIN_SHORT_SIDE:
        record.flags.add("small_image")

    if not record.detections:
        record.flags.add("empty_detection")
        record.flags.add("expected_missing")
        return

    expected_found = any(det.label == record.expected_class for det in record.detections)
    if not expected_found:
        record.flags.add("expected_missing")
    if expected_found and any(det.source == "expected_fallback" for det in record.detections):
        record.flags.add("fallback_used")

    labels = {det.label for det in record.detections}
    if labels and record.expected_class not in labels:
        record.flags.add("cross_class_only")

    if any(det.score < 0.35 for det in record.detections):
        record.flags.add("low_confidence")

    if len(record.detections) >= 8:
        record.flags.add("busy_scene")


def build_inference_batches(
    items: Sequence[ImageRecord],
    batch_size: int,
    max_batch_pixels: int,
) -> list[list[ImageRecord]]:
    batches: list[list[ImageRecord]] = []
    current: list[ImageRecord] = []
    current_pixels = 0

    for record in items:
        record_pixels = record.width * record.height
        exceed_count = len(current) >= batch_size
        exceed_pixels = (
            max_batch_pixels > 0
            and current
            and current_pixels + record_pixels > max_batch_pixels
        )
        if exceed_count or exceed_pixels:
            batches.append(current)
            current = []
            current_pixels = 0

        current.append(record)
        current_pixels += record_pixels

    if current:
        batches.append(current)
    return batches


def run_full_prompt_batch(
    records: Sequence[ImageRecord],
    model,
    processor,
    runtime: RuntimeConfig,
    args: argparse.Namespace,
) -> tuple[dict[Path, Image.Image], list[ImageRecord]]:
    images_by_record: dict[Path, Image.Image] = {}
    for record in records:
        images_by_record[record.path] = prepare_inference_image(
            load_image_rgb(record.path),
            runtime.inference_max_side,
        )

    results = predict_prompt_batch(
        model,
        processor,
        runtime,
        [images_by_record[record.path] for record in records],
        [FULL_PROMPT] * len(records),
        [(record.height, record.width) for record in records],
        args.box_threshold,
        args.text_threshold,
    )

    missing_expected: list[ImageRecord] = []
    for record, result in zip(records, results):
        record.detections = parse_multi_class_detections(result)
        if not any(det.label == record.expected_class for det in record.detections):
            missing_expected.append(record)
    return images_by_record, missing_expected


def run_fallback_batches(
    missing_records: Sequence[ImageRecord],
    images_by_record: dict[Path, Image.Image],
    model,
    processor,
    runtime: RuntimeConfig,
    args: argparse.Namespace,
) -> None:
    by_expected_class: dict[str, list[ImageRecord]] = defaultdict(list)
    for record in missing_records:
        by_expected_class[record.expected_class].append(record)

    for label, records in by_expected_class.items():
        prompt = str(PROMPT_SPECS[label]["fallback"])
        for batch_records in build_inference_batches(
            records,
            runtime.batch_size,
            runtime.max_batch_pixels,
        ):
            results = predict_prompt_batch(
                model,
                processor,
                runtime,
                [images_by_record[record.path] for record in batch_records],
                [prompt] * len(batch_records),
                [(record.height, record.width) for record in batch_records],
                args.fallback_box_threshold,
                args.fallback_text_threshold,
            )
            for record, result in zip(batch_records, results):
                record.detections.extend(parse_single_class_detections(result, label))


def finalize_batch(records: Sequence[ImageRecord]) -> None:
    for record in records:
        record.detections = classwise_nms(record.detections)
        assign_quality_flags(record)


def detect_records_batch(
    records: Sequence[ImageRecord],
    model,
    processor,
    runtime: RuntimeConfig,
    args: argparse.Namespace,
) -> None:
    images_by_record, missing_expected = run_full_prompt_batch(
        records,
        model,
        processor,
        runtime,
        args,
    )
    if missing_expected:
        run_fallback_batches(
            missing_expected,
            images_by_record,
            model,
            processor,
            runtime,
            args,
        )
    finalize_batch(records)


def is_cuda_oom(error: Exception) -> bool:
    text = str(error).lower()
    return "out of memory" in text and "cuda" in text


def release_cuda_memory() -> None:
    import torch

    gc.collect()
    try:
        torch.cuda.synchronize()
    except Exception:
        pass
    try:
        torch.cuda.empty_cache()
    except Exception:
        pass


def detect_records_adaptive(
    records: Sequence[ImageRecord],
    model,
    processor,
    runtime: RuntimeConfig,
    args: argparse.Namespace,
) -> None:
    try:
        detect_records_batch(records, model, processor, runtime, args)
    except RuntimeError as exc:
        if runtime.device == "cuda" and is_cuda_oom(exc) and len(records) > 1:
            release_cuda_memory()
            left_size = len(records) // 2
            right_size = len(records) - left_size
            print(
                f"[warn] CUDA OOM on batch size {len(records)}; "
                f"retrying with {left_size} + {right_size}"
            )
            detect_records_adaptive(records[:left_size], model, processor, runtime, args)
            detect_records_adaptive(records[left_size:], model, processor, runtime, args)
            return
        if runtime.device == "cuda" and is_cuda_oom(exc):
            raise RuntimeError(
                "CUDA ran out of memory even on a minimal batch. "
                "Try lowering --batch-size or --max-batch-pixels."
            ) from exc
        raise


def reset_outputs(dataset_dir: Path) -> None:
    for relative_path in MANAGED_PATHS:
        path = dataset_dir / relative_path
        if path.is_dir():
            shutil.rmtree(path, ignore_errors=True)
        elif path.exists():
            path.unlink()


def group_records(records: Sequence[ImageRecord]) -> list[list[ImageRecord]]:
    grouped: dict[str, list[ImageRecord]] = defaultdict(list)
    for record in records:
        grouped[record.sha1].append(record)
    return list(grouped.values())


def annotate_duplicate_flags(groups: Sequence[Sequence[ImageRecord]]) -> None:
    for group in groups:
        if len(group) <= 1:
            continue
        expected_classes = {record.expected_class for record in group}
        for record in group:
            record.flags.add("exact_duplicate")
            if len(expected_classes) > 1:
                record.flags.add("cross_class_duplicate")


def allocate_splits(
    records: Sequence[ImageRecord],
    train_ratio: float,
    val_ratio: float,
    seed: int,
) -> None:
    groups = group_records(records)
    grouped_by_primary: dict[str, list[list[ImageRecord]]] = defaultdict(list)
    for group in groups:
        primary_class = Counter(record.expected_class for record in group).most_common(1)[0][0]
        grouped_by_primary[primary_class].append(list(group))

    rng = random.Random(seed)
    test_ratio = 1.0 - train_ratio - val_ratio
    split_order = ("train", "val", "test")

    for label, bucket in grouped_by_primary.items():
        rng.shuffle(bucket)
        total = sum(len(group) for group in bucket)
        targets = {
            "train": total * train_ratio,
            "val": total * val_ratio,
            "test": total * test_ratio,
        }
        current = {name: 0 for name in split_order}

        for group in sorted(bucket, key=len, reverse=True):
            deficits = {
                split: targets[split] - current[split]
                for split in split_order
            }
            split = max(split_order, key=lambda name: (deficits[name], -current[name]))
            for record in group:
                record.split = split
                current[split] += 1

        print(
            f"[split] {label:<18} total={total:<4} "
            f"train={current['train']:<4} val={current['val']:<4} test={current['test']:<4}"
        )


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def write_label_file(path: Path, record: ImageRecord) -> None:
    lines = [
        f"{LABEL_TO_IDX[det.label]} {to_yolo(det.box, record.width, record.height)}"
        for det in record.detections
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def export_record(record: ImageRecord, image_dir: Path, label_dir: Path) -> None:
    ensure_dir(image_dir)
    ensure_dir(label_dir)
    shutil.copy2(record.path, image_dir / record.file_name)
    write_label_file(label_dir / f"{record.path.stem}.txt", record)


def export_dataset(
    dataset_dir: Path,
    records: Sequence[ImageRecord],
    review_policy: str,
    train_ratio: float,
    val_ratio: float,
    seed: int,
) -> tuple[list[ImageRecord], list[ImageRecord]]:
    kept: list[ImageRecord] = []
    review: list[ImageRecord] = []

    for record in records:
        if review_policy == "exclude" and record.review_required:
            review.append(record)
            continue
        kept.append(record)

    allocate_splits(kept, train_ratio, val_ratio, seed)

    for record in kept:
        export_record(
            record,
            dataset_dir / "images" / str(record.split),
            dataset_dir / "labels" / str(record.split),
        )

    review_image_dir = dataset_dir / REVIEW_DIR_NAME / REVIEW_IMAGE_DIR
    review_label_dir = dataset_dir / REVIEW_DIR_NAME / REVIEW_LABEL_DIR
    for record in review:
        export_record(record, review_image_dir, review_label_dir)

    return kept, review


def make_data_yaml(dataset_dir: Path) -> None:
    names = {cfg["idx"]: label for label, cfg in CATEGORIES.items()}
    names_str = "\n".join(f"  {idx}: {names[idx]}" for idx in sorted(names))
    content = (
        "# Auto-generated FengShui detection dataset\n"
        f"path: {dataset_dir.resolve()}\n"
        "train: images/train\n"
        "val: images/val\n"
        "test: images/test\n\n"
        f"nc: {len(CATEGORIES)}\n"
        "names:\n"
        f"{names_str}\n"
    )
    (dataset_dir / "data.yaml").write_text(content, encoding="utf-8")


def write_review_outputs(dataset_dir: Path, records: Sequence[ImageRecord]) -> None:
    review_dir = dataset_dir / REVIEW_DIR_NAME
    ensure_dir(review_dir)

    needs_review = [record for record in records if record.review_required]
    (review_dir / NEEDS_REVIEW_FILE).write_text(
        "\n".join(str(record.path) for record in needs_review),
        encoding="utf-8",
    )

    manifest_path = dataset_dir / "split_manifest.csv"
    with manifest_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "source_image",
                "expected_class",
                "split",
                "review_required",
                "flags",
                "detected_labels",
                "box_count",
                "sha1",
                "ahash",
            ]
        )
        for record in sorted(records, key=lambda item: (item.expected_class, item.path.name)):
            writer.writerow(
                [
                    str(record.path),
                    record.expected_class,
                    record.split or "review",
                    int(record.review_required),
                    ";".join(sorted(record.flags)),
                    ";".join(record.unique_detected_labels),
                    len(record.detections),
                    record.sha1,
                    record.ahash,
                ]
            )


def class_counter(records: Sequence[ImageRecord]) -> dict[str, int]:
    counts = Counter(record.expected_class for record in records)
    return {label: counts.get(label, 0) for label in CATEGORIES}


def detected_counter(records: Sequence[ImageRecord]) -> dict[str, int]:
    counts = Counter()
    for record in records:
        for label in record.unique_detected_labels:
            counts[label] += 1
    return {label: counts.get(label, 0) for label in CATEGORIES}


def flag_counter(records: Sequence[ImageRecord]) -> dict[str, int]:
    counts = Counter()
    for record in records:
        for flag in record.flags:
            counts[flag] += 1
    return dict(sorted(counts.items()))


def write_summary(
    dataset_dir: Path,
    all_records: Sequence[ImageRecord],
    kept_records: Sequence[ImageRecord],
    review_records: Sequence[ImageRecord],
) -> None:
    split_counts = Counter(record.split for record in kept_records)
    summary = {
        "source_images": len(all_records),
        "exported_images": len(kept_records),
        "review_images": len(review_records),
        "source_expected_class_counts": class_counter(all_records),
        "export_expected_class_counts": class_counter(kept_records),
        "review_expected_class_counts": class_counter(review_records),
        "detected_label_coverage": detected_counter(all_records),
        "split_counts": dict(sorted(split_counts.items())),
        "flag_counts": flag_counter(all_records),
    }
    (dataset_dir / "annotation_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def collect_records(
    dataset_dir: Path,
    limit: int = 0,
    scan_workers: int = DEFAULT_SCAN_WORKERS,
) -> list[ImageRecord]:
    pairs = list(iter_source_images(dataset_dir))
    if limit:
        pairs = pairs[:limit]

    print(f"[scan] Preparing metadata for {len(pairs)} source images...")
    if not pairs:
        return []

    if scan_workers <= 1:
        return [
            build_record(path, label)
            for label, path in tqdm(pairs, desc="Scan source images", unit="img")
        ]

    with ThreadPoolExecutor(max_workers=scan_workers) as executor:
        records = list(
            tqdm(
                executor.map(lambda pair: build_record(pair[1], pair[0]), pairs),
                total=len(pairs),
                desc=f"Scan source images ({scan_workers} workers)",
                unit="img",
            )
        )
    return records


def print_summary(all_records: Sequence[ImageRecord], kept: Sequence[ImageRecord], review: Sequence[ImageRecord]) -> None:
    split_counts = Counter(record.split for record in kept)
    print("\n[summary]")
    print(f"  source images : {len(all_records)}")
    print(f"  exported      : {len(kept)}")
    print(f"  review queue  : {len(review)}")
    print(
        "  splits        : "
        f"train={split_counts.get('train', 0)} "
        f"val={split_counts.get('val', 0)} "
        f"test={split_counts.get('test', 0)}"
    )
    flags = flag_counter(all_records)
    if flags:
        print("  flags:")
        for name, count in flags.items():
            print(f"    - {name}: {count}")


def main(args: argparse.Namespace) -> None:
    validate_args(args)
    dataset_dir = Path(args.dataset_dir)
    if not dataset_dir.exists():
        raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")

    if args.clean:
        print("[clean] Removing previous generated artifacts.")
        reset_outputs(dataset_dir)

    print(f"[stage] Dataset directory: {dataset_dir.resolve()}")
    source_records = collect_records(
        dataset_dir,
        limit=args.limit,
        scan_workers=args.scan_workers,
    )
    if not source_records:
        raise RuntimeError("No source images found under the dataset directory.")

    print("[stage] Checking for exact duplicates...")
    duplicate_groups = group_records(source_records)
    annotate_duplicate_flags(duplicate_groups)

    print("[stage] Loading Grounding DINO model...")
    model, processor, resolved_model, local_files_only = load_model(args.model_id)
    runtime = setup_runtime(args, local_files_only)
    model = move_model_to_runtime(model, runtime)
    precision_name = (
        str(runtime.autocast_dtype).split(".")[-1]
        if runtime.use_autocast and runtime.autocast_dtype is not None
        else "fp32"
    )
    print(
        f"[runtime] device={runtime.device} gpu_memory_gb={runtime.gpu_memory_gb:.1f} "
        f"batch_size={runtime.batch_size} "
        f"amp={precision_name} max_batch_pixels={runtime.max_batch_pixels} "
        f"inference_max_side={runtime.inference_max_side} "
        f"model={resolved_model}"
    )
    print("[stage] Starting multi-class auto-labeling...")
    inference_batches = build_inference_batches(
        source_records,
        runtime.batch_size,
        runtime.max_batch_pixels,
    )
    total_batches = len(inference_batches)
    for batch_records in tqdm(
        inference_batches,
        total=total_batches,
        desc="Auto-label batches",
        unit="batch",
    ):
        detect_records_adaptive(batch_records, model, processor, runtime, args)

    print("[stage] Exporting train/val/test and review queues...")
    kept_records, review_records = export_dataset(
        dataset_dir,
        source_records,
        review_policy=args.review_policy,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )
    make_data_yaml(dataset_dir)
    write_review_outputs(dataset_dir, source_records)
    write_summary(dataset_dir, source_records, kept_records, review_records)
    print_summary(source_records, kept_records, review_records)

    print(f"\n[next] Review queue: {(dataset_dir / REVIEW_DIR_NAME).resolve()}")
    print(f"[next] Training config: {(dataset_dir / 'data.yaml').resolve()}")


if __name__ == "__main__":
    args = parse_args()
    try:
        main(args)
    except Exception as exc:
        print(f"[error] {exc}")
        print("Install deps if needed: pip install torch torchvision transformers pillow tqdm")
        raise
