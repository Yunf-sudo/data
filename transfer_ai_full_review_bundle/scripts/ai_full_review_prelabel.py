from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Sequence

from PIL import Image, ImageDraw, ImageFont, ImageOps
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from auto_label import (  # noqa: E402
    LOCAL_MODEL_DIR,
    PROMPT_SPECS,
    load_image_rgb,
    load_model,
    move_model_to_runtime,
    predict_prompt_batch,
    prepare_inference_image,
    setup_runtime,
)


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
DEFAULT_CLASS_NAMES = [
    "beam",
    "bed",
    "broad_leaf_live",
    "cabinet",
    "coffee_table",
    "desk",
    "dining_table",
    "fake_plant",
    "floor_window",
    "main_door",
    "mirror",
    "normal_window",
    "room_door",
    "sharp_leaf_live",
    "sink",
    "sofa",
    "stairs",
    "stove",
    "toilet",
    "water_feature",
]
ALIAS_PRIORITY = [
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
DEFAULT_FULL_BOX_THRESHOLD = 0.28
DEFAULT_FULL_TEXT_THRESHOLD = 0.24
DEFAULT_AUDIT_BOX_THRESHOLD = 0.22
DEFAULT_AUDIT_TEXT_THRESHOLD = 0.18
DEFAULT_LOW_CONFIDENCE_THRESHOLD = 0.35
DEFAULT_BUSY_SCENE_THRESHOLD = 8


@dataclass
class Detection:
    label: str
    score: float
    box: tuple[float, float, float, float]
    source: str


@dataclass
class SourceRecord:
    source_path: Path
    source_class: str
    width: int
    height: int
    sha1: str
    export_stem: str
    detections: list[Detection] = field(default_factory=list)
    flags: set[str] = field(default_factory=set)

    @property
    def export_image_name(self) -> str:
        return f"{self.export_stem}{self.source_path.suffix.lower()}"

    @property
    def export_label_name(self) -> str:
        return f"{self.export_stem}.txt"

    @property
    def unique_detected_labels(self) -> list[str]:
        return sorted({det.label for det in self.detections})

    @property
    def export_image_relpath(self) -> Path:
        return Path(self.source_class) / self.export_image_name

    @property
    def export_label_relpath(self) -> Path:
        return Path(self.source_class) / self.export_label_name

    @property
    def export_preview_relpath(self) -> Path:
        return Path(self.source_class) / f"{self.export_stem}.jpg"


@dataclass
class SkippedImage:
    source_path: Path
    source_class: str
    reason: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="AI pre-label all 20 FengShui classes and export a full human-review workspace."
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=Path("FengShui_Dataset"),
        help="Dataset root containing the 20 raw class folders.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("FengShui_Dataset/ai_full_review"),
        help="Output directory for unified review workspace.",
    )
    parser.add_argument(
        "--classes-yaml",
        type=Path,
        default=Path("windows_label_tool/data.yaml"),
        help="Class order yaml used by the Windows label tool.",
    )
    parser.add_argument(
        "--model-id",
        default=str(LOCAL_MODEL_DIR),
        help="Local Grounding DINO folder or Hugging Face model id.",
    )
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
        help="Mixed precision mode on CUDA.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=0,
        help="Inference batch size. Default is auto tuned.",
    )
    parser.add_argument(
        "--max-batch-pixels",
        type=int,
        default=0,
        help="Max pixels per inference batch. Default is auto tuned.",
    )
    parser.add_argument(
        "--inference-max-side",
        type=int,
        default=0,
        help="Resize long side for inference only. Labels stay in original coordinates.",
    )
    parser.add_argument(
        "--full-box-threshold",
        type=float,
        default=DEFAULT_FULL_BOX_THRESHOLD,
        help="Grounding DINO box threshold for the 20-class full prompt pass.",
    )
    parser.add_argument(
        "--full-text-threshold",
        type=float,
        default=DEFAULT_FULL_TEXT_THRESHOLD,
        help="Grounding DINO text threshold for the 20-class full prompt pass.",
    )
    parser.add_argument(
        "--audit-box-threshold",
        type=float,
        default=DEFAULT_AUDIT_BOX_THRESHOLD,
        help="Grounding DINO box threshold for the second-pass targeted audit.",
    )
    parser.add_argument(
        "--audit-text-threshold",
        type=float,
        default=DEFAULT_AUDIT_TEXT_THRESHOLD,
        help="Grounding DINO text threshold for the second-pass targeted audit.",
    )
    parser.add_argument(
        "--audit-mode",
        choices=("source_only", "missing_all", "none"),
        default="source_only",
        help=(
            "Second-pass audit scope. source_only is the safest default: "
            "all 20 classes are checked once by the full prompt, and only the "
            "source folder class gets a stricter second look."
        ),
    )
    parser.add_argument(
        "--low-confidence-threshold",
        type=float,
        default=DEFAULT_LOW_CONFIDENCE_THRESHOLD,
        help="Images with detections below this score get flagged.",
    )
    parser.add_argument(
        "--busy-scene-threshold",
        type=int,
        default=DEFAULT_BUSY_SCENE_THRESHOLD,
        help="Images with this many boxes or more get flagged as busy_scene.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Optional debug limit on source images.",
    )
    parser.add_argument(
        "--force-redetect",
        action="store_true",
        help="Ignore cache and rerun detection for every image.",
    )
    parser.add_argument(
        "--render-previews",
        action="store_true",
        help="Render boxed preview images for faster manual review.",
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Delete the previous review workspace before exporting.",
    )
    return parser.parse_args()


def load_class_names(data_yaml: Path) -> list[str]:
    if not data_yaml.exists():
        return list(DEFAULT_CLASS_NAMES)

    lines = data_yaml.read_text(encoding="utf-8", errors="ignore").splitlines()
    in_names = False
    mapping: dict[int, str] = {}
    for line in lines:
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if stripped == "names:":
            in_names = True
            continue
        if in_names:
            if not line.startswith((" ", "\t")):
                break
            if ":" not in stripped:
                continue
            key, value = stripped.split(":", 1)
            key = key.strip()
            value = value.strip().strip("'\"")
            if key.isdigit() and value:
                mapping[int(key)] = value
    if not mapping:
        return list(DEFAULT_CLASS_NAMES)
    return [mapping[idx] for idx in sorted(mapping)]


def normalize_text(text: str) -> str:
    cleaned = []
    for char in text.lower():
        cleaned.append(char if char.isalnum() else " ")
    return " ".join("".join(cleaned).split())


def build_alias_lookup(class_names: Sequence[str]) -> list[tuple[str, tuple[str, ...]]]:
    ordered_names = [name for name in ALIAS_PRIORITY if name in class_names]
    ordered_names.extend(name for name in class_names if name not in ordered_names)
    lookup: list[tuple[str, tuple[str, ...]]] = []
    for label in ordered_names:
        aliases = tuple(normalize_text(alias) for alias in PROMPT_SPECS[label]["aliases"])
        lookup.append((label, aliases))
    return lookup


def canonicalize_text_label(raw_label: str, alias_lookup: Sequence[tuple[str, tuple[str, ...]]]) -> str | None:
    text = normalize_text(str(raw_label))
    if not text:
        return None
    for label, aliases in alias_lookup:
        if any(alias and alias in text for alias in aliases):
            return label
    return None


def file_sha1(data: bytes) -> str:
    return hashlib.sha1(data).hexdigest()


def build_source_record(path: Path, source_class: str) -> SourceRecord:
    data = path.read_bytes()
    sha1 = file_sha1(data)
    with Image.open(path) as image:
        image = ImageOps.exif_transpose(image)
        width, height = image.size
    stem = f"{source_class}__{path.stem}__{sha1[:12]}"
    return SourceRecord(
        source_path=path,
        source_class=source_class,
        width=width,
        height=height,
        sha1=sha1,
        export_stem=stem,
    )


def iter_source_paths(dataset_dir: Path, class_names: Sequence[str]) -> Iterable[tuple[str, Path]]:
    for class_name in class_names:
        class_dir = dataset_dir / class_name
        if not class_dir.exists():
            continue
        for path in sorted(class_dir.iterdir()):
            if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES:
                yield class_name, path


def collect_records(
    dataset_dir: Path,
    class_names: Sequence[str],
    limit: int,
) -> tuple[list[SourceRecord], list[SkippedImage]]:
    pairs = list(iter_source_paths(dataset_dir, class_names))
    if limit > 0:
        pairs = pairs[:limit]
    records: list[SourceRecord] = []
    skipped: list[SkippedImage] = []
    for source_class, path in tqdm(pairs, desc="Scan source images", unit="img"):
        try:
            records.append(build_source_record(path, source_class))
        except Exception as exc:
            skipped.append(
                SkippedImage(
                    source_path=path,
                    source_class=source_class,
                    reason=str(exc),
                )
            )
    return records, skipped


def build_inference_batches(
    records: Sequence[SourceRecord],
    batch_size: int,
    max_batch_pixels: int,
) -> list[list[SourceRecord]]:
    batches: list[list[SourceRecord]] = []
    current: list[SourceRecord] = []
    current_pixels = 0
    for record in records:
        record_pixels = record.width * record.height
        exceed_count = len(current) >= batch_size
        exceed_pixels = max_batch_pixels > 0 and current and (current_pixels + record_pixels > max_batch_pixels)
        if exceed_count or exceed_pixels:
            batches.append(current)
            current = []
            current_pixels = 0
        current.append(record)
        current_pixels += record_pixels
    if current:
        batches.append(current)
    return batches


def result_lists(result: dict) -> tuple[list, list, list]:
    boxes = result["boxes"]
    scores = result["scores"]
    text_labels = result.get("text_labels", result.get("labels", []))
    if hasattr(boxes, "tolist"):
        boxes = boxes.tolist()
    if hasattr(scores, "tolist"):
        scores = scores.tolist()
    return list(boxes), list(scores), list(text_labels)


def parse_multi_class_detections(
    result: dict,
    alias_lookup: Sequence[tuple[str, tuple[str, ...]]],
) -> list[Detection]:
    detections: list[Detection] = []
    boxes, scores, text_labels = result_lists(result)
    for box, score, raw_label in zip(boxes, scores, text_labels):
        label = canonicalize_text_label(str(raw_label), alias_lookup)
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
                source="class_audit",
            )
        )
    return detections


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
    class_to_idx: dict[str, int],
    iou_threshold: float = 0.5,
) -> list[Detection]:
    grouped: dict[str, list[Detection]] = defaultdict(list)
    for detection in detections:
        grouped[detection.label].append(detection)

    kept: list[Detection] = []
    for label, items in grouped.items():
        pending = sorted(items, key=lambda item: item.score, reverse=True)
        while pending:
            current = pending.pop(0)
            kept.append(current)
            pending = [candidate for candidate in pending if iou(candidate.box, current.box) < iou_threshold]
    return sorted(kept, key=lambda item: (class_to_idx[item.label], -item.score))


def assign_flags(record: SourceRecord, low_conf_threshold: float, busy_scene_threshold: int) -> None:
    record.flags = {"manual_review_all"}
    if not record.detections:
        record.flags.add("empty_detection")
        record.flags.add("source_class_missing")
        return

    labels = {det.label for det in record.detections}
    if record.source_class not in labels:
        record.flags.add("source_class_missing")
    if labels and record.source_class not in labels:
        record.flags.add("cross_class_only")
    if any(det.score < low_conf_threshold for det in record.detections):
        record.flags.add("low_confidence")
    if len(record.detections) >= busy_scene_threshold:
        record.flags.add("busy_scene")


def cache_path_for(record: SourceRecord, cache_dir: Path) -> Path:
    return cache_dir / f"{record.export_stem}.json"


def save_cache(record: SourceRecord, cache_dir: Path) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "source_path": str(record.source_path),
        "source_class": record.source_class,
        "width": record.width,
        "height": record.height,
        "sha1": record.sha1,
        "export_stem": record.export_stem,
        "flags": sorted(record.flags),
        "detections": [
            {
                "label": det.label,
                "score": det.score,
                "box": list(det.box),
                "source": det.source,
            }
            for det in record.detections
        ],
    }
    cache_path_for(record, cache_dir).write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def try_load_cache(record: SourceRecord, cache_dir: Path) -> bool:
    path = cache_path_for(record, cache_dir)
    if not path.exists():
        return False
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("sha1") != record.sha1:
        return False
    if payload.get("source_class") != record.source_class:
        return False
    record.flags = set(payload.get("flags", []))
    record.detections = [
        Detection(
            label=item["label"],
            score=float(item["score"]),
            box=tuple(float(value) for value in item["box"]),
            source=str(item.get("source", "cache")),
        )
        for item in payload.get("detections", [])
    ]
    return True


def detect_batch(
    batch_records: Sequence[SourceRecord],
    model,
    processor,
    runtime,
    class_names: Sequence[str],
    full_prompt: str,
    alias_lookup: Sequence[tuple[str, tuple[str, ...]]],
    class_to_idx: dict[str, int],
    args: argparse.Namespace,
) -> None:
    images_by_path: dict[Path, Image.Image] = {}
    for record in batch_records:
        images_by_path[record.source_path] = prepare_inference_image(
            load_image_rgb(record.source_path),
            runtime.inference_max_side,
        )

    results = predict_prompt_batch(
        model,
        processor,
        runtime,
        [images_by_path[record.source_path] for record in batch_records],
        [full_prompt] * len(batch_records),
        [(record.height, record.width) for record in batch_records],
        args.full_box_threshold,
        args.full_text_threshold,
    )

    missing_by_class: dict[str, list[SourceRecord]] = defaultdict(list)
    for record, result in zip(batch_records, results):
        record.detections = parse_multi_class_detections(result, alias_lookup)
        found = {det.label for det in record.detections}
        if args.audit_mode == "missing_all":
            for class_name in class_names:
                if class_name not in found:
                    missing_by_class[class_name].append(record)
        elif args.audit_mode == "source_only":
            source_scores = [det.score for det in record.detections if det.label == record.source_class]
            if not source_scores or max(source_scores) < args.low_confidence_threshold:
                missing_by_class[record.source_class].append(record)

    for class_name in class_names:
        if args.audit_mode == "none":
            break
        missing_records = missing_by_class.get(class_name, [])
        if not missing_records:
            continue
        prompt = str(PROMPT_SPECS[class_name]["fallback"])
        sub_batches = build_inference_batches(
            missing_records,
            max(1, runtime.batch_size),
            runtime.max_batch_pixels,
        )
        for sub_batch in sub_batches:
            audit_results = predict_prompt_batch(
                model,
                processor,
                runtime,
                [images_by_path[record.source_path] for record in sub_batch],
                [prompt] * len(sub_batch),
                [(record.height, record.width) for record in sub_batch],
                args.audit_box_threshold,
                args.audit_text_threshold,
            )
            for record, result in zip(sub_batch, audit_results):
                record.detections.extend(parse_single_class_detections(result, class_name))

    for record in batch_records:
        record.detections = classwise_nms(record.detections, class_to_idx)
        assign_flags(record, args.low_confidence_threshold, args.busy_scene_threshold)


def ensure_clean_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def reset_output_dir(output_dir: Path) -> None:
    if output_dir.exists():
        shutil.rmtree(output_dir)


def to_yolo(box: tuple[float, float, float, float], width: int, height: int) -> str:
    x1, y1, x2, y2 = box
    center_x = max(0.0, min(1.0, (x1 + x2) / 2.0 / width))
    center_y = max(0.0, min(1.0, (y1 + y2) / 2.0 / height))
    box_width = max(0.001, min(1.0, (x2 - x1) / width))
    box_height = max(0.001, min(1.0, (y2 - y1) / height))
    return f"{center_x:.6f} {center_y:.6f} {box_width:.6f} {box_height:.6f}"


def link_or_copy(src: Path, dst: Path) -> None:
    if dst.exists():
        return
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def color_for_class(class_id: int) -> tuple[int, int, int]:
    return (
        80 + (class_id * 47) % 140,
        80 + (class_id * 83) % 140,
        80 + (class_id * 109) % 140,
    )


def draw_preview(
    image_path: Path,
    detections: Sequence[Detection],
    class_to_idx: dict[str, int],
    preview_path: Path,
) -> None:
    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    line_width = max(2, round(min(image.size) / 250))
    padding = max(2, line_width)

    for det in detections:
        class_id = class_to_idx[det.label]
        x1, y1, x2, y2 = det.box
        x1 = max(0, min(image.width - 1, x1))
        y1 = max(0, min(image.height - 1, y1))
        x2 = max(0, min(image.width - 1, x2))
        y2 = max(0, min(image.height - 1, y2))
        if x2 <= x1 or y2 <= y1:
            continue

        color = color_for_class(class_id)
        draw.rectangle((x1, y1, x2, y2), outline=color, width=line_width)
        label = f"{class_id}:{det.label} {det.score:.2f}"
        left, top, right, bottom = draw.textbbox((0, 0), label, font=font)
        label_height = bottom - top
        label_width = right - left
        box_top = max(0, y1 - label_height - padding * 2)
        draw.rectangle(
            (
                x1,
                box_top,
                x1 + label_width + padding * 2,
                box_top + label_height + padding * 2,
            ),
            fill=color,
        )
        draw.text((x1 + padding, box_top + padding), label, fill=(255, 255, 255), font=font)

    preview_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(preview_path, quality=92)


def write_data_yaml(output_dir: Path, class_names: Sequence[str]) -> None:
    names = "\n".join(f"  {idx}: {name}" for idx, name in enumerate(class_names))
    content = (
        "# Auto-generated review workspace config\n"
        f"path: {output_dir.resolve()}\n"
        "train: images\n"
        "val: images\n"
        "test: images\n\n"
        f"nc: {len(class_names)}\n"
        "names:\n"
        f"{names}\n"
    )
    (output_dir / "data.yaml").write_text(content, encoding="utf-8")


def write_labels_and_assets(
    records: Sequence[SourceRecord],
    output_dir: Path,
    class_to_idx: dict[str, int],
    render_previews: bool,
) -> None:
    image_dir = output_dir / "images"
    label_dir = output_dir / "labels_prelabel"
    preview_dir = output_dir / "annotated_previews"
    if image_dir.exists():
        shutil.rmtree(image_dir)
    if label_dir.exists():
        shutil.rmtree(label_dir)
    if preview_dir.exists():
        shutil.rmtree(preview_dir)
    ensure_clean_dir(image_dir)
    ensure_clean_dir(label_dir)
    if render_previews:
        ensure_clean_dir(preview_dir)

    for record in tqdm(records, desc="Export review workspace", unit="img"):
        export_image_path = image_dir / record.export_image_relpath
        export_label_path = label_dir / record.export_label_relpath
        export_image_path.parent.mkdir(parents=True, exist_ok=True)
        export_label_path.parent.mkdir(parents=True, exist_ok=True)
        link_or_copy(record.source_path, export_image_path)
        lines = [
            f"{class_to_idx[det.label]} {to_yolo(det.box, record.width, record.height)}"
            for det in record.detections
        ]
        export_label_path.write_text("\n".join(lines), encoding="utf-8")
        if render_previews:
            preview_path = preview_dir / record.export_preview_relpath
            preview_path.parent.mkdir(parents=True, exist_ok=True)
            draw_preview(
                export_image_path,
                record.detections,
                class_to_idx,
                preview_path,
            )


def write_manifest(
    records: Sequence[SourceRecord],
    output_dir: Path,
    class_names: Sequence[str],
) -> None:
    manifest_path = output_dir / "review_manifest.csv"
    header = [
        "export_image",
        "export_label",
        "source_image",
        "source_class",
        "width",
        "height",
        "box_count",
        "detected_labels",
        "flags",
        "sha1",
    ]
    header.extend(f"pred_{name}" for name in class_names)
    header.extend(f"human_check_{name}" for name in class_names)

    with manifest_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        for record in sorted(records, key=lambda item: (item.source_class, item.source_path.name)):
            labels = set(record.unique_detected_labels)
            row = [
                str(record.export_image_relpath).replace("\\", "/"),
                str(record.export_label_relpath).replace("\\", "/"),
                str(record.source_path),
                record.source_class,
                record.width,
                record.height,
                len(record.detections),
                ";".join(record.unique_detected_labels),
                ";".join(sorted(record.flags)),
                record.sha1,
            ]
            row.extend(1 if class_name in labels else 0 for class_name in class_names)
            row.extend("" for _ in class_names)
            writer.writerow(row)

    (output_dir / "review_all.txt").write_text(
        "\n".join(str(record.source_path) for record in records),
        encoding="utf-8",
    )


def write_summary(
    records: Sequence[SourceRecord],
    output_dir: Path,
    class_names: Sequence[str],
    skipped: Sequence[SkippedImage],
) -> None:
    image_presence = Counter()
    instance_counts = Counter()
    flag_counts = Counter()
    source_counts = Counter()

    for record in records:
        source_counts[record.source_class] += 1
        for flag in record.flags:
            flag_counts[flag] += 1
        present_labels = set()
        for det in record.detections:
            instance_counts[det.label] += 1
            present_labels.add(det.label)
        for label in present_labels:
            image_presence[label] += 1

    summary = {
        "total_images": len(records),
        "skipped_images": len(skipped),
        "class_order": list(class_names),
        "source_folder_counts": {name: source_counts.get(name, 0) for name in class_names},
        "predicted_image_presence": {name: image_presence.get(name, 0) for name in class_names},
        "predicted_instance_counts": {name: instance_counts.get(name, 0) for name in class_names},
        "flag_counts": dict(sorted(flag_counts.items())),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    lines = [
        f"total_images: {len(records)}",
        f"skipped_images: {len(skipped)}",
        "",
        "flag counts:",
    ]
    for flag, count in sorted(flag_counts.items()):
        lines.append(f"- {flag}: {count}")
    lines.append("")
    lines.append("predicted class coverage:")
    for class_name in class_names:
        lines.append(
            f"- {class_name}: images={image_presence.get(class_name, 0)}, "
            f"instances={instance_counts.get(class_name, 0)}"
        )
    (output_dir / "summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_skipped_report(skipped: Sequence[SkippedImage], output_dir: Path) -> None:
    report_path = output_dir / "skipped_images.csv"
    with report_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["source_class", "source_image", "reason"])
        for item in skipped:
            writer.writerow([item.source_class, str(item.source_path), item.reason])


def write_readme(output_dir: Path) -> None:
    content = (
        "AI full-review workspace\n"
        "\n"
        "1. Open the label tool.\n"
        "2. Choose images folder: images/\n"
        "3. Choose labels folder: labels_prelabel/\n"
        "4. Load classes from data.yaml if needed.\n"
        "5. Review every image and check all 20 classes, not only the source folder class.\n"
        "6. You can track progress in review_manifest.csv.\n"
        "\n"
        "Files:\n"
        "- images/: unified review images\n"
        "- labels_prelabel/: AI prelabels in YOLO txt format\n"
        "- annotated_previews/: optional boxed previews if enabled\n"
        "- review_manifest.csv: one row per image, with pred_* columns for all 20 classes\n"
        "- summary.txt / summary.json: review statistics\n"
        "- data.yaml: class order used for label ids\n"
    )
    (output_dir / "README_review.txt").write_text(content, encoding="utf-8")


def main() -> None:
    args = parse_args()
    dataset_dir = args.dataset_dir.resolve()
    output_dir = args.output_dir.resolve()
    cache_dir = output_dir / "cache"

    if args.batch_size < 0:
        raise ValueError("--batch-size must be >= 0")
    if args.max_batch_pixels < 0:
        raise ValueError("--max-batch-pixels must be >= 0")
    if args.inference_max_side < 0:
        raise ValueError("--inference-max-side must be >= 0")

    class_names = load_class_names(args.classes_yaml.resolve())
    missing_prompt_classes = [name for name in class_names if name not in PROMPT_SPECS]
    if missing_prompt_classes:
        raise ValueError(f"Prompt specs missing for classes: {missing_prompt_classes}")

    if not dataset_dir.exists():
        raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")

    if args.clean:
        print(f"[clean] Removing previous output: {output_dir}")
        reset_output_dir(output_dir)

    ensure_clean_dir(output_dir)
    class_to_idx = {name: idx for idx, name in enumerate(class_names)}
    alias_lookup = build_alias_lookup(class_names)
    full_prompt = " . ".join(str(PROMPT_SPECS[name]["phrase"]) for name in class_names) + " ."

    print(f"[stage] Dataset directory: {dataset_dir}")
    print(f"[stage] Review workspace: {output_dir}")
    print(f"[stage] Class order: {', '.join(class_names)}")

    records, skipped = collect_records(dataset_dir, class_names, args.limit)
    if not records:
        raise RuntimeError("No source images found in the 20 class folders.")
    if skipped:
        print(f"[warn] Skipped unreadable images: {len(skipped)}")

    cached = 0
    pending: list[SourceRecord] = []
    if not args.force_redetect:
        for record in records:
            if try_load_cache(record, cache_dir):
                cached += 1
            else:
                pending.append(record)
    else:
        pending = list(records)

    print(f"[stage] Cached detections reused: {cached}")
    print(f"[stage] Images to detect now: {len(pending)}")

    if pending:
        print("[stage] Loading Grounding DINO model...")
        model, processor, _, local_files_only = load_model(args.model_id)
        runtime = setup_runtime(args, local_files_only)
        model = move_model_to_runtime(model, runtime)
        precision_name = (
            str(runtime.autocast_dtype).split(".")[-1]
            if runtime.use_autocast and runtime.autocast_dtype is not None
            else "fp32"
        )
        print(
            f"[runtime] device={runtime.device} batch_size={runtime.batch_size} "
            f"amp={precision_name} max_batch_pixels={runtime.max_batch_pixels} "
            f"inference_max_side={runtime.inference_max_side}"
        )

        batches = build_inference_batches(
            pending,
            max(1, runtime.batch_size),
            runtime.max_batch_pixels,
        )
        for batch in tqdm(batches, desc="AI prelabel batches", unit="batch"):
            detect_batch(
                batch,
                model,
                processor,
                runtime,
                class_names,
                full_prompt,
                alias_lookup,
                class_to_idx,
                args,
            )
            for record in batch:
                save_cache(record, cache_dir)

    print("[stage] Exporting unified review workspace...")
    write_labels_and_assets(records, output_dir, class_to_idx, args.render_previews)
    write_data_yaml(output_dir, class_names)
    write_manifest(records, output_dir, class_names)
    write_summary(records, output_dir, class_names, skipped)
    write_skipped_report(skipped, output_dir)
    write_readme(output_dir)

    print(f"[done] Images exported to: {output_dir / 'images'}")
    print(f"[done] Labels exported to: {output_dir / 'labels_prelabel'}")
    print(f"[done] Review manifest: {output_dir / 'review_manifest.csv'}")
    if args.render_previews:
        print(f"[done] Annotated previews: {output_dir / 'annotated_previews'}")


if __name__ == "__main__":
    main()
