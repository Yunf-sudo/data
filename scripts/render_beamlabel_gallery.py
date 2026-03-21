from __future__ import annotations

import argparse
import csv
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
DEFAULT_CLASSES = [
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


@dataclass
class YoloBox:
    class_id: int
    x_center: float
    y_center: float
    width: float
    height: float

    @classmethod
    def from_line(cls, line: str) -> "YoloBox | None":
        parts = line.strip().split()
        if len(parts) != 5:
            return None
        try:
            class_id = int(float(parts[0]))
            x_center, y_center, width, height = [float(item) for item in parts[1:]]
        except ValueError:
            return None
        return cls(
            class_id=class_id,
            x_center=max(0.0, min(1.0, x_center)),
            y_center=max(0.0, min(1.0, y_center)),
            width=max(0.0, min(1.0, width)),
            height=max(0.0, min(1.0, height)),
        )

    def to_pixel_box(self, image_width: int, image_height: int) -> tuple[float, float, float, float]:
        half_w = self.width * image_width / 2
        half_h = self.height * image_height / 2
        center_x = self.x_center * image_width
        center_y = self.y_center * image_height
        return (
            center_x - half_w,
            center_y - half_h,
            center_x + half_w,
            center_y + half_h,
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render YOLO labels into annotated images, contact sheets, and class statistics."
    )
    parser.add_argument(
        "--images-dir",
        type=Path,
        default=Path("FengShui_Dataset/beam"),
        help="Directory containing original images.",
    )
    parser.add_argument(
        "--labels-dir",
        type=Path,
        default=Path("FengShui_Dataset/BeamLabel"),
        help="Directory containing YOLO txt label files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("FengShui_Dataset/beamlabel_output"),
        help="Directory where rendered images, sheets, and stats will be saved.",
    )
    parser.add_argument(
        "--data-yaml",
        type=Path,
        default=Path("windows_label_tool/data.yaml"),
        help="Optional data.yaml used to load class names.",
    )
    parser.add_argument(
        "--sheet-cols",
        type=int,
        default=4,
        help="Number of columns per contact sheet.",
    )
    parser.add_argument(
        "--sheet-rows",
        type=int,
        default=4,
        help="Number of rows per contact sheet.",
    )
    parser.add_argument(
        "--thumb-width",
        type=int,
        default=420,
        help="Thumbnail width used in contact sheets.",
    )
    parser.add_argument(
        "--thumb-height",
        type=int,
        default=320,
        help="Thumbnail height used in contact sheets.",
    )
    return parser.parse_args()


def load_class_names(data_yaml: Path) -> list[str]:
    if not data_yaml.exists():
        return list(DEFAULT_CLASSES)

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
        return list(DEFAULT_CLASSES)
    return [mapping[idx] for idx in sorted(mapping)]


def build_image_index(images_dir: Path) -> dict[str, Path]:
    image_index: dict[str, Path] = {}
    for path in sorted(images_dir.iterdir()):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            image_index[path.stem] = path
    return image_index


def load_boxes(label_path: Path) -> tuple[list[YoloBox], int]:
    boxes: list[YoloBox] = []
    invalid_lines = 0
    for raw_line in label_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        box = YoloBox.from_line(line)
        if box is None:
            invalid_lines += 1
            continue
        boxes.append(box)
    return boxes, invalid_lines


def color_for_class(class_id: int) -> tuple[int, int, int]:
    rng = random.Random(class_id * 7919 + 17)
    return (
        64 + rng.randrange(160),
        64 + rng.randrange(160),
        64 + rng.randrange(160),
    )


def fit_text_box(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont) -> tuple[int, int]:
    left, top, right, bottom = draw.textbbox((0, 0), text, font=font)
    return right - left, bottom - top


def draw_boxes(image_path: Path, boxes: list[YoloBox], class_names: list[str], output_path: Path) -> None:
    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    line_width = max(2, round(min(image.size) / 250))
    text_padding = max(2, line_width)

    for box in boxes:
        class_name = (
            class_names[box.class_id]
            if 0 <= box.class_id < len(class_names)
            else f"class_{box.class_id}"
        )
        color = color_for_class(box.class_id)
        x1, y1, x2, y2 = box.to_pixel_box(*image.size)
        x1 = max(0, min(image.width - 1, x1))
        y1 = max(0, min(image.height - 1, y1))
        x2 = max(0, min(image.width - 1, x2))
        y2 = max(0, min(image.height - 1, y2))

        if x2 <= x1 or y2 <= y1:
            continue

        draw.rectangle((x1, y1, x2, y2), outline=color, width=line_width)
        label = f"{box.class_id}:{class_name}"
        text_width, text_height = fit_text_box(draw, label, font)
        label_left = x1
        label_top = max(0, y1 - text_height - text_padding * 2)
        label_box = (
            label_left,
            label_top,
            label_left + text_width + text_padding * 2,
            label_top + text_height + text_padding * 2,
        )
        draw.rectangle(label_box, fill=color)
        draw.text(
            (label_left + text_padding, label_top + text_padding),
            label,
            fill=(255, 255, 255),
            font=font,
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path, quality=95)


def create_contact_sheets(
    rendered_paths: list[Path],
    output_dir: Path,
    cols: int,
    rows: int,
    thumb_width: int,
    thumb_height: int,
) -> list[Path]:
    if not rendered_paths:
        return []

    per_page = cols * rows
    sheet_dir = output_dir / "contact_sheets"
    sheet_dir.mkdir(parents=True, exist_ok=True)
    font = ImageFont.load_default()
    created: list[Path] = []

    for page_index in range(math.ceil(len(rendered_paths) / per_page)):
        page_paths = rendered_paths[page_index * per_page : (page_index + 1) * per_page]
        sheet = Image.new(
            "RGB",
            (cols * thumb_width, rows * (thumb_height + 20)),
            color=(245, 245, 245),
        )
        draw = ImageDraw.Draw(sheet)

        for idx, rendered_path in enumerate(page_paths):
            row = idx // cols
            col = idx % cols
            offset_x = col * thumb_width
            offset_y = row * (thumb_height + 20)

            thumbnail = Image.open(rendered_path).convert("RGB")
            thumbnail.thumbnail((thumb_width, thumb_height))
            paste_x = offset_x + (thumb_width - thumbnail.width) // 2
            paste_y = offset_y + (thumb_height - thumbnail.height) // 2
            sheet.paste(thumbnail, (paste_x, paste_y))

            caption = rendered_path.name
            draw.text((offset_x + 6, offset_y + thumb_height + 2), caption, fill=(0, 0, 0), font=font)

        sheet_path = sheet_dir / f"sheet_{page_index + 1:04d}.jpg"
        sheet.save(sheet_path, quality=92)
        created.append(sheet_path)

    return created


def save_statistics(
    output_dir: Path,
    class_names: list[str],
    instance_counts: dict[int, int],
    image_counts: dict[int, int],
    summary: dict[str, int | str],
) -> None:
    stats_dir = output_dir / "stats"
    stats_dir.mkdir(parents=True, exist_ok=True)

    csv_path = stats_dir / "class_counts.csv"
    with csv_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.writer(handle)
        writer.writerow(["class_id", "class_name", "instance_count", "image_count"])
        for class_id, class_name in enumerate(class_names):
            writer.writerow(
                [
                    class_id,
                    class_name,
                    instance_counts.get(class_id, 0),
                    image_counts.get(class_id, 0),
                ]
            )

    json_payload = {
        "summary": summary,
        "classes": [
            {
                "class_id": class_id,
                "class_name": class_name,
                "instance_count": instance_counts.get(class_id, 0),
                "image_count": image_counts.get(class_id, 0),
            }
            for class_id, class_name in enumerate(class_names)
        ],
    }
    (stats_dir / "class_counts.json").write_text(
        json.dumps(json_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    lines = [
        f"images_dir: {summary['images_dir']}",
        f"labels_dir: {summary['labels_dir']}",
        f"output_dir: {summary['output_dir']}",
        f"total_images_in_images_dir: {summary['total_images_in_images_dir']}",
        f"total_label_files: {summary['total_label_files']}",
        f"matched_images: {summary['matched_images']}",
        f"empty_label_files: {summary['empty_label_files']}",
        f"missing_images: {summary['missing_images']}",
        f"invalid_label_lines: {summary['invalid_label_lines']}",
        "",
        "class statistics:",
    ]
    for class_id, class_name in enumerate(class_names):
        lines.append(
            f"- {class_id:02d} {class_name}: "
            f"instances={instance_counts.get(class_id, 0)}, "
            f"images={image_counts.get(class_id, 0)}"
        )
    (stats_dir / "summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    images_dir = args.images_dir.resolve()
    labels_dir = args.labels_dir.resolve()
    output_dir = args.output_dir.resolve()
    rendered_dir = output_dir / "annotated_images"

    if not images_dir.exists():
        raise FileNotFoundError(f"Images directory not found: {images_dir}")
    if not labels_dir.exists():
        raise FileNotFoundError(f"Labels directory not found: {labels_dir}")

    class_names = load_class_names(args.data_yaml.resolve())
    image_index = build_image_index(images_dir)
    label_paths = sorted(path for path in labels_dir.iterdir() if path.is_file() and path.suffix.lower() == ".txt")

    instance_counts: dict[int, int] = {}
    image_counts: dict[int, int] = {}
    invalid_label_lines = 0
    empty_label_files = 0
    missing_images = 0
    rendered_paths: list[Path] = []

    for label_path in label_paths:
        image_path = image_index.get(label_path.stem)
        if image_path is None:
            missing_images += 1
            continue

        boxes, file_invalid_lines = load_boxes(label_path)
        invalid_label_lines += file_invalid_lines
        if not boxes:
            empty_label_files += 1

        class_ids_in_image: set[int] = set()
        for box in boxes:
            instance_counts[box.class_id] = instance_counts.get(box.class_id, 0) + 1
            class_ids_in_image.add(box.class_id)
        for class_id in class_ids_in_image:
            image_counts[class_id] = image_counts.get(class_id, 0) + 1

        output_path = rendered_dir / f"{image_path.stem}.jpg"
        draw_boxes(image_path, boxes, class_names, output_path)
        rendered_paths.append(output_path)

    sheet_paths = create_contact_sheets(
        rendered_paths=rendered_paths,
        output_dir=output_dir,
        cols=max(1, args.sheet_cols),
        rows=max(1, args.sheet_rows),
        thumb_width=max(80, args.thumb_width),
        thumb_height=max(80, args.thumb_height),
    )

    summary = {
        "images_dir": str(images_dir),
        "labels_dir": str(labels_dir),
        "output_dir": str(output_dir),
        "total_images_in_images_dir": len(image_index),
        "total_label_files": len(label_paths),
        "matched_images": len(rendered_paths),
        "empty_label_files": empty_label_files,
        "missing_images": missing_images,
        "invalid_label_lines": invalid_label_lines,
        "contact_sheet_pages": len(sheet_paths),
    }
    save_statistics(output_dir, class_names, instance_counts, image_counts, summary)

    print(f"Rendered images: {len(rendered_paths)}")
    print(f"Contact sheet pages: {len(sheet_paths)}")
    print(f"Statistics directory: {output_dir / 'stats'}")
    print(f"Annotated images directory: {rendered_dir}")


if __name__ == "__main__":
    main()
