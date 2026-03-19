"""
Train a lightweight segmentation model on the SAM-generated dataset.

Example:
    python train_light_seg.py --data FengShui_SAM_5cls/data.yaml
"""

from __future__ import annotations

import argparse
import os
import tempfile
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a lightweight YOLO segmentation model.")
    parser.add_argument(
        "--data",
        default="FengShui_SAM_5cls/data.yaml",
        help="Path to the YOLO segmentation data.yaml file.",
    )
    parser.add_argument(
        "--model",
        default="yolov8n-seg.pt",
        help="Lightweight segmentation checkpoint to fine-tune.",
    )
    parser.add_argument("--epochs", type=int, default=80, help="Training epochs.")
    parser.add_argument("--imgsz", type=int, default=640, help="Training image size.")
    parser.add_argument("--batch", type=int, default=16, help="Batch size.")
    parser.add_argument(
        "--device",
        default="auto",
        help="Device string understood by Ultralytics, for example auto, cpu, 0.",
    )
    parser.add_argument("--workers", type=int, default=8, help="Data loader workers.")
    parser.add_argument("--project", default="runs/segment", help="Training project directory.")
    parser.add_argument("--name", default="fengshui_5cls_nseg", help="Run name.")
    parser.add_argument("--patience", type=int, default=20, help="Early stopping patience.")
    parser.add_argument(
        "--cache",
        choices=("false", "ram", "disk"),
        default="disk",
        help="Dataset cache mode.",
    )
    parser.add_argument(
        "--export",
        choices=("none", "onnx"),
        default="none",
        help="Export format after training.",
    )
    return parser.parse_args()


def resolve_device(device: str) -> str | int:
    if device != "auto":
        return device

    try:
        import torch

        return 0 if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def image_count(image_dir: Path) -> int:
    suffixes = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    if not image_dir.exists():
        return 0
    return sum(1 for path in image_dir.iterdir() if path.suffix.lower() in suffixes)


def resolve_data_yaml(data_path: Path) -> Path:
    root = data_path.parent
    train_count = image_count(root / "images" / "train")
    val_count = image_count(root / "images" / "val")

    if train_count == 0:
        raise FileNotFoundError(f"No training images found under: {root / 'images' / 'train'}")
    if val_count > 0:
        return data_path

    text = data_path.read_text(encoding="utf-8")
    lines = text.splitlines()
    patched = []
    for line in lines:
        if line.startswith("val:"):
            patched.append("val: images/train")
        else:
            patched.append(line)

    temp_dir = Path(tempfile.mkdtemp(prefix="fengshui_seg_yaml_"))
    patched_path = temp_dir / data_path.name
    patched_path.write_text("\n".join(patched) + "\n", encoding="utf-8")
    print(f"[warn] Validation split is empty. Falling back to train split for validation: {patched_path}")
    return patched_path


def main(args: argparse.Namespace) -> None:
    try:
        from ultralytics import YOLO
    except ModuleNotFoundError as exc:
        raise SystemExit("Missing ultralytics. Install with: pip install ultralytics") from exc

    data_path = Path(args.data).resolve()
    if not data_path.exists():
        raise FileNotFoundError(f"Missing data yaml: {data_path}")
    data_path = resolve_data_yaml(data_path)

    device = resolve_device(args.device)
    cache = False if args.cache == "false" else args.cache

    model = YOLO(args.model)
    model.train(
        data=str(data_path),
        task="segment",
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=device,
        workers=args.workers,
        project=args.project,
        name=args.name,
        patience=args.patience,
        cache=cache,
        pretrained=True,
    )

    best_path = Path(model.trainer.best)
    print(f"[done] Best checkpoint: {best_path.resolve()}")

    if args.export == "none":
        return

    exported_model = YOLO(str(best_path))
    try:
        exported_model.export(format=args.export, imgsz=args.imgsz, device=device)
    except Exception as exc:
        print(f"[warn] Export failed but training weights are available: {exc}")


if __name__ == "__main__":
    main(parse_args())
