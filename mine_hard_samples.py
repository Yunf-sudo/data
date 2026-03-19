"""
Mine hard samples for manual review from grouped classification datasets.

The script loads one or more trained group classifiers, scores a dataset split,
and exports:
- full predictions csv
- hard-sample csv
- summary json
- optional copied/hardlinked review images
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets.folder import default_loader
from torchvision.transforms import Compose, Normalize, Resize, CenterCrop, ToTensor

from group_classifier_config import GROUPS
from train_group_classifier import build_model, resolve_device


DEFAULT_MODEL_PATHS = {
    "plant": "runs/group_cls_effb0_v1/plant/best.pt",
    "table": "runs/group_cls_effb0_v1/table/best.pt",
    "door": "runs/group_cls_efficientv2s_v1/door/best.pt",
    "window": "runs/group_cls_effb0_v1/window/best.pt",
}


@dataclass
class SampleRecord:
    path: Path
    split: str
    true_label: str
    pred_label: str
    confidence: float
    margin: float
    error: bool
    reasons: list[str]


class PathImageDataset(Dataset):
    def __init__(self, samples: list[tuple[Path, int, str]], transform):
        self.samples = samples
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        path, target, split = self.samples[index]
        image = default_loader(str(path))
        image = self.transform(image)
        return image, target, str(path), split


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Mine hard samples from grouped classifiers.")
    parser.add_argument("--data-root", default="FengShui_GroupCls_Raw", help="Grouped dataset root.")
    parser.add_argument("--output-dir", default="review_manifests", help="Review manifest root.")
    parser.add_argument("--group", choices=sorted(GROUPS), nargs="+", help="Groups to mine. Default mines plant/door/window.")
    parser.add_argument("--batch-size", type=int, default=64, help="Inference batch size.")
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:0.")
    parser.add_argument("--num-workers", type=int, default=0, help="Loader workers. Default 0 for stability.")
    parser.add_argument("--confidence-threshold", type=float, default=0.70, help="Low-confidence threshold.")
    parser.add_argument("--margin-threshold", type=float, default=0.20, help="Low-margin threshold between top-1 and top-2.")
    parser.add_argument("--top-k-copy", type=int, default=120, help="Copy up to this many hardest samples per group.")
    parser.add_argument("--copy-mode", choices=("hardlink", "copy"), default="hardlink", help="Review image export mode.")
    return parser.parse_args()


def selected_groups(args: argparse.Namespace) -> list[str]:
    if args.group:
        return args.group
    return ["plant", "door", "window"]


def build_eval_transform(image_size: int):
    return Compose(
        [
            Resize(int(image_size * 1.14)),
            CenterCrop(image_size),
            ToTensor(),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


def list_group_samples(group_root: Path, class_names: list[str]) -> list[tuple[Path, int, str]]:
    samples: list[tuple[Path, int, str]] = []
    for split in ("train", "val", "test"):
        split_root = group_root / split
        if not split_root.exists():
            continue
        for class_index, class_name in enumerate(class_names):
            class_dir = split_root / class_name
            if not class_dir.exists():
                continue
            for path in sorted(class_dir.iterdir()):
                if path.is_file():
                    samples.append((path, class_index, split))
    return samples


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


def load_checkpoint(model_path: Path, device: torch.device):
    checkpoint = torch.load(model_path, map_location=device)
    class_names = checkpoint["class_names"]
    architecture = checkpoint.get("architecture", "mobilenet_v3_small")
    image_size = int(checkpoint.get("image_size", 224))
    model = build_model(
        model_name=architecture,
        num_classes=len(class_names),
        dropout=0.2,
        pretrained=False,
    ).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model, class_names, architecture, image_size


def mine_group(args: argparse.Namespace, group: str, device: torch.device) -> dict[str, object]:
    model_path = Path(DEFAULT_MODEL_PATHS[group])
    if not model_path.exists():
        raise FileNotFoundError(f"Missing model for group {group}: {model_path}")

    model, class_names, architecture, image_size = load_checkpoint(model_path, device)
    group_root = Path(args.data_root) / group
    samples = list_group_samples(group_root, class_names)
    dataset = PathImageDataset(samples, transform=build_eval_transform(image_size))
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    records: list[SampleRecord] = []
    with torch.no_grad():
        for images, targets, paths, splits in loader:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            logits = model(images)
            probs = torch.softmax(logits, dim=1)
            top2_probs, top2_idx = torch.topk(probs, k=min(2, probs.shape[1]), dim=1)

            for i in range(images.size(0)):
                true_idx = int(targets[i].item())
                pred_idx = int(top2_idx[i, 0].item())
                confidence = float(top2_probs[i, 0].item())
                second_prob = float(top2_probs[i, 1].item()) if top2_probs.shape[1] > 1 else 0.0
                margin = confidence - second_prob
                error = pred_idx != true_idx
                reasons: list[str] = []
                if error:
                    reasons.append("misclassified")
                if confidence < args.confidence_threshold:
                    reasons.append("low_confidence")
                if margin < args.margin_threshold:
                    reasons.append("low_margin")
                if not reasons:
                    continue

                records.append(
                    SampleRecord(
                        path=Path(paths[i]),
                        split=str(splits[i]),
                        true_label=class_names[true_idx],
                        pred_label=class_names[pred_idx],
                        confidence=confidence,
                        margin=margin,
                        error=error,
                        reasons=reasons,
                    )
                )

    def priority_key(item: SampleRecord):
        return (
            0 if item.error else 1,
            item.confidence,
            item.margin,
            item.split != "train",
        )

    records.sort(key=priority_key)

    output_root = Path(args.output_dir) / group
    output_root.mkdir(parents=True, exist_ok=True)
    full_csv = output_root / "hard_samples.csv"
    with full_csv.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.writer(handle)
        writer.writerow(["path", "split", "true_label", "pred_label", "confidence", "margin", "error", "reasons"])
        for item in records:
            writer.writerow(
                [
                    str(item.path.resolve()),
                    item.split,
                    item.true_label,
                    item.pred_label,
                    f"{item.confidence:.6f}",
                    f"{item.margin:.6f}",
                    int(item.error),
                    "|".join(item.reasons),
                ]
            )

    review_dir = output_root / "review_images"
    for index, item in enumerate(records[: args.top_k_copy], start=1):
        prefixed_name = f"{index:03d}_{item.split}_{item.true_label}_as_{item.pred_label}_{item.path.name}"
        link_or_copy(item.path, review_dir / prefixed_name, args.copy_mode)

    by_reason: dict[str, int] = {}
    by_true_label: dict[str, int] = {}
    misclassified_by_pair: dict[str, int] = {}
    for item in records:
        for reason in item.reasons:
            by_reason[reason] = by_reason.get(reason, 0) + 1
        by_true_label[item.true_label] = by_true_label.get(item.true_label, 0) + 1
        if item.error:
            key = f"{item.true_label}->{item.pred_label}"
            misclassified_by_pair[key] = misclassified_by_pair.get(key, 0) + 1

    summary = {
        "group": group,
        "model_path": str(model_path.resolve()),
        "architecture": architecture,
        "image_size": image_size,
        "total_flagged": len(records),
        "misclassified": sum(1 for item in records if item.error),
        "by_reason": by_reason,
        "by_true_label": by_true_label,
        "misclassified_by_pair": dict(sorted(misclassified_by_pair.items(), key=lambda kv: kv[1], reverse=True)),
        "top_k_copy": min(args.top_k_copy, len(records)),
    }
    (output_root / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return summary


def main(args: argparse.Namespace) -> None:
    device = resolve_device(args.device)
    groups = selected_groups(args)
    combined = {}
    for group in groups:
        print(f"[mine] {group}")
        combined[group] = mine_group(args, group, device)

    root = Path(args.output_dir)
    root.mkdir(parents=True, exist_ok=True)
    (root / "summary.json").write_text(json.dumps(combined, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"[done] review manifests written to: {root.resolve()}")


if __name__ == "__main__":
    main(parse_args())
