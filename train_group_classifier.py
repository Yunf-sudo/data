"""
Train grouped classifiers on raw or masked FengShui crops.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import warnings

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms
from torchvision.datasets.folder import default_loader
from torchvision.models import (
    EfficientNet_B0_Weights,
    EfficientNet_V2_S_Weights,
    MobileNet_V3_Small_Weights,
    ResNet18_Weights,
)

from group_classifier_config import DEFAULT_CLASSIFIER_RUNS, DEFAULT_MASKED_CLS_OUTPUT, GROUPS

warnings.filterwarnings("ignore", category=UserWarning, module="PIL.Image")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train lightweight grouped classifiers.")
    parser.add_argument("--data-root", default=DEFAULT_MASKED_CLS_OUTPUT, help="Root grouped dataset directory.")
    parser.add_argument("--group", choices=sorted(GROUPS), help="Single group to train.")
    parser.add_argument("--all-groups", action="store_true", help="Train every configured group.")
    parser.add_argument("--epochs", type=int, default=20, help="Epoch count.")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size.")
    parser.add_argument("--image-size", type=int, default=224, help="Classifier input size.")
    parser.add_argument("--workers", type=int, default=8, help="Data loader workers.")
    parser.add_argument(
        "--model",
        choices=("mobilenet_v3_small", "efficientnet_b0", "efficientnet_v2_s", "resnet18"),
        default="mobilenet_v3_small",
        help="Backbone architecture.",
    )
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate.")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="AdamW weight decay.")
    parser.add_argument("--label-smoothing", type=float, default=0.05, help="CrossEntropy label smoothing.")
    parser.add_argument("--dropout", type=float, default=0.2, help="Classifier dropout.")
    parser.add_argument("--patience", type=int, default=5, help="Early stopping patience.")
    parser.add_argument("--min-lr", type=float, default=1e-6, help="Minimum LR for cosine decay.")
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:0.")
    parser.add_argument("--project", default=DEFAULT_CLASSIFIER_RUNS, help="Output root for trained models.")
    parser.add_argument("--pretrained", action="store_true", help="Use ImageNet pretrained MobileNetV3 weights.")
    parser.add_argument("--export-ts", action="store_true", help="Export a TorchScript copy of the best model.")
    return parser.parse_args()


def selected_groups(args: argparse.Namespace) -> list[str]:
    if args.all_groups:
        return list(GROUPS)
    if args.group:
        return [args.group]
    raise SystemExit("Choose either --group <name> or --all-groups.")


def resolve_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def group_paths(data_root: Path, group: str) -> tuple[Path, Path, Path]:
    group_root = data_root / group
    return group_root / "train", group_root / "val", group_root / "test"


def build_transforms(image_size: int):
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    train_tf = transforms.Compose(
        [
            transforms.RandomResizedCrop(image_size, scale=(0.72, 1.0), ratio=(0.85, 1.15)),
            transforms.RandomHorizontalFlip(),
            transforms.RandAugment(num_ops=2, magnitude=7),
            transforms.ColorJitter(brightness=0.18, contrast=0.18, saturation=0.18),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
            transforms.RandomErasing(p=0.2, scale=(0.02, 0.12)),
        ]
    )
    eval_tf = transforms.Compose(
        [
            transforms.Resize(int(image_size * 1.14)),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )
    return train_tf, eval_tf


def build_loaders(train_dir: Path, val_dir: Path, test_dir: Path, image_size: int, batch_size: int, workers: int):
    train_tf, eval_tf = build_transforms(image_size)
    train_ds = datasets.ImageFolder(str(train_dir), transform=train_tf, loader=default_loader)
    val_ds = datasets.ImageFolder(str(val_dir), transform=eval_tf, loader=default_loader)
    test_ds = datasets.ImageFolder(str(test_dir), transform=eval_tf, loader=default_loader)

    if len(train_ds) == 0:
        raise FileNotFoundError(f"No training images found in {train_dir}")
    if len(val_ds) == 0:
        val_ds = datasets.ImageFolder(str(train_dir), transform=eval_tf, loader=default_loader)
    if len(test_ds) == 0:
        test_ds = datasets.ImageFolder(str(val_dir if val_dir.exists() else train_dir), transform=eval_tf, loader=default_loader)

    pin_memory = torch.cuda.is_available()
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=workers, pin_memory=pin_memory)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=workers, pin_memory=pin_memory)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=workers, pin_memory=pin_memory)
    return train_ds, val_ds, test_ds, train_loader, val_loader, test_loader


def build_model(model_name: str, num_classes: int, dropout: float, pretrained: bool) -> nn.Module:
    if model_name == "mobilenet_v3_small":
        weights = MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
        model = models.mobilenet_v3_small(weights=weights)
        in_features = model.classifier[3].in_features
        model.classifier[3] = nn.Linear(in_features, num_classes)
        if dropout != 0.2:
            model.classifier[2] = nn.Dropout(p=dropout, inplace=True)
        return model

    if model_name == "efficientnet_b0":
        weights = EfficientNet_B0_Weights.DEFAULT if pretrained else None
        model = models.efficientnet_b0(weights=weights)
        in_features = model.classifier[1].in_features
        model.classifier = nn.Sequential(nn.Dropout(p=dropout, inplace=True), nn.Linear(in_features, num_classes))
        return model

    if model_name == "efficientnet_v2_s":
        weights = EfficientNet_V2_S_Weights.DEFAULT if pretrained else None
        model = models.efficientnet_v2_s(weights=weights)
        in_features = model.classifier[1].in_features
        model.classifier = nn.Sequential(nn.Dropout(p=dropout, inplace=True), nn.Linear(in_features, num_classes))
        return model

    if model_name == "resnet18":
        weights = ResNet18_Weights.DEFAULT if pretrained else None
        model = models.resnet18(weights=weights)
        in_features = model.fc.in_features
        model.fc = nn.Sequential(nn.Dropout(p=dropout, inplace=True), nn.Linear(in_features, num_classes))
        return model

    raise ValueError(f"Unsupported model: {model_name}")


def run_epoch(model, loader, criterion, optimizer, device, train: bool, scaler):
    if train:
        model.train()
    else:
        model.eval()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    use_amp = scaler is not None

    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        if train:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(train):
            if use_amp:
                with torch.autocast(device_type=device.type, enabled=True):
                    outputs = model(images)
                    loss = criterion(outputs, targets)
            else:
                outputs = model(images)
                loss = criterion(outputs, targets)

            if train:
                if scaler is not None:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()

        predictions = outputs.argmax(dim=1)
        total_loss += float(loss.item()) * targets.size(0)
        total_correct += int((predictions == targets).sum().item())
        total_samples += int(targets.size(0))

    return {
        "loss": total_loss / max(1, total_samples),
        "acc": total_correct / max(1, total_samples),
        "samples": total_samples,
    }


@torch.no_grad()
def evaluate_with_confusion(model, loader, device, class_names: list[str]) -> dict[str, object]:
    model.eval()
    confusion = torch.zeros((len(class_names), len(class_names)), dtype=torch.int64)
    total_correct = 0
    total_samples = 0

    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        outputs = model(images)
        predictions = outputs.argmax(dim=1)
        total_correct += int((predictions == targets).sum().item())
        total_samples += int(targets.size(0))
        for truth, pred in zip(targets.cpu(), predictions.cpu()):
            confusion[int(truth), int(pred)] += 1

    per_class = {}
    for index, name in enumerate(class_names):
        row_sum = int(confusion[index].sum().item())
        per_class[name] = {
            "total": row_sum,
            "correct": int(confusion[index, index].item()),
            "accuracy": (int(confusion[index, index].item()) / row_sum) if row_sum else 0.0,
        }

    return {
        "accuracy": total_correct / max(1, total_samples),
        "samples": total_samples,
        "confusion_matrix": confusion.tolist(),
        "per_class": per_class,
    }


def save_checkpoint(model, class_names: list[str], target: Path, architecture: str, image_size: int) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "state_dict": model.state_dict(),
        "class_names": class_names,
        "architecture": architecture,
        "image_size": image_size,
    }
    torch.save(payload, target)


def maybe_export_torchscript(model: nn.Module, image_size: int, target: Path, device: torch.device) -> None:
    example = torch.randn(1, 3, image_size, image_size, device=device)
    scripted = torch.jit.trace(model, example)
    scripted.save(str(target))


def train_group(args: argparse.Namespace, group: str, device: torch.device) -> Path:
    data_root = Path(args.data_root)
    train_dir, val_dir, test_dir = group_paths(data_root, group)
    train_ds, val_ds, test_ds, train_loader, val_loader, test_loader = build_loaders(
        train_dir=train_dir,
        val_dir=val_dir,
        test_dir=test_dir,
        image_size=args.image_size,
        batch_size=args.batch_size,
        workers=args.workers,
    )

    class_names = train_ds.classes
    model = build_model(
        model_name=args.model,
        num_classes=len(class_names),
        dropout=args.dropout,
        pretrained=args.pretrained,
    ).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.amp.GradScaler("cuda") if device.type == "cuda" else None
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(1, args.epochs),
        eta_min=args.min_lr,
    )

    project_root = Path(args.project) / group
    project_root.mkdir(parents=True, exist_ok=True)
    best_path = project_root / "best.pt"
    history: list[dict[str, float | int]] = []

    best_acc = -1.0
    epochs_without_improve = 0
    start_time = time.time()

    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(model, train_loader, criterion, optimizer, device, True, scaler)
        val_metrics = run_epoch(model, val_loader, criterion, optimizer, device, False, None)
        row = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_acc": train_metrics["acc"],
            "val_loss": val_metrics["loss"],
            "val_acc": val_metrics["acc"],
        }
        history.append(row)
        print(
            f"[{group}] epoch {epoch}/{args.epochs} "
            f"train_acc={train_metrics['acc']:.4f} val_acc={val_metrics['acc']:.4f} lr={optimizer.param_groups[0]['lr']:.6f}"
        )

        if val_metrics["acc"] > best_acc:
            best_acc = float(val_metrics["acc"])
            epochs_without_improve = 0
            save_checkpoint(model, class_names, best_path, architecture=args.model, image_size=args.image_size)
        else:
            epochs_without_improve += 1

        scheduler.step()

        if epochs_without_improve >= args.patience:
            print(f"[{group}] early stop after {epoch} epochs")
            break

    checkpoint = torch.load(best_path, map_location=device)
    model.load_state_dict(checkpoint["state_dict"])
    test_metrics = evaluate_with_confusion(model, test_loader, device, class_names)

    metrics = {
        "group": group,
        "class_names": class_names,
        "model": args.model,
        "image_size": args.image_size,
        "train_samples": len(train_ds),
        "val_samples": len(val_ds),
        "test_samples": len(test_ds),
        "best_val_acc": best_acc,
        "test": test_metrics,
        "seconds": time.time() - start_time,
        "history": history,
    }
    (project_root / "metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (project_root / "labels.json").write_text(json.dumps(class_names, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    if args.export_ts:
        maybe_export_torchscript(model, args.image_size, project_root / "best.ts", device)

    print(f"[{group}] best model -> {best_path.resolve()}")
    print(f"[{group}] test acc  -> {test_metrics['accuracy']:.4f}")
    return best_path


def main(args: argparse.Namespace) -> None:
    device = resolve_device(args.device)
    groups = selected_groups(args)
    results = {}
    for group in groups:
        results[group] = str(train_group(args, group, device))

    print("[done] trained groups:")
    for group, path in results.items():
        print(f"  {group}: {path}")


if __name__ == "__main__":
    main(parse_args())
