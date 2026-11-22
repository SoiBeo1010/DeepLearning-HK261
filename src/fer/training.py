"""Reusable FER2013 training with a separate validation split and raw checkpoints."""

import argparse
import csv
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import random

import numpy as np
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from tqdm.auto import tqdm

from fer.config import EMOTION_LABELS, IMAGE_SIZE, NORMALIZE_MEAN, NORMALIZE_STD
from fer.model import CCT_ResNet
from fer.predictor import resolve_device


@dataclass(frozen=True)
class TrainConfig:
    data_dir: Path
    output_dir: Path = Path("outputs/training")
    epochs: int = 50
    batch_size: int = 64
    lr: float = 5e-4
    backbone_lr: float = 1e-5
    weight_decay: float = 1e-4
    validation_fraction: float = 0.1
    num_workers: int = 0
    seed: int = 42
    device: str = "auto"
    pretrained: bool = True
    focal_gamma: float = 2.0
    label_smoothing: float = 0.05
    curriculum_threshold: float = 0.5
    mixup_alpha: float = 0.2

    def __post_init__(self) -> None:
        if self.epochs < 1 or self.batch_size < 2 or self.num_workers < 0:
            raise ValueError("Use epochs >= 1, batch_size >= 2 and num_workers >= 0.")
        if not 0 < self.validation_fraction < 1:
            raise ValueError("validation_fraction must be between zero and one.")
        if self.lr <= 0 or self.backbone_lr <= 0 or self.weight_decay < 0:
            raise ValueError("Learning rates must be positive; weight_decay must be nonnegative.")
        if not 0 <= self.label_smoothing < 1 or self.focal_gamma < 0:
            raise ValueError("Use 0 <= label_smoothing < 1 and focal_gamma >= 0.")
        if not 0 <= self.curriculum_threshold <= 1 or self.mixup_alpha < 0:
            raise ValueError("Use 0 <= curriculum_threshold <= 1 and mixup_alpha >= 0.")


class FocalLoss(nn.Module):
    def __init__(self, gamma: float = 2.0, label_smoothing: float = 0.05) -> None:
        super().__init__()
        self.gamma = gamma
        self.label_smoothing = label_smoothing

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce = F.cross_entropy(inputs, targets, reduction="none")
        smooth_ce = F.cross_entropy(
            inputs, targets, reduction="none", label_smoothing=self.label_smoothing
        )
        return ((1 - torch.exp(-ce)).pow(self.gamma) * smooth_ce).mean()


def build_transforms(training: bool = False) -> transforms.Compose:
    operations = [
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.Grayscale(num_output_channels=3),
    ]
    if training:
        operations.extend([transforms.RandomHorizontalFlip(), transforms.RandomRotation(10)])
    operations.extend(
        [transforms.ToTensor(), transforms.Normalize(NORMALIZE_MEAN, NORMALIZE_STD)]
    )
    if training:
        operations.append(
            transforms.RandomErasing(p=0.1, scale=(0.02, 0.1), ratio=(0.3, 3.3))
        )
    return transforms.Compose(operations)


def check_classes(dataset: datasets.ImageFolder) -> None:
    expected = [label.lower() for label in EMOTION_LABELS]
    if dataset.classes != expected:
        raise ValueError(f"Expected class folders {expected}, found {dataset.classes}.")


def build_loaders(config: TrainConfig) -> tuple[DataLoader, DataLoader]:
    train_dir = Path(config.data_dir) / "train"
    val_dir = Path(config.data_dir) / "val"
    train_dataset = datasets.ImageFolder(train_dir, transform=build_transforms(training=True))
    check_classes(train_dataset)
    if val_dir.is_dir():
        val_dataset = datasets.ImageFolder(val_dir, transform=build_transforms())
        check_classes(val_dataset)
        train_subset = train_dataset
        val_subset = val_dataset
    else:
        # Split only train/. test/ is reserved for the final evaluation.
        train_indices, val_indices = train_test_split(
            np.arange(len(train_dataset)), test_size=config.validation_fraction,
            random_state=config.seed, stratify=train_dataset.targets,
        )
        val_dataset = datasets.ImageFolder(train_dir, transform=build_transforms())
        train_subset = Subset(train_dataset, train_indices.tolist())
        val_subset = Subset(val_dataset, val_indices.tolist())
    options = {
        "batch_size": config.batch_size,
        "num_workers": config.num_workers,
        "pin_memory": resolve_device(config.device).type == "cuda",
    }
    generator = torch.Generator().manual_seed(config.seed)
    return (
        DataLoader(train_subset, shuffle=True, generator=generator, **options),
        DataLoader(val_subset, shuffle=False, **options),
    )


def train_epoch(model, loader, optimizer, criterion, device, mixup_alpha: float) -> dict:
    model.train()
    loss_total = correct = total = 0.0
    for inputs, targets in tqdm(loader, desc="Train", leave=False):
        inputs, targets = inputs.to(device), targets.to(device)
        lam = float(np.random.beta(mixup_alpha, mixup_alpha)) if mixup_alpha > 0 else 1.0
        index = torch.randperm(inputs.size(0), device=device)
        mixed_inputs = lam * inputs + (1 - lam) * inputs[index]
        optimizer.zero_grad(set_to_none=True)
        logits = model(mixed_inputs)
        loss = lam * criterion(logits, targets) + (1 - lam) * criterion(logits, targets[index])
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        predictions = logits.argmax(dim=1)
        correct += lam * predictions.eq(targets).sum().item()
        correct += (1 - lam) * predictions.eq(targets[index]).sum().item()
        total += targets.size(0)
        loss_total += loss.item() * targets.size(0)
    return {"loss": loss_total / total, "accuracy": correct / total}


@torch.inference_mode()
def evaluate(model, loader, device, criterion=None) -> dict:
    model.eval()
    targets_all, predictions_all = [], []
    loss_total = 0.0
    for inputs, targets in tqdm(loader, desc="Evaluate", leave=False):
        inputs, targets = inputs.to(device), targets.to(device)
        logits = model(inputs)
        if criterion is not None:
            loss_total += criterion(logits, targets).item() * targets.size(0)
        targets_all.extend(targets.cpu().tolist())
        predictions_all.extend(logits.argmax(dim=1).cpu().tolist())
    if not targets_all:
        raise ValueError("The evaluation dataset is empty.")
    return {
        "loss": loss_total / len(targets_all) if criterion is not None else None,
        "accuracy": float(np.mean(np.array(targets_all) == predictions_all)),
        "macro_f1": float(f1_score(
            targets_all, predictions_all, labels=list(range(len(EMOTION_LABELS))),
            average="macro", zero_division=0,
        )),
        "targets": targets_all,
        "predictions": predictions_all,
    }


def train(config: TrainConfig) -> dict:
    output_dir = Path(config.output_dir)
    checkpoint = output_dir / "cct_resnet_fer2013_best.pth"
    # Check before initializing the backbone, which may download ImageNet weights.
    if checkpoint.exists():
        raise FileExistsError(f"Checkpoint already exists: {checkpoint}. Choose a new output_dir.")
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    device = resolve_device(config.device)
    train_loader, val_loader = build_loaders(config)
    model = CCT_ResNet(pretrained=config.pretrained, dropout=0.1).to(device)
    nn.init.normal_(model.positional_emb, std=0.02)
    criterion = FocalLoss(config.focal_gamma, config.label_smoothing)
    optimizer = torch.optim.AdamW(
        [
            {"params": model.tokenizer.parameters(), "lr": config.backbone_lr},
            {"params": [p for name, p in model.named_parameters()
                        if not name.startswith("tokenizer.")], "lr": config.lr},
        ],
        weight_decay=config.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.epochs)
    output_dir.mkdir(parents=True, exist_ok=True)
    best_accuracy = -1.0
    hard_mode = False
    history = []
    print(f"Device: {device} | Train: {len(train_loader.dataset)} | Val: {len(val_loader.dataset)}")
    for epoch in range(1, config.epochs + 1):
        if best_accuracy >= config.curriculum_threshold:
            hard_mode = True
        base_dataset = train_loader.dataset
        if isinstance(base_dataset, Subset):
            base_dataset = base_dataset.dataset
        base_dataset.transform.transforms[-1].p = 0.5 if hard_mode else 0.1
        training = train_epoch(
            model, train_loader, optimizer, criterion, device,
            config.mixup_alpha if hard_mode else 0.0,
        )
        validation = evaluate(model, val_loader, device, criterion)
        scheduler.step()
        row = {
            "epoch": epoch, "train_loss": training["loss"],
            "train_accuracy": training["accuracy"], "val_loss": validation["loss"],
            "val_accuracy": validation["accuracy"], "val_macro_f1": validation["macro_f1"],
        }
        history.append(row)
        if validation["accuracy"] > best_accuracy:
            best_accuracy = validation["accuracy"]
            torch.save(model.state_dict(), checkpoint)
        with (output_dir / "history.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(row))
            writer.writeheader()
            writer.writerows(history)
        metadata = {
            "config": asdict(config), "classes": [s.lower() for s in EMOTION_LABELS],
            "best_val_accuracy": best_accuracy,
            "train_size": len(train_loader.dataset), "val_size": len(val_loader.dataset),
        }
        (output_dir / "training.json").write_text(
            json.dumps(metadata, indent=2, default=str) + "\n", encoding="utf-8"
        )
        print(
            f"Epoch {epoch}/{config.epochs} | Train loss {training['loss']:.4f} | "
            f"Val loss {validation['loss']:.4f} | Val acc {validation['accuracy']:.2%} | "
            f"Val macro F1 {validation['macro_f1']:.2%}"
        )
    return {"checkpoint": checkpoint, "best_val_accuracy": best_accuracy, "history": history}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train CCT + ResNet18 on FER2013 ImageFolder data.")
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/training"))
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--validation-fraction", type=float, default=0.1)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    parser.add_argument("--no-pretrained", action="store_true")
    args = parser.parse_args(argv)
    try:
        train(TrainConfig(
            data_dir=args.data_dir, output_dir=args.output_dir,
            epochs=args.epochs, batch_size=args.batch_size, num_workers=args.num_workers,
            validation_fraction=args.validation_fraction, device=args.device,
            pretrained=not args.no_pretrained,
        ))
    except (OSError, ValueError, RuntimeError) as exc:
        parser.exit(1, f"Training failed: {exc}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
