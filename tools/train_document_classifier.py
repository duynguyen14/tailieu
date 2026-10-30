from __future__ import annotations

import argparse
import csv
import json
import random
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageOps
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from tqdm.auto import tqdm

from tools.document_classifier_config import DocumentClassifierSettings
from tools.document_classifier_model import build_mobilenet_v3_small


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class PadToSquare:
    def __init__(self, fill: int = 255) -> None:
        self.fill = fill

    def __call__(self, image: Image.Image) -> Image.Image:
        width, height = image.size
        side = max(width, height)
        horizontal = side - width
        vertical = side - height
        border = (
            horizontal // 2,
            vertical // 2,
            horizontal - horizontal // 2,
            vertical - vertical // 2,
        )
        return ImageOps.expand(image, border=border, fill=self.fill)


class ManifestDataset(Dataset[tuple[torch.Tensor, int]]):
    def __init__(
        self,
        *,
        dataset_root: Path,
        manifest_path: Path,
        class_to_index: dict[str, int],
        transform: transforms.Compose,
    ) -> None:
        self.dataset_root = dataset_root
        self.class_to_index = class_to_index
        self.transform = transform
        with manifest_path.open("r", encoding="utf-8-sig", newline="") as source:
            self.rows = list(csv.DictReader(source))
        if not self.rows:
            raise RuntimeError(f"Manifest has no records: {manifest_path}")

        for row in self.rows:
            label = row.get("label", "")
            if label not in class_to_index:
                raise RuntimeError(f"Unknown label {label!r} in {manifest_path}")
            image_path = (dataset_root / row["image_path"]).resolve()
            if not image_path.is_relative_to(dataset_root) or not image_path.is_file():
                raise RuntimeError(f"Missing or unsafe image path in manifest: {row['image_path']}")

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        row = self.rows[index]
        image_path = (self.dataset_root / row["image_path"]).resolve()
        with Image.open(image_path) as image:
            rgb_image = image.convert("RGB")
        return self.transform(rgb_image), self.class_to_index[row["label"]]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the document page classifier locally.")
    parser.add_argument("--dataset-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default=None)
    parser.add_argument("--resume", type=Path, default=None)
    return parser.parse_args()


def set_seed(seed: int, deterministic: bool) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = not deterministic
    torch.backends.cudnn.deterministic = deterministic
    if deterministic:
        torch.use_deterministic_algorithms(True)


def select_device(preference: str, gpu_device_id: int) -> torch.device:
    if preference == "cpu":
        return torch.device("cpu")
    if preference == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("TRAIN_DEVICE=cuda but CUDA is not available to PyTorch.")
    if torch.cuda.is_available() and preference != "cpu":
        if gpu_device_id >= torch.cuda.device_count():
            raise RuntimeError(
                f"TRAIN_GPU_DEVICE_ID={gpu_device_id} but only "
                f"{torch.cuda.device_count()} CUDA device(s) are available."
            )
        return torch.device(f"cuda:{gpu_device_id}")
    return torch.device("cpu")


def build_transforms(image_size: int) -> tuple[transforms.Compose, transforms.Compose]:
    train_transform = transforms.Compose(
        [
            PadToSquare(),
            transforms.RandomAffine(
                degrees=2.0,
                translate=(0.02, 0.02),
                scale=(0.95, 1.05),
                fill=255,
                interpolation=InterpolationMode.BILINEAR,
            ),
            transforms.ColorJitter(brightness=0.12, contrast=0.12),
            transforms.Resize(
                (image_size, image_size),
                interpolation=InterpolationMode.BICUBIC,
                antialias=True,
            ),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
    eval_transform = transforms.Compose(
        [
            PadToSquare(),
            transforms.Resize(
                (image_size, image_size),
                interpolation=InterpolationMode.BICUBIC,
                antialias=True,
            ),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
    return train_transform, eval_transform


def load_class_mapping(dataset_root: Path) -> dict[str, int]:
    labels_path = dataset_root / "labels.json"
    payload = json.loads(labels_path.read_text(encoding="utf-8"))
    class_to_index = payload.get("class_to_index")
    if not isinstance(class_to_index, dict) or not class_to_index:
        raise RuntimeError(f"Invalid class_to_index in {labels_path}")
    indexes = sorted(int(value) for value in class_to_index.values())
    if indexes != list(range(len(indexes))):
        raise RuntimeError("Class indexes must be contiguous and start at zero.")
    return {str(label): int(index) for label, index in class_to_index.items()}


def class_weights(dataset: ManifestDataset, class_count: int) -> torch.Tensor:
    counts = Counter(dataset.class_to_index[row["label"]] for row in dataset.rows)
    missing = [index for index in range(class_count) if counts.get(index, 0) == 0]
    if missing:
        inverse = {index: label for label, index in dataset.class_to_index.items()}
        missing_names = ", ".join(inverse[index] for index in missing)
        raise RuntimeError(f"Training split has no samples for: {missing_names}")
    total = len(dataset)
    return torch.tensor(
        [total / (class_count * counts[index]) for index in range(class_count)],
        dtype=torch.float32,
    )


def run_epoch(
    *,
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
    scaler: torch.amp.GradScaler | None,
    use_amp: bool,
    progress_description: str,
) -> tuple[float, list[int], list[int]]:
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    targets: list[int] = []
    predictions: list[int] = []

    progress = tqdm(
        loader,
        desc=progress_description,
        unit="batch",
        dynamic_ncols=True,
        leave=False,
        mininterval=0.5,
    )
    processed_samples = 0
    for images, labels in progress:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        if training:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(training):
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=use_amp,
            ):
                logits = model(images)
                loss = criterion(logits, labels)

            if training:
                assert scaler is not None
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

        total_loss += loss.item() * labels.size(0)
        processed_samples += labels.size(0)
        targets.extend(labels.detach().cpu().tolist())
        predictions.extend(logits.argmax(dim=1).detach().cpu().tolist())

        postfix: dict[str, str] = {
            "loss": f"{total_loss / processed_samples:.4f}",
        }
        if optimizer is not None:
            postfix["lr"] = f"{optimizer.param_groups[0]['lr']:.2e}"
        if device.type == "cuda":
            allocated_gb = torch.cuda.memory_allocated(device) / (1024**3)
            reserved_gb = torch.cuda.memory_reserved(device) / (1024**3)
            postfix["vram"] = f"{allocated_gb:.1f}/{reserved_gb:.1f}GB"
        progress.set_postfix(postfix, refresh=False)

    return total_loss / len(loader.dataset), targets, predictions


def metric_summary(
    targets: list[int], predictions: list[int], class_count: int
) -> dict[str, float]:
    labels = list(range(class_count))
    return {
        "accuracy": float(accuracy_score(targets, predictions)),
        "macro_f1": float(
            f1_score(
                targets,
                predictions,
                labels=labels,
                average="macro",
                zero_division=0,
            )
        ),
        "weighted_f1": float(
            f1_score(
                targets,
                predictions,
                labels=labels,
                average="weighted",
                zero_division=0,
            )
        ),
    }


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    args = parse_args()
    config = DocumentClassifierSettings()
    dataset_root = DocumentClassifierSettings.resolve_path(
        args.dataset_dir or config.dataset_export_dir
    )
    output_dir = DocumentClassifierSettings.resolve_path(args.output_dir or config.train_output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    set_seed(config.train_seed, config.train_deterministic)
    device = select_device(args.device or config.train_device, config.train_gpu_device_id)
    use_amp = config.train_mixed_precision and device.type == "cuda"
    class_to_index = load_class_mapping(dataset_root)
    index_to_class = {index: label for label, index in class_to_index.items()}
    train_transform, eval_transform = build_transforms(config.train_image_size)

    train_dataset = ManifestDataset(
        dataset_root=dataset_root,
        manifest_path=dataset_root / "train.csv",
        class_to_index=class_to_index,
        transform=train_transform,
    )
    val_dataset = ManifestDataset(
        dataset_root=dataset_root,
        manifest_path=dataset_root / "val.csv",
        class_to_index=class_to_index,
        transform=eval_transform,
    )
    test_dataset = ManifestDataset(
        dataset_root=dataset_root,
        manifest_path=dataset_root / "test.csv",
        class_to_index=class_to_index,
        transform=eval_transform,
    )

    loader_options = {
        "batch_size": config.train_batch_size,
        "num_workers": config.train_num_workers,
        "pin_memory": device.type == "cuda",
        "persistent_workers": config.train_num_workers > 0,
    }
    train_loader = DataLoader(train_dataset, shuffle=True, **loader_options)
    val_loader = DataLoader(val_dataset, shuffle=False, **loader_options)
    test_loader = DataLoader(test_dataset, shuffle=False, **loader_options)

    model = build_mobilenet_v3_small(
        num_classes=len(class_to_index),
        pretrained=config.train_pretrained,
        pretrained_weights_path=config.pretrained_weights_path,
    ).to(device)
    criterion = nn.CrossEntropyLoss(
        weight=class_weights(train_dataset, len(class_to_index)).to(device)
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.train_learning_rate,
        weight_decay=config.train_weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.train_epochs
    )
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    start_epoch = 1
    best_macro_f1 = -1.0
    history: list[dict[str, Any]] = []
    best_checkpoint_path = output_dir / "best_model.pt"
    if args.resume:
        resume_path = DocumentClassifierSettings.resolve_path(args.resume)
        checkpoint = torch.load(resume_path, map_location=device, weights_only=False)
        if checkpoint["class_to_index"] != class_to_index:
            raise RuntimeError("Resume checkpoint class mapping does not match the dataset.")
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        start_epoch = int(checkpoint["epoch"]) + 1
        best_macro_f1 = float(checkpoint.get("best_macro_f1", -1.0))
        history = list(checkpoint.get("history", []))

    print(f"Training on {device}; AMP={use_amp}; classes={len(class_to_index)}")
    stale_epochs = 0
    for epoch in range(start_epoch, config.train_epochs + 1):
        train_loss, train_targets, train_predictions = run_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            device=device,
            optimizer=optimizer,
            scaler=scaler,
            use_amp=use_amp,
            progress_description=f"Train {epoch:03d}/{config.train_epochs:03d}",
        )
        val_loss, val_targets, val_predictions = run_epoch(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device,
            optimizer=None,
            scaler=None,
            use_amp=use_amp,
            progress_description=f"Val   {epoch:03d}/{config.train_epochs:03d}",
        )
        train_metrics = metric_summary(
            train_targets, train_predictions, len(class_to_index)
        )
        val_metrics = metric_summary(val_targets, val_predictions, len(class_to_index))
        epoch_result = {
            "epoch": epoch,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "train_loss": train_loss,
            "val_loss": val_loss,
            "train": train_metrics,
            "val": val_metrics,
        }
        history.append(epoch_result)
        scheduler.step()
        write_json(output_dir / "history.json", history)
        print(
            f"Epoch {epoch:03d}: train_loss={train_loss:.4f} "
            f"val_loss={val_loss:.4f} val_macro_f1={val_metrics['macro_f1']:.4f}"
        )

        improved = val_metrics["macro_f1"] > best_macro_f1
        if improved:
            best_macro_f1 = val_metrics["macro_f1"]
            stale_epochs = 0
            torch.save(
                {
                    "format_version": 1,
                    "created_at": datetime.now(UTC).isoformat(),
                    "architecture": config.train_model_name,
                    "image_size": config.train_image_size,
                    "class_to_index": class_to_index,
                    "normalization": {"mean": IMAGENET_MEAN, "std": IMAGENET_STD},
                    "epoch": epoch,
                    "best_macro_f1": best_macro_f1,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "history": history,
                },
                best_checkpoint_path,
            )
        else:
            stale_epochs += 1
            if (
                config.train_early_stopping_patience > 0
                and stale_epochs >= config.train_early_stopping_patience
            ):
                print(f"Early stopping after {stale_epochs} epoch(s) without improvement.")
                break

    best_checkpoint = torch.load(best_checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(best_checkpoint["model_state_dict"])
    test_loss, test_targets, test_predictions = run_epoch(
        model=model,
        loader=test_loader,
        criterion=criterion,
        device=device,
        optimizer=None,
        scaler=None,
        use_amp=use_amp,
        progress_description="Test",
    )
    label_indexes = list(range(len(class_to_index)))
    target_names = [index_to_class[index] for index in label_indexes]
    test_payload = {
        "checkpoint": str(best_checkpoint_path),
        "best_epoch": best_checkpoint["epoch"],
        "test_loss": test_loss,
        "metrics": metric_summary(test_targets, test_predictions, len(class_to_index)),
        "classification_report": classification_report(
            test_targets,
            test_predictions,
            labels=label_indexes,
            target_names=target_names,
            output_dict=True,
            zero_division=0,
        ),
        "confusion_matrix": confusion_matrix(
            test_targets, test_predictions, labels=label_indexes
        ).tolist(),
        "labels": target_names,
    }
    write_json(output_dir / "test_metrics.json", test_payload)
    print(f"Best checkpoint: {best_checkpoint_path}")
    print(f"Test macro F1: {test_payload['metrics']['macro_f1']:.4f}")


if __name__ == "__main__":
    main()

