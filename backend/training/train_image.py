from __future__ import annotations

import argparse
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import classification_report, confusion_matrix

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.core.config import settings
from app.models.frame_classifier import DeepfakeFrameClassifier
from training.datasets.face_dataset import FaceFrameDataset


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    total = 0
    all_labels = []
    all_preds = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * labels.size(0)
            total += labels.size(0)

            all_labels.extend(labels.cpu().numpy().tolist())
            all_preds.extend(outputs.argmax(dim=1).cpu().numpy().tolist())

    accuracy = (
        sum(a == b for a, b in zip(all_labels, all_preds)) / total
        if total else 0.0
    )

    return accuracy, running_loss / max(total, 1), all_labels, all_preds


def train(args):
    set_seed(args.seed)

    device = settings.DEVICE
    print(f"Device: {device}")
    print(f"Input size: {settings.IMAGE_INPUT_SIZE}x{settings.IMAGE_INPUT_SIZE}")
    print("Initializing EfficientNetV2-S from random weights.")

    dataset = FaceFrameDataset(args.data_root)

    labels = [label for _, label in dataset.samples]
    class_counts = np.bincount(labels, minlength=2)

    if min(class_counts) == 0:
        raise ValueError("Both real and fake classes must contain images.")

    print(f"Total images: {len(dataset)}")
    print(f"Real images: {class_counts[0]}")
    print(f"Fake images: {class_counts[1]}")

    val_size = max(1, int(len(dataset) * args.val_split))
    train_size = len(dataset) - val_size

    generator = torch.Generator().manual_seed(args.seed)
    train_ds, val_ds = random_split(
        dataset,
        [train_size, val_size],
        generator=generator,
    )

    print(f"Training images: {train_size}")
    print(f"Validation images: {val_size}")

    workers = args.num_workers
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=workers,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=workers,
    )

    model = DeepfakeFrameClassifier(
        pretrained_backbone=False,
        freeze_backbone=False,
    ).to(device)

    # Inverse-frequency weights reduce the effect of class imbalance.
    train_labels = [labels[i] for i in train_ds.indices]
    train_counts = np.bincount(train_labels, minlength=2)
    class_weights = len(train_labels) / (
        2.0 * torch.tensor(train_counts, dtype=torch.float32)
    )
    criterion = nn.CrossEntropyLoss(
        weight=class_weights.to(device)
    )

    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=1e-4,
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
    )

    output_path = Path(args.output).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    best_val_loss = float("inf")

    for epoch in range(1, args.epochs + 1):
        model.train()
        start = time.time()
        total_loss = 0.0
        total_correct = 0
        total_seen = 0

        for batch_index, (images, labels_batch) in enumerate(
            train_loader, start=1
        ):
            images = images.to(device)
            labels_batch = labels_batch.to(device)

            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = criterion(logits, labels_batch)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * labels_batch.size(0)
            total_correct += (
                logits.argmax(dim=1) == labels_batch
            ).sum().item()
            total_seen += labels_batch.size(0)

            if batch_index % 100 == 0:
                print(
                    f"Epoch {epoch}/{args.epochs} "
                    f"batch {batch_index}/{len(train_loader)} "
                    f"loss={loss.item():.4f}"
                )

        scheduler.step()

        train_loss = total_loss / max(total_seen, 1)
        train_acc = total_correct / max(total_seen, 1)

        val_acc, val_loss, y_true, y_pred = evaluate(
            model, val_loader, criterion, device
        )

        print(
            f"\nEpoch {epoch}/{args.epochs} | "
            f"train_loss={train_loss:.4f} "
            f"train_acc={train_acc:.4f} | "
            f"val_loss={val_loss:.4f} "
            f"val_acc={val_acc:.4f} | "
            f"time={time.time() - start:.1f}s"
        )

        # Save by validation loss, rather than accuracy alone.
        if np.isfinite(val_loss) and val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), output_path)
            print(f"Saved best checkpoint: {output_path}")

        print("\nValidation classification report:")
        print(
            classification_report(
                y_true,
                y_pred,
                labels=[0, 1],
                target_names=["REAL", "FAKE"],
                zero_division=0,
                digits=4,
            )
        )
        print("Confusion matrix (rows=actual, columns=predicted):")
        print(confusion_matrix(y_true, y_pred, labels=[0, 1]))
        print()

    print(f"Training finished. Best validation loss: {best_val_loss:.4f}")
    print(f"Checkpoint: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-root",
        default="dataset_512",
        help="Folder containing real/ and fake/ subfolders",
    )
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--val-split", type=float, default=0.15)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output",
        default="weights/image_efficientnetv2_scratch_v2.pt",
    )
    train(parser.parse_args())
