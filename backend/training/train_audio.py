"""
Train a CNN spectrogram classifier for audio deepfake/spoof detection.

Supported datasets:

1. ASVspoof 2019 LA:
   - Training: official ASVspoof training partition
   - Validation: official ASVspoof development partition
   - Evaluation: kept separate for final testing

2. Custom audio dataset:
   dataset_root/
       real/
           audio1.wav
       fake/
           audio2.wav

Example ASVspoof training:
python backend/training/train_audio.py \
    --asvspoof-protocol ~/LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.train.trn.txt \
    --asvspoof-audio-dir ~/LA/ASVspoof2019_LA_train/flac \
    --dev-protocol ~/LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.dev.trl.txt \
    --dev-audio-dir ~/LA/ASVspoof2019_LA_dev/flac \
    --epochs 5 \
    --batch-size 4 \
    --num-workers 0 \
    --output backend/weights/audio_cnn_spectrogram.pt

Labels:
    0 = bonafide / real
    1 = spoof / fake
"""

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

# Add backend directory to Python import path.
BACKEND_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND_DIR))

from app.core.config import settings  # noqa: E402
from app.models.audio_classifier import AudioSpectrogramClassifier  # noqa: E402
from training.datasets.audio_dataset import (  # noqa: E402
    ASVspoofDataset,
    SimpleAudioFolderDataset,
)


def collate_pad(batch):
    """Pad variable-length spectrograms along the time axis."""
    max_t = max(item[0].shape[-1] for item in batch)

    specs = []
    labels = []

    for spec, label in batch:
        pad_amount = max_t - spec.shape[-1]

        if pad_amount > 0:
            spec = np.pad(
                spec,
                ((0, 0), (0, 0), (0, pad_amount)),
                mode="constant",
            )

        specs.append(torch.as_tensor(spec, dtype=torch.float32))
        labels.append(label)

    return torch.stack(specs), torch.tensor(labels, dtype=torch.long)


def create_dataloader(dataset, batch_size, shuffle, num_workers):
    """Create a DataLoader suitable for macOS."""
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_pad,
        pin_memory=False,
    )


def train(args: argparse.Namespace) -> None:
    # Reproducibility.
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device(settings.DEVICE)
    print(f"\nTraining device: {device}")
    print(f"PyTorch version: {torch.__version__}")

    # ---------------------------------------------------------
    # 1. Load datasets
    # ---------------------------------------------------------
    if args.asvspoof_protocol:
        if not args.asvspoof_audio_dir:
            raise SystemExit(
                "--asvspoof-audio-dir is required with "
                "--asvspoof-protocol."
            )

        if not args.dev_protocol or not args.dev_audio_dir:
            raise SystemExit(
                "ASVspoof training requires both --dev-protocol "
                "and --dev-audio-dir so the official development "
                "partition can be used for validation."
            )

        train_ds = ASVspoofDataset(
            args.asvspoof_protocol,
            args.asvspoof_audio_dir,
        )

        val_ds = ASVspoofDataset(
            args.dev_protocol,
            args.dev_audio_dir,
        )

        print("\nDataset: ASVspoof 2019 LA")
        print("Training: official training partition")
        print("Validation: official development partition")

    elif args.data_root:
        dataset = SimpleAudioFolderDataset(args.data_root)

        if len(dataset) < 2:
            raise SystemExit(
                "At least two audio samples are required "
                "for training and validation."
            )

        val_size = max(1, int(len(dataset) * args.val_split))
        train_size = len(dataset) - val_size

        if train_size < 1:
            raise SystemExit(
                "Not enough samples. Reduce --val-split "
                "or add more audio files."
            )

        generator = torch.Generator().manual_seed(args.seed)

        train_ds, val_ds = random_split(
            dataset,
            [train_size, val_size],
            generator=generator,
        )

        print("\nDataset: custom real/fake audio folders")

    else:
        raise SystemExit(
            "Provide either --asvspoof-protocol with its audio "
            "and development paths, or --data-root."
        )

    if len(train_ds) == 0 or len(val_ds) == 0:
        raise SystemExit(
            "The training and validation datasets must both "
            "contain audio samples."
        )

    print(f"Training samples:   {len(train_ds)}")
    print(f"Validation samples: {len(val_ds)}")
    print(f"Batch size:         {args.batch_size}")
    print(f"DataLoader workers: {args.num_workers}")

    # ---------------------------------------------------------
    # 2. Create DataLoaders
    # ---------------------------------------------------------
    train_loader = create_dataloader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )

    val_loader = create_dataloader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    # ---------------------------------------------------------
    # 3. Initialize model
    # ---------------------------------------------------------
    model = AudioSpectrogramClassifier().to(device)

    criterion = nn.CrossEntropyLoss()

    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=1e-4,
    )

    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
    )

    output_path = (
        Path(settings.AUDIO_MODEL_PATH)
        if args.output is None
        else Path(args.output)
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)

    best_val_acc = -1.0

    # ---------------------------------------------------------
    # 4. Training loop
    # ---------------------------------------------------------
    for epoch in range(1, args.epochs + 1):
        model.train()

        start_time = time.time()
        running_loss = 0.0
        correct = 0
        total = 0

        for batch_index, (specs, labels) in enumerate(
            train_loader,
            start=1,
        ):
            specs = specs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad(set_to_none=True)

            outputs = model(specs)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            batch_count = labels.size(0)

            running_loss += loss.item() * batch_count
            correct += (
                outputs.argmax(dim=1) == labels
            ).sum().item()
            total += batch_count

            if (
                args.log_every > 0
                and batch_index % args.log_every == 0
            ):
                print(
                    f"  Batch {batch_index}/{len(train_loader)} "
                    f"| loss={loss.item():.4f}"
                )

        if total == 0:
            raise RuntimeError(
                "No training batches were processed."
            )

        train_loss = running_loss / total
        train_acc = correct / total

        # -----------------------------------------------------
        # 5. Validation on development partition
        # -----------------------------------------------------
        val_acc, val_loss = evaluate(
            model,
            val_loader,
            criterion,
            device,
        )

        scheduler.step()

        elapsed = time.time() - start_time

        print(
            f"\nEpoch {epoch}/{args.epochs} "
            f"| train_loss={train_loss:.4f} "
            f"| train_acc={train_acc:.4f} "
            f"| val_loss={val_loss:.4f} "
            f"| val_acc={val_acc:.4f} "
            f"| time={elapsed:.1f}s"
        )

        # Save the checkpoint with the best validation accuracy.
        if val_acc > best_val_acc:
            best_val_acc = val_acc

            torch.save(
                model.state_dict(),
                output_path,
            )

            print(
                f"Saved best checkpoint: {output_path} "
                f"(validation accuracy={val_acc:.4f})"
            )

    print("\nTraining complete.")
    print(f"Best validation accuracy: {best_val_acc:.4f}")
    print(f"Saved model: {output_path}")


@torch.no_grad()
def evaluate(
    model,
    loader,
    criterion,
    device,
) -> tuple[float, float]:
    """Evaluate the model and return accuracy and average loss."""
    model.eval()

    running_loss = 0.0
    correct = 0
    total = 0

    for specs, labels in loader:
        specs = specs.to(device)
        labels = labels.to(device)

        outputs = model(specs)
        loss = criterion(outputs, labels)

        batch_count = labels.size(0)

        running_loss += loss.item() * batch_count
        correct += (
            outputs.argmax(dim=1) == labels
        ).sum().item()
        total += batch_count

    if total == 0:
        raise RuntimeError(
            "Validation dataset produced no batches."
        )

    return correct / total, running_loss / total


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train an audio deepfake/spoof classifier."
    )

    # ASVspoof training partition.
    parser.add_argument(
        "--asvspoof-protocol",
        default=None,
        help="Path to the ASVspoof training protocol.",
    )
    parser.add_argument(
        "--asvspoof-audio-dir",
        default=None,
        help="Directory containing training FLAC files.",
    )

    # Official ASVspoof development partition.
    parser.add_argument(
        "--dev-protocol",
        default=None,
        help="Path to the ASVspoof development protocol.",
    )
    parser.add_argument(
        "--dev-audio-dir",
        default=None,
        help="Directory containing development FLAC files.",
    )

    # Custom real/fake dataset.
    parser.add_argument(
        "--data-root",
        default=None,
        help="Folder containing real/ and fake/ subdirectories.",
    )

    # Training configuration.
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--val-split", type=float, default=0.15)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--log-every",
        type=int,
        default=100,
        help="Print training progress every N batches; use 0 to disable.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Path for the saved model checkpoint.",
    )

    train(parser.parse_args())