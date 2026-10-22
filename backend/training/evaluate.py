"""
Standalone evaluation: computes accuracy, precision, recall, F1, and
confusion matrix for a trained image/video model on a held-out
real/fake folder, or an audio model on a held-out ASVspoof/custom set.

Usage:
    python evaluate.py --task image --data-root /data/ffpp_test --weights ../weights/image_efficientnetv2.pt
    python evaluate.py --task audio --data-root /data/asvspoof_test --weights ../weights/audio_cnn_spectrogram.pt
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader

sys.path.append(str(Path(__file__).resolve().parents[1]))
from app.core.config import settings  # noqa: E402
from app.models.audio_classifier import load_audio_classifier  # noqa: E402
from app.models.frame_classifier import load_frame_classifier  # noqa: E402
from training.datasets.audio_dataset import SimpleAudioFolderDataset  # noqa: E402
from training.datasets.face_dataset import FaceFrameDataset  # noqa: E402
from training.train_audio import collate_pad  # noqa: E402


@torch.no_grad()
def evaluate_image_or_video(data_root: str, weights: str, batch_size: int = 32):
    device = settings.DEVICE
    dataset = FaceFrameDataset(data_root)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    model = load_frame_classifier(Path(weights),device,pretrained_backbone=False,)

    y_true, y_pred = [], []
    for images, labels in loader:
        images = images.to(device)
        outputs = model(images)
        preds = outputs.argmax(1).cpu().numpy()
        y_pred.extend(preds.tolist())
        y_true.extend(labels.numpy().tolist())

    print(classification_report(y_true, y_pred, target_names=["REAL", "FAKE"]))
    print("Confusion matrix (rows=true, cols=pred):")
    print(confusion_matrix(y_true, y_pred))


@torch.no_grad()
def evaluate_audio(data_root: str, weights: str, batch_size: int = 16):
    device = settings.DEVICE
    dataset = SimpleAudioFolderDataset(data_root)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_pad)
    model = load_audio_classifier(Path(weights), device)

    y_true, y_pred = [], []
    for specs, labels in loader:
        specs = specs.to(device)
        outputs = model(specs)
        preds = outputs.argmax(1).cpu().numpy()
        y_pred.extend(preds.tolist())
        y_true.extend(labels.numpy().tolist())

    print(classification_report(y_true, y_pred, target_names=["REAL", "FAKE"]))
    print("Confusion matrix (rows=true, cols=pred):")
    print(confusion_matrix(y_true, y_pred))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=["image", "video", "audio"], required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    if args.task in ("image", "video"):
        evaluate_image_or_video(args.data_root, args.weights, args.batch_size)
    else:
        evaluate_audio(args.data_root, args.weights, args.batch_size)
