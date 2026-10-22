"""
Fine-tune the video frame classifier.

The video pipeline reuses the exact same EfficientNetV2 architecture as
the image pipeline (app.models.frame_classifier.DeepfakeFrameClassifier)
but is trained on frames extracted specifically from videos (see
training/datasets/extract_faces_from_videos.py), which captures
video-specific compression artifacts and temporal-manipulation traces
that a still-image dataset won't.

Usage:
    python training/datasets/extract_faces_from_videos.py \\
        --real-dir /data/ffpp/original_sequences \\
        --fake-dir /data/ffpp/manipulated_sequences/Deepfakes \\
        --output-root /data/ffpp_video_frames --frames-per-video 20

    python train_video.py --data-root /data/ffpp_video_frames --epochs 10
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))
from app.core.config import settings  # noqa: E402
from training.train_image import train as train_image_model  # noqa: E402


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the video (frame-level) deepfake classifier")
    parser.add_argument("--data-root", required=True, help="Folder with real/ and fake/ video-frame subfolders")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--val-split", type=float, default=0.15)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--freeze-backbone", action="store_true")
    parser.add_argument("--output", default=str(settings.VIDEO_MODEL_PATH))
    args = parser.parse_args()
    train_image_model(args)
