"""
Central configuration for the Multimodal Deepfake Detection System.

All tunable paths, thresholds, and device settings live here so that
services/routers never hardcode values.
"""
from __future__ import annotations

import os
from pathlib import Path

import torch


class Settings:
    """Application-wide settings, loaded once at import time."""

    # ---- Paths -----------------------------------------------------
    BASE_DIR: Path = Path(__file__).resolve().parents[2]  # backend/
    WEIGHTS_DIR: Path = BASE_DIR / "weights"
    ASSETS_DIR: Path = BASE_DIR / "assets"
    TMP_DIR: Path = BASE_DIR / "tmp"

    IMAGE_MODEL_PATH: Path = WEIGHTS_DIR / "image_efficientnetv2_final.pt"
    VIDEO_MODEL_PATH: Path = WEIGHTS_DIR / "video_efficientnetv2.pt"
    AUDIO_MODEL_PATH: Path = WEIGHTS_DIR / "audio_cnn_spectrogram.pt"
    FACE_CASCADE_PATH: str = "haarcascade_frontalface_default.xml"  # bundled with opencv-python

    # ---- Device ------------------------------------------------------
    DEVICE: str = (

    "cuda" if torch.cuda.is_available()

    else "mps" if torch.backends.mps.is_available()

    else "cpu"

)

    # ---- Inference tuning ---------------------------------------------
    IMAGE_INPUT_SIZE: int = 224
    VIDEO_MAX_FRAMES: int = 32          # frames sampled per video for analysis
    VIDEO_FRAME_STRIDE: int = 5          # sample every Nth frame after face detection
    AUDIO_SAMPLE_RATE: int = 16000
    AUDIO_DURATION_SEC: float = 4.0      # fixed-length window fed to the model
    FAKE_THRESHOLD: float = 0.5          # probability above this => classified FAKE

    # ---- Upload limits --------------------------------------------------
    MAX_IMAGE_MB: int = 15
    MAX_VIDEO_MB: int = 200
    MAX_AUDIO_MB: int = 30

    ALLOWED_IMAGE_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    ALLOWED_VIDEO_EXT = {".mp4", ".avi", ".mov", ".mkv"}
    ALLOWED_AUDIO_EXT = {
        ".wav", ".mp3", ".flac", ".m4a", ".opus",
        ".aac", ".ogg", ".wma", ".aiff", ".aif",
        ".amr", ".caf", ".webm", ".mp4", ".mp2", ".au",
    }

    # ---- CORS ------------------------------------------------------------
    CORS_ORIGINS = os.getenv("CORS_ORIGINS", "http://localhost:5173").split(",")

    def __init__(self) -> None:
        self.WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
        self.TMP_DIR.mkdir(parents=True, exist_ok=True)
        self.ASSETS_DIR.mkdir(parents=True, exist_ok=True)


settings = Settings()
