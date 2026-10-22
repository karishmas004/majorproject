"""
Loads every model exactly once at application startup and exposes them
as module-level singletons. Routers/services import from here instead
of constructing models per-request (which would be slow and would
re-download/re-read weights on every call).
"""
from __future__ import annotations

import logging
import joblib

from app.core.config import settings
from app.models.audio_classifier import load_audio_classifier
from app.models.frame_classifier import load_frame_classifier

logger = logging.getLogger("deepfake_api.model_registry")

image_model = None
video_model = None
audio_model = None
audio_acoustic_model = None


def load_all_models() -> None:
    """Called once from the FastAPI startup event."""
    global image_model, video_model, audio_model, audio_acoustic_model

    logger.info("Loading image model on %s ...", settings.DEVICE)
    image_model = load_frame_classifier(
    settings.IMAGE_MODEL_PATH,
    settings.DEVICE,
    pretrained_backbone=False,
)
    if not settings.IMAGE_MODEL_PATH.exists():
        logger.warning(
            "No fine-tuned weights found at %s — using an ImageNet-initialized "
            "backbone with an untrained head. Predictions will not be meaningful "
            "until you run training/train_image.py. See README.",
            settings.IMAGE_MODEL_PATH,
        )

    # Video reuses the same architecture as image (frame-level classifier).
    logger.info("Loading video frame model on %s ...", settings.DEVICE)
    video_model = load_frame_classifier(settings.VIDEO_MODEL_PATH, settings.DEVICE)
    if not settings.VIDEO_MODEL_PATH.exists():
        logger.warning(
            "No fine-tuned weights found at %s — see README to train.",
            settings.VIDEO_MODEL_PATH,
        )

    logger.info("Loading audio model on %s ...", settings.DEVICE)
    audio_model = load_audio_classifier(settings.AUDIO_MODEL_PATH, settings.DEVICE)
    if not settings.AUDIO_MODEL_PATH.exists():
        logger.warning(
            "No fine-tuned weights found at %s — see README to train.",
            settings.AUDIO_MODEL_PATH,
        )

    acoustic_path = settings.AUDIO_MODEL_PATH.parent / "audio_acoustic_rf.joblib"
    audio_acoustic_model = None

    if acoustic_path.exists():
        logger.info("Loading acoustic audio model from %s ...", acoustic_path)
        audio_acoustic_model = joblib.load(acoustic_path)
        logger.info("Acoustic audio model loaded.")
    else:
        logger.warning("Acoustic model not found at %s", acoustic_path)

    logger.info("All models loaded.")


def models_ready() -> dict:
    return {
        "image_model_finetuned": settings.IMAGE_MODEL_PATH.exists(),
        "video_model_finetuned": settings.VIDEO_MODEL_PATH.exists(),
        "audio_model_finetuned": settings.AUDIO_MODEL_PATH.exists(),
        "device": settings.DEVICE,
    }
