"""Shared preprocessing helpers for image/video/audio pipelines."""
from __future__ import annotations

import base64

import cv2
import numpy as np
import torch
from torchvision import transforms

from app.core.config import settings

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

image_transform = transforms.Compose(
    [
        transforms.ToPILImage(),
        transforms.Resize((settings.IMAGE_INPUT_SIZE, settings.IMAGE_INPUT_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ]
)


def bgr_to_tensor(bgr_image: np.ndarray) -> torch.Tensor:
    """BGR np.ndarray (OpenCV) -> normalized (1, 3, H, W) tensor."""
    rgb = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
    tensor = image_transform(rgb)
    return tensor.unsqueeze(0)


def bgr_to_base64_jpeg(bgr_image: np.ndarray, quality: int = 90) -> str:
    ok, buf = cv2.imencode(".jpg", bgr_image, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ok:
        raise ValueError("Failed to encode image to JPEG")
    return base64.b64encode(buf).decode("utf-8")


def base64_to_bgr(b64_str: str) -> np.ndarray:
    """Decode a base64 (data-URL or raw) JPEG/PNG string into a BGR image."""
    if "," in b64_str and b64_str.strip().startswith("data:"):
        b64_str = b64_str.split(",", 1)[1]
    raw = base64.b64decode(b64_str)
    arr = np.frombuffer(raw, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Could not decode base64 image data")
    return img
