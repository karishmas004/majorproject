"""Image deepfake detection pipeline: face detect -> preprocess -> infer -> GradCAM."""
from __future__ import annotations

import time

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from app.core.config import settings
from app.services import model_registry
from app.services.face_detector import face_detector
from app.services.gradcam import GradCAM, overlay_heatmap
from app.utils.exceptions import CorruptedFileError, NoFaceDetectedError
from app.utils.preprocessing import bgr_to_base64_jpeg, bgr_to_tensor


def decode_image(file_bytes: bytes, filename: str) -> np.ndarray:
    arr = np.frombuffer(file_bytes, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise CorruptedFileError(filename, "the file is not a valid/decodable image")
    return img


def analyze_image(file_bytes: bytes, filename: str) -> dict:
    start = time.perf_counter()

    bgr_image = decode_image(file_bytes, filename)
    face_box = face_detector.largest_face(bgr_image)

    if face_box is None:
        raise NoFaceDetectedError("the uploaded image")

    region_for_inference = face_detector.crop_with_margin(bgr_image, face_box)
    if region_for_inference.size == 0:
        raise NoFaceDetectedError("the uploaded image")
    face_detected = True

    input_tensor = bgr_to_tensor(region_for_inference).to(settings.DEVICE)
    input_tensor.requires_grad_(True)

    model = model_registry.image_model
    cam = GradCAM(model, model.get_target_layer())
    try:
        heatmap, pred_class = cam(input_tensor)
        with torch.no_grad():
            logits = model(input_tensor)
            probs = F.softmax(logits, dim=1).squeeze(0).cpu().numpy()
    finally:
        cam.close()

    real_prob, fake_prob = float(probs[0]), float(probs[1])
    label = "FAKE" if fake_prob >= settings.FAKE_THRESHOLD else "REAL"
    confidence = fake_prob if label == "FAKE" else real_prob

    # Build heatmap overlay on the region actually fed to the model.
    region_resized = cv2.resize(region_for_inference, (settings.IMAGE_INPUT_SIZE, settings.IMAGE_INPUT_SIZE))
    overlaid = overlay_heatmap(region_resized, heatmap)

    elapsed_ms = (time.perf_counter() - start) * 1000

    return {
        "label": label,
        "confidence": round(confidence, 4),
        "fake_probability": round(fake_prob, 4),
        "real_probability": round(real_prob, 4),
        "processing_time_ms": round(elapsed_ms, 2),
        "face_detected": face_detected,
        "face_region": face_box.as_dict() if face_box else None,
        "heatmap_base64": bgr_to_base64_jpeg(overlaid),
        "original_base64": bgr_to_base64_jpeg(bgr_image),
    }
