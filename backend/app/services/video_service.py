"""
Video deepfake detection pipeline.

1. Decode video with OpenCV VideoCapture (ffmpeg backend under the hood).
2. Sample up to VIDEO_MAX_FRAMES frames, spaced by VIDEO_FRAME_STRIDE.
3. Detect + crop the face in each sampled frame.
4. Run the frame classifier on every face crop.
5. Aggregate with temporal voting (mean fake-probability + majority vote)
   into a single video-level verdict, plus a per-frame timeline and a
   GradCAM heatmap for the single most suspicious frame.
"""
from __future__ import annotations

import os
import tempfile
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


def _sample_frame_indices(total_frames: int) -> list[int]:
    if total_frames <= 0:
        return []
    step = max(settings.VIDEO_FRAME_STRIDE, total_frames // settings.VIDEO_MAX_FRAMES or 1)
    indices = list(range(0, total_frames, step))[: settings.VIDEO_MAX_FRAMES]
    return indices


def analyze_video(file_bytes: bytes, filename: str) -> dict:
    start = time.perf_counter()
    suffix = os.path.splitext(filename)[1] or ".mp4"

    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(file_bytes)
        tmp_path = tmp.name

    try:
        cap = cv2.VideoCapture(tmp_path)
        if not cap.isOpened():
            raise CorruptedFileError(filename, "the video container/codec could not be opened by OpenCV/FFmpeg")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        indices_to_sample = set(_sample_frame_indices(total_frames))

        model = model_registry.image_model
        timeline: list[dict] = []
        fake_probs: list[float] = []
        frames_with_face = 0

        best_fake_prob = -1.0
        best_frame_bgr = None
        best_face_crop_resized = None

        frame_idx = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if frame_idx not in indices_to_sample:
                frame_idx += 1
                continue

            face_box = face_detector.largest_face(frame)
            face_detected = face_box is not None

            if not face_detected:
                frame_idx += 1
                continue

            region = face_detector.crop_with_margin(frame, face_box)

            if region.size == 0:
                frame_idx += 1
                continue

            frames_with_face += 1
            if region.size == 0:
                frame_idx += 1
                continue

            

            input_tensor = bgr_to_tensor(region).to(settings.DEVICE)
            with torch.no_grad():
                logits = model(input_tensor)
                probs = F.softmax(logits, dim=1).squeeze(0).cpu().numpy()
            fake_prob = float(probs[1])
            fake_probs.append(fake_prob)

            timeline.append(
                {
                    "frame_index": frame_idx,
                    "timestamp_sec": round(frame_idx / fps, 2),
                    "fake_probability": round(fake_prob, 4),
                    "face_detected": face_detected,
                }
            )

            if fake_prob > best_fake_prob:
                best_fake_prob = fake_prob
                best_frame_bgr = frame.copy()
                best_face_crop_resized = cv2.resize(
                    region, (settings.IMAGE_INPUT_SIZE, settings.IMAGE_INPUT_SIZE)
                )

            frame_idx += 1

        cap.release()

        if not fake_probs:
            raise NoFaceDetectedError("any analyzed frame of the uploaded video")

        # --- Temporal aggregation: mean probability + majority vote -------
        mean_fake_prob = float(np.mean(fake_probs))
        label = "FAKE" if mean_fake_prob >= settings.FAKE_THRESHOLD else "REAL"
        confidence = mean_fake_prob if label == "FAKE" else 1 - mean_fake_prob

        # --- GradCAM on the single most suspicious frame -------------------
        heatmap_b64 = None
        frame_b64 = None
        if best_face_crop_resized is not None:
            input_tensor = bgr_to_tensor(best_face_crop_resized).to(settings.DEVICE)
            input_tensor.requires_grad_(True)
            cam = GradCAM(model, model.get_target_layer())
            try:
                heatmap, _ = cam(input_tensor, class_idx=1)  # explain the FAKE class
            finally:
                cam.close()
            overlaid = overlay_heatmap(best_face_crop_resized, heatmap)
            heatmap_b64 = bgr_to_base64_jpeg(overlaid)
            frame_b64 = bgr_to_base64_jpeg(best_frame_bgr)

        elapsed_ms = (time.perf_counter() - start) * 1000

        return {
            "label": label,
            "confidence": round(confidence, 4),
            "fake_probability": round(mean_fake_prob, 4),
            "real_probability": round(1 - mean_fake_prob, 4),
            "frames_analyzed": len(fake_probs),
            "frames_with_face": frames_with_face,
            "processing_time_ms": round(elapsed_ms, 2),
            "timeline": timeline,
            "most_suspicious_frame_base64": frame_b64,
            "most_suspicious_heatmap_base64": heatmap_b64,
        }
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
