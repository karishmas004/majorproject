"""
Live webcam deepfake detection.

Design choice: the BROWSER captures the webcam via getUserMedia (this is
what "Access webcam" / "Display live camera" means in a web app — a
server cannot reach into a user's laptop camera). The frontend streams
JPEG frames to this WebSocket endpoint as base64, the backend runs the
same face-detect + classify + GradCAM-free fast-path pipeline on each
frame, and streams back label/confidence/fps/processing-time so the UI
can draw the green/red border overlay in real time.
"""
from __future__ import annotations

import time

import cv2
import torch
import torch.nn.functional as F
from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from app.core.config import settings
from app.services import model_registry
from app.services.face_detector import face_detector
from app.utils.preprocessing import base64_to_bgr, bgr_to_tensor

router = APIRouter(prefix="/api/webcam", tags=["Live Webcam Detection"])


@router.websocket("/stream")
async def webcam_stream(websocket: WebSocket):
    await websocket.accept()
    model = model_registry.image_model
    frame_times: list[float] = []

    try:
        while True:
            payload = await websocket.receive_json()
            frame_b64 = payload.get("frame")
            if not frame_b64:
                await websocket.send_json({"error": "invalid_frame", "detail": "No 'frame' field in payload"})
                continue

            t0 = time.perf_counter()
            try:
                bgr_image = base64_to_bgr(frame_b64)
            except ValueError as exc:
                await websocket.send_json({"error": "corrupted_frame", "detail": str(exc)})
                continue

            face_box = face_detector.largest_face(bgr_image)
            if face_box is None:
                await websocket.send_json(
                    {
                        "face_detected": False,
                        "label": None,
                        "confidence": None,
                        "fps": _rolling_fps(frame_times),
                        "processing_time_ms": round((time.perf_counter() - t0) * 1000, 2),
                    }
                )
                continue

            region = face_detector.crop_with_margin(bgr_image, face_box)
            if region.size == 0:
                continue

            input_tensor = bgr_to_tensor(region).to(settings.DEVICE)
            with torch.no_grad():
                logits = model(input_tensor)
                probs = F.softmax(logits, dim=1).squeeze(0).cpu().numpy()

            fake_prob = float(probs[1])
            real_prob = float(probs[0])
            label = "FAKE" if fake_prob >= settings.FAKE_THRESHOLD else "REAL"
            confidence = fake_prob if label == "FAKE" else real_prob

            elapsed = time.perf_counter() - t0
            frame_times.append(elapsed)

            await websocket.send_json(
                {
                    "face_detected": True,
                    "label": label,
                    "confidence": round(confidence, 4),
                    "fake_probability": round(fake_prob, 4),
                    "face_region": face_box.as_dict(),
                    "fps": _rolling_fps(frame_times),
                    "processing_time_ms": round(elapsed * 1000, 2),
                }
            )
    except WebSocketDisconnect:
        pass


def _rolling_fps(frame_times: list[float], window: int = 15) -> float:
    if not frame_times:
        return 0.0
    recent = frame_times[-window:]
    avg = sum(recent) / len(recent)
    return round(1.0 / avg, 1) if avg > 0 else 0.0
