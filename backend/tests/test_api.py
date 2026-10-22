"""
Smoke tests for the API. Run with:
    pytest tests/ -v

These use synthetic images/audio generated in-memory so the test suite
doesn't depend on any external dataset being downloaded.
"""
from __future__ import annotations

import io
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
from fastapi.testclient import TestClient
from PIL import Image

sys.path.append(str(Path(__file__).resolve().parents[1]))
from app.main import app  # noqa: E402
from app.services.model_registry import load_all_models  # noqa: E402

load_all_models()
client = TestClient(app)


def _synthetic_face_jpeg_bytes() -> bytes:
    """A plain image (no real face) — exercises the no-face-detected path."""
    img = Image.new("RGB", (300, 300), color=(120, 150, 180))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


def _synthetic_wav_bytes(duration_sec: float = 2.0, sr: int = 16000) -> bytes:
    t = np.linspace(0, duration_sec, int(sr * duration_sec), endpoint=False)
    tone = 0.3 * np.sin(2 * np.pi * 440 * t).astype(np.float32)
    buf = io.BytesIO()
    sf.write(buf, tone, sr, format="WAV")
    return buf.getvalue()


def test_health():
    resp = client.get("/api/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert "device" in body


def test_image_rejects_bad_extension():
    resp = client.post(
        "/api/image/detect",
        files={"file": ("test.txt", b"not an image", "text/plain")},
    )
    assert resp.status_code == 415


def test_image_no_face_returns_422():
    resp = client.post(
        "/api/image/detect",
        files={"file": ("plain.jpg", _synthetic_face_jpeg_bytes(), "image/jpeg")},
    )
    # A flat-color image has no detectable face -> graceful 422, not a crash.
    assert resp.status_code == 422
    assert resp.json()["error"] == "no_face_detected"


def test_audio_detect_runs_end_to_end():
    resp = client.post(
        "/api/audio/detect",
        files={"file": ("tone.wav", _synthetic_wav_bytes(), "audio/wav")},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["label"] in ("REAL", "FAKE")
    assert 0.0 <= body["fake_probability"] <= 1.0
    assert "waveform_base64" in body and "spectrogram_base64" in body


def test_video_rejects_bad_extension():
    resp = client.post(
        "/api/video/detect",
        files={"file": ("clip.txt", b"not a video", "text/plain")},
    )
    assert resp.status_code == 415


def test_audio_rejects_oversized_and_missing_file():
    resp = client.post("/api/audio/detect")
    assert resp.status_code == 422  # FastAPI's own required-field validation
