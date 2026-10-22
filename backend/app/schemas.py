"""Pydantic models describing API request/response shapes."""
from __future__ import annotations

from pydantic import BaseModel, Field


class FaceRegion(BaseModel):
    x: int
    y: int
    width: int
    height: int


class ImageDetectionResponse(BaseModel):
    label: str = Field(..., description="'REAL' or 'FAKE'")
    confidence: float = Field(..., description="Confidence of the predicted label, 0-1")
    fake_probability: float = Field(..., description="Raw probability the image is fake, 0-1")
    real_probability: float
    processing_time_ms: float
    face_detected: bool
    face_region: FaceRegion | None = None
    heatmap_base64: str | None = Field(None, description="Grad-CAM overlay, JPEG base64")
    original_base64: str = Field(..., description="Original uploaded image re-encoded, JPEG base64")


class FrameScore(BaseModel):
    frame_index: int
    timestamp_sec: float
    fake_probability: float
    face_detected: bool


class VideoDetectionResponse(BaseModel):
    label: str
    confidence: float
    fake_probability: float
    real_probability: float
    frames_analyzed: int
    frames_with_face: int
    processing_time_ms: float
    timeline: list[FrameScore]
    most_suspicious_frame_base64: str | None = None
    most_suspicious_heatmap_base64: str | None = None


class AudioDetectionResponse(BaseModel):
    label: str
    confidence: float
    fake_probability: float
    real_probability: float
    processing_time_ms: float
    duration_sec: float
    waveform_base64: str = Field(..., description="Waveform plot, PNG base64")
    spectrogram_base64: str = Field(..., description="Log-mel spectrogram plot, PNG base64")

    cnn_label: str
    cnn_fake_probability: float
    acoustic_label: str
    acoustic_fake_probability: float
    hybrid_prediction: bool

    # Individual detector outputs
    cnn_label: str | None = None
    cnn_fake_probability: float | None = None
    acoustic_label: str | None = None
    acoustic_fake_probability: float | None = None
    hybrid_prediction: bool = False


class ErrorResponse(BaseModel):
    error: str
    detail: str


class HealthResponse(BaseModel):
    status: str
    device: str
    image_model_finetuned: bool
    video_model_finetuned: bool
    audio_model_finetuned: bool
