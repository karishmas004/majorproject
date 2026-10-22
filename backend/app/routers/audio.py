from fastapi import APIRouter, File, UploadFile

from app.core.config import settings
from app.schemas import AudioDetectionResponse
from app.services.audio_service import analyze_audio
from app.utils.validation import validate_upload

router = APIRouter(prefix="/api/audio", tags=["Audio Detection"])


@router.post(
    "/detect",
    response_model=AudioDetectionResponse,
    summary="Detect whether an uploaded audio clip is a voice deepfake",
)
async def detect_audio(file: UploadFile = File(..., description="wav, mp3, flac, or m4a")):
    """
    Upload an audio clip. Returns REAL/FAKE label, confidence, a
    waveform plot, and a log-mel spectrogram plot.
    """
    file_bytes = await file.read()
    validate_upload(file, file_bytes, settings.ALLOWED_AUDIO_EXT, settings.MAX_AUDIO_MB)
    result = analyze_audio(file_bytes, file.filename)
    return AudioDetectionResponse(**result)
