from fastapi import APIRouter, File, UploadFile

from app.core.config import settings
from app.schemas import VideoDetectionResponse
from app.services.video_service import analyze_video
from app.utils.validation import validate_upload

router = APIRouter(prefix="/api/video", tags=["Video Detection"])


@router.post(
    "/detect",
    response_model=VideoDetectionResponse,
    summary="Detect whether an uploaded video contains deepfake content",
)
async def detect_video(file: UploadFile = File(..., description="mp4, avi, mov, or mkv")):
    """
    Upload a video. Frames are sampled, faces detected, each face
    classified, and the results aggregated via temporal voting into a
    single verdict, along with a per-frame timeline and a heatmap for
    the most suspicious frame.
    """
    file_bytes = await file.read()
    validate_upload(file, file_bytes, settings.ALLOWED_VIDEO_EXT, settings.MAX_VIDEO_MB)
    result = analyze_video(file_bytes, file.filename)
    return VideoDetectionResponse(**result)
