from fastapi import APIRouter, File, UploadFile

from app.core.config import settings
from app.schemas import ImageDetectionResponse
from app.services.image_service import analyze_image
from app.utils.validation import validate_upload

router = APIRouter(prefix="/api/image", tags=["Image Detection"])


@router.post(
    "/detect",
    response_model=ImageDetectionResponse,
    summary="Detect whether an uploaded image is a deepfake",
)
async def detect_image(file: UploadFile = File(..., description="jpg, jpeg, png, bmp, or webp")):
    """
    Upload a single image. Returns REAL/FAKE label, confidence,
    per-class probabilities, the detected face region, a Grad-CAM
    explainability heatmap, and processing time.
    """
    file_bytes = await file.read()
    validate_upload(file, file_bytes, settings.ALLOWED_IMAGE_EXT, settings.MAX_IMAGE_MB)
    result = analyze_image(file_bytes, file.filename)
    return ImageDetectionResponse(**result)
