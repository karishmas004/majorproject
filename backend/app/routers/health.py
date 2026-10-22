from fastapi import APIRouter

from app.schemas import HealthResponse
from app.services.model_registry import models_ready

router = APIRouter(prefix="/api", tags=["Health"])


@router.get("/health", response_model=HealthResponse, summary="Service + model status")
async def health():
    status = models_ready()
    return HealthResponse(status="ok", **status)
