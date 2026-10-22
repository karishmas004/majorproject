"""
Multimodal Deepfake Detection System — FastAPI entrypoint.

Run with:
    uvicorn app.main:app --reload
"""
from __future__ import annotations

import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.core.config import settings
from app.routers import audio, health, image, video, webcam
from app.services.model_registry import load_all_models
from app.utils.exceptions import register_exception_handlers

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("deepfake_api")

app = FastAPI(
    title="Multimodal Deepfake Detection System",
    description=(
        "Detects deepfakes across images, video, audio, and a live webcam feed "
        "using EfficientNetV2-based frame classifiers and a CNN spectrogram "
        "classifier, with Grad-CAM explainability."
    ),
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

register_exception_handlers(app)

app.include_router(health.router)
app.include_router(image.router)
app.include_router(video.router)
app.include_router(audio.router)
app.include_router(webcam.router)


@app.on_event("startup")
async def on_startup():
    logger.info("Starting Multimodal Deepfake Detection API on device=%s", settings.DEVICE)
    load_all_models()


@app.get("/", tags=["Root"])
async def root():
    return {
        "service": "Multimodal Deepfake Detection System",
        "docs": "/docs",
        "health": "/api/health",
    }
