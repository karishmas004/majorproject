"""Custom exceptions and their FastAPI handlers, so every failure mode
described in the spec (unsupported files, corrupted files, missing
faces, camera unavailable, missing weights, invalid uploads, low
memory) returns a clean JSON error instead of a raw 500 traceback."""
from __future__ import annotations

import logging

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

logger = logging.getLogger("deepfake_api.errors")


class UnsupportedFileTypeError(Exception):
    def __init__(self, filename: str, allowed: set[str]):
        self.filename = filename
        self.allowed = allowed
        super().__init__(f"Unsupported file type for '{filename}'. Allowed: {sorted(allowed)}")


class FileTooLargeError(Exception):
    def __init__(self, filename: str, size_mb: float, limit_mb: int):
        self.filename = filename
        self.size_mb = size_mb
        self.limit_mb = limit_mb
        super().__init__(f"'{filename}' is {size_mb:.1f}MB, exceeds the {limit_mb}MB limit")


class CorruptedFileError(Exception):
    def __init__(self, filename: str, reason: str):
        self.filename = filename
        self.reason = reason
        super().__init__(f"Could not process '{filename}': {reason}")


class NoFaceDetectedError(Exception):
    def __init__(self, context: str = "the uploaded media"):
        self.context = context
        super().__init__(f"No face could be detected in {context}")


class ModelWeightsMissingError(Exception):
    def __init__(self, model_name: str, expected_path: str):
        self.model_name = model_name
        self.expected_path = expected_path
        super().__init__(
            f"Fine-tuned weights for {model_name} not found at {expected_path}. "
            f"The model will still run using its ImageNet-initialized backbone, "
            f"but predictions are not meaningful until you train it — see README."
        )


class CameraUnavailableError(Exception):
    def __init__(self, reason: str):
        self.reason = reason
        super().__init__(f"Webcam unavailable: {reason}")


def register_exception_handlers(app: FastAPI) -> None:
    @app.exception_handler(UnsupportedFileTypeError)
    async def unsupported_type_handler(request: Request, exc: UnsupportedFileTypeError):
        logger.warning(str(exc))
        return JSONResponse(status_code=415, content={"error": "unsupported_file_type", "detail": str(exc)})

    @app.exception_handler(FileTooLargeError)
    async def too_large_handler(request: Request, exc: FileTooLargeError):
        logger.warning(str(exc))
        return JSONResponse(status_code=413, content={"error": "file_too_large", "detail": str(exc)})

    @app.exception_handler(CorruptedFileError)
    async def corrupted_handler(request: Request, exc: CorruptedFileError):
        logger.warning(str(exc))
        return JSONResponse(status_code=422, content={"error": "corrupted_file", "detail": str(exc)})

    @app.exception_handler(NoFaceDetectedError)
    async def no_face_handler(request: Request, exc: NoFaceDetectedError):
        logger.info(str(exc))
        return JSONResponse(status_code=422, content={"error": "no_face_detected", "detail": str(exc)})

    @app.exception_handler(ModelWeightsMissingError)
    async def weights_missing_handler(request: Request, exc: ModelWeightsMissingError):
        logger.warning(str(exc))
        return JSONResponse(status_code=200, content={"warning": "model_not_finetuned", "detail": str(exc)})

    @app.exception_handler(CameraUnavailableError)
    async def camera_unavailable_handler(request: Request, exc: CameraUnavailableError):
        logger.warning(str(exc))
        return JSONResponse(status_code=503, content={"error": "camera_unavailable", "detail": str(exc)})

    @app.exception_handler(MemoryError)
    async def memory_error_handler(request: Request, exc: MemoryError):
        logger.error("MemoryError: %s", exc)
        return JSONResponse(
            status_code=507,
            content={"error": "low_memory", "detail": "Server ran out of memory processing this request. Try a smaller file."},
        )

    @app.exception_handler(Exception)
    async def unhandled_exception_handler(request: Request, exc: Exception):
        logger.exception("Unhandled exception on %s", request.url)
        return JSONResponse(
            status_code=500,
            content={"error": "internal_server_error", "detail": "An unexpected error occurred. Check server logs."},
        )
