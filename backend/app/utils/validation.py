"""Shared upload validation: extension whitelist + size limit checks."""
from __future__ import annotations

import os

from fastapi import UploadFile

from app.utils.exceptions import FileTooLargeError, UnsupportedFileTypeError


def validate_upload(file: UploadFile, file_bytes: bytes, allowed_ext: set[str], max_mb: int) -> None:
    if file.filename is None:
        raise UnsupportedFileTypeError("<unnamed file>", allowed_ext)

    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in allowed_ext:
        raise UnsupportedFileTypeError(file.filename, allowed_ext)

    size_mb = len(file_bytes) / (1024 * 1024)
    if size_mb > max_mb:
        raise FileTooLargeError(file.filename, size_mb, max_mb)

    if size_mb == 0:
        raise UnsupportedFileTypeError(file.filename, allowed_ext)
