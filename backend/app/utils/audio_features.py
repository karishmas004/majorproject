"""Audio loading and feature extraction (waveform + log-mel spectrogram)."""
from __future__ import annotations

import io
import subprocess

import librosa
import numpy as np
import torch

from app.core.config import settings


def decode_audio_ffmpeg(file_bytes: bytes) -> np.ndarray:
    """
    Decode supported audio formats in memory using FFmpeg.
    No intermediate audio files are created.
    Returns mono float32 audio at the configured sample rate.
    """
    if not file_bytes:
        raise ValueError("Uploaded audio file is empty.")

    command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel", "error",
        "-nostdin",
        "-i", "pipe:0",
        "-map", "0:a:0",
        "-vn",
        "-ac", "1",
        "-ar", str(settings.AUDIO_SAMPLE_RATE),
        "-f", "f32le",
        "-acodec", "pcm_f32le",
        "pipe:1",
    ]

    try:
        result = subprocess.run(
            command,
            input=file_bytes,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=60,
            check=False,
        )
    except FileNotFoundError as exc:
        raise RuntimeError(
            "FFmpeg was not found. Install it and ensure it is on the backend PATH."
        ) from exc
    except subprocess.TimeoutExpired as exc:
        raise ValueError("Audio decoding timed out.") from exc

    if result.returncode != 0:
        details = result.stderr.decode("utf-8", errors="replace").strip()
        raise ValueError(f"FFmpeg could not decode this audio: {details[-1000:]}")

    waveform = np.frombuffer(result.stdout, dtype="<f4").copy()

    if waveform.size == 0:
        raise ValueError("FFmpeg produced no audio samples.")

    waveform = np.nan_to_num(waveform, copy=False).astype(np.float32, copy=False)
    return waveform


def load_audio_any_format(file_bytes: bytes) -> np.ndarray:
    """
    Decode audio formats supported by FFmpeg into a mono waveform
    at the configured sample rate, entirely in memory.
    """
    return decode_audio_ffmpeg(file_bytes)


def fix_length(waveform: np.ndarray, sr: int, duration_sec: float) -> np.ndarray:
    """Pad with silence or center-crop the waveform to a fixed duration."""
    target_len = int(sr * duration_sec)
    if len(waveform) >= target_len:
        start = (len(waveform) - target_len) // 2
        return waveform[start : start + target_len]
    pad_total = target_len - len(waveform)
    pad_left = pad_total // 2
    pad_right = pad_total - pad_left
    return np.pad(waveform, (pad_left, pad_right), mode="constant")


def waveform_to_log_mel(waveform: np.ndarray, sr: int, n_mels: int = 128) -> np.ndarray:
    mel = librosa.feature.melspectrogram(y=waveform, sr=sr, n_mels=n_mels, n_fft=1024, hop_length=256)
    log_mel = librosa.power_to_db(mel, ref=np.max)
    # normalize to [0, 1] for stable CNN training/inference
    log_mel = (log_mel - log_mel.min()) / (log_mel.max() - log_mel.min() + 1e-8)
    return log_mel.astype(np.float32)


def log_mel_to_tensor(log_mel: np.ndarray) -> torch.Tensor:
    """(n_mels, T) -> (1, 1, n_mels, T) tensor ready for the CNN."""
    tensor = torch.from_numpy(log_mel).unsqueeze(0).unsqueeze(0)
    return tensor


def audio_pipeline(file_bytes: bytes) -> tuple[np.ndarray, np.ndarray, torch.Tensor]:
    """
    Full pipeline: bytes -> (raw waveform for plotting, fixed-length
    waveform, model-ready spectrogram tensor).
    """
    waveform = load_audio_any_format(file_bytes)
    fixed = fix_length(waveform, settings.AUDIO_SAMPLE_RATE, settings.AUDIO_DURATION_SEC)
    log_mel = waveform_to_log_mel(fixed, settings.AUDIO_SAMPLE_RATE)
    tensor = log_mel_to_tensor(log_mel)
    return waveform, log_mel, tensor
