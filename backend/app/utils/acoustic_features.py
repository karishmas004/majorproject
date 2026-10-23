"""Extract acoustic features for a complementary audio spoof detector."""

from __future__ import annotations

import librosa
import numpy as np

from app.core.config import settings
from app.utils.audio_features import decode_audio_ffmpeg


SAMPLE_RATE = 16000


def extract_acoustic_features(
    audio_bytes: bytes,
    sample_rate: int = SAMPLE_RATE,
) -> dict[str, float]:
    """Extract stable, interpretable acoustic measurements from audio."""

    try:
        waveform = decode_audio_ffmpeg(audio_bytes)
        sr = settings.AUDIO_SAMPLE_RATE

        if sr != sample_rate:
            waveform = librosa.resample(
                waveform,
                orig_sr=sr,
                target_sr=sample_rate,
            )
            sr = sample_rate

        if waveform.size == 0:
            raise ValueError("Audio is empty.")

        waveform = np.nan_to_num(waveform)

        if len(waveform) < int(0.25 * sr):
            raise ValueError("Audio must contain at least 0.25 seconds.")

        waveform = waveform.astype(np.float32)

        # Amplitude and energy dynamics
        rms = librosa.feature.rms(
            y=waveform,
            frame_length=1024,
            hop_length=256,
        )[0]

        rms_mean = float(np.mean(rms))
        rms_std = float(np.std(rms))

        # Pitch: ignore unvoiced frames and invalid estimates.
        f0 = librosa.yin(
            waveform,
            fmin=65,
            fmax=500,
            sr=sr,
            frame_length=1024,
            hop_length=256,
        )

        f0 = f0[np.isfinite(f0) & (f0 >= 65) & (f0 <= 500)]

        if f0.size:
            pitch_mean = float(np.mean(f0))
            pitch_std = float(np.std(f0))
            pitch_min = float(np.min(f0))
            pitch_max = float(np.max(f0))
            pitch_cv = pitch_std / max(pitch_mean, 1e-8)
        else:
            pitch_mean = 0.0
            pitch_std = 0.0
            pitch_min = 0.0
            pitch_max = 0.0
            pitch_cv = 0.0

        # Spectral characteristics
        centroid = librosa.feature.spectral_centroid(
            y=waveform,
            sr=sr,
            n_fft=1024,
            hop_length=256,
        )[0]

        bandwidth = librosa.feature.spectral_bandwidth(
            y=waveform,
            sr=sr,
            n_fft=1024,
            hop_length=256,
        )[0]

        flatness = librosa.feature.spectral_flatness(
            y=waveform,
            n_fft=1024,
            hop_length=256,
        )[0]

        zcr = librosa.feature.zero_crossing_rate(
            y=waveform,
            frame_length=1024,
            hop_length=256,
        )[0]

        # Energy dynamics and approximate silence proportion
        silence_threshold = max(rms_mean * 0.1, 1e-6)

        features = {
            "duration_sec": len(waveform) / sr,
            "rms_mean": rms_mean,
            "rms_std": rms_std,
            "rms_cv": rms_std / max(rms_mean, 1e-8),
            "pitch_mean": pitch_mean,
            "pitch_std": pitch_std,
            "pitch_min": pitch_min,
            "pitch_max": pitch_max,
            "pitch_cv": pitch_cv,
            "pitch_valid_fraction": (
                float(f0.size / max(len(waveform) / 256, 1))
            ),
            "spectral_centroid_mean": float(np.mean(centroid)),
            "spectral_centroid_std": float(np.std(centroid)),
            "spectral_bandwidth_mean": float(np.mean(bandwidth)),
            "spectral_bandwidth_std": float(np.std(bandwidth)),
            "spectral_flatness_mean": float(np.mean(flatness)),
            "spectral_flatness_std": float(np.std(flatness)),
            "zero_crossing_mean": float(np.mean(zcr)),
            "zero_crossing_std": float(np.std(zcr)),
            "low_energy_fraction": float(np.mean(rms < silence_threshold)),
        }

        return {
            key: float(value)
            for key, value in features.items()
            if np.isfinite(value)
        }

    except (ValueError, RuntimeError) as exc:
        raise ValueError(f"Could not extract acoustic features: {exc}") from exc
