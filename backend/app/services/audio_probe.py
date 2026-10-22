"""wav2vec2 embedding + trained classifier for voice deepfake detection."""
from __future__ import annotations

from functools import lru_cache

import joblib
import librosa
import numpy as np
import torch
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2Model

from app.core.config import settings

SSL_NAME = "facebook/wav2vec2-base"
LAYERS = (3, 6, 9, 12)          # middle layers keep the most artifact information
SR = 16000
WIN = 4 * SR
MAX_WINDOWS = 3
PROBE_PATH = settings.WEIGHTS_DIR / "audio_probe.joblib"


@lru_cache(maxsize=1)
def _ssl():
    fe = Wav2Vec2FeatureExtractor.from_pretrained(SSL_NAME)
    model = Wav2Vec2Model.from_pretrained(SSL_NAME).to(settings.DEVICE).eval()
    return fe, model


def load_windows(waveform):
    """Trim silence (a known ASVspoof shortcut), peak-normalize, cut into 4 s windows."""
    w = np.asarray(waveform, dtype=np.float32)
    trimmed, _ = librosa.effects.trim(w, top_db=30)
    if len(trimmed) >= SR:
        w = trimmed
    if len(w) == 0:
        w = np.zeros(WIN, dtype=np.float32)
    peak = float(np.abs(w).max())
    if peak > 0:
        w = w / peak * 0.9
    if len(w) < WIN:
        w = np.pad(w, (0, WIN - len(w)), mode="wrap")
    n = max(1, min(MAX_WINDOWS, len(w) // WIN))
    starts = np.linspace(0, len(w) - WIN, n).astype(int)
    return [w[s:s + WIN] for s in starts]


@torch.no_grad()
def embed_clip(waveform):
    fe, model = _ssl()
    inputs = fe(load_windows(waveform), sampling_rate=SR, return_tensors="pt")
    out = model(inputs.input_values.to(settings.DEVICE), output_hidden_states=True)
    feats = torch.cat([out.hidden_states[i].mean(dim=1) for i in LAYERS], dim=1)
    return feats.mean(dim=0).cpu().numpy()


@lru_cache(maxsize=1)
def _probe():
    if not PROBE_PATH.exists():
        raise FileNotFoundError(f"{PROBE_PATH} not found - run training/auto_audio.py first")
    return joblib.load(PROBE_PATH)


def predict_fake_prob(waveform) -> float:
    return float(_probe().predict_proba(embed_clip(waveform)[None, :])[0, 1])
