"""
ASVspoof-style audio dataset loader.

Expects the standard ASVspoof protocol file format:
    <speaker_id> <file_id> - <system_id> <bonafide/spoof>

and audio files named `<file_id>.flac` (or .wav) under `audio_dir`.

For a custom dataset, you can instead lay files out as:
    dataset_root/real/*.wav
    dataset_root/fake/*.wav
and use `SimpleAudioFolderDataset` below.
"""
from __future__ import annotations

from pathlib import Path

from torch.utils.data import Dataset

from app.utils.audio_features import fix_length, load_audio_any_format, waveform_to_log_mel
from app.core.config import settings


class ASVspoofDataset(Dataset):
    """label 0 = bonafide/real, label 1 = spoof/fake."""

    def __init__(self, protocol_path: str | Path, audio_dir: str | Path):
        self.audio_dir = Path(audio_dir)
        self.samples: list[tuple[str, int]] = []

        with open(protocol_path, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 5:
                    continue
                file_id = parts[1]
                key = parts[-1].lower()
                label = 0 if key == "bonafide" else 1
                self.samples.append((file_id, label))

        if not self.samples:
            raise RuntimeError(f"No entries parsed from protocol file {protocol_path}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        file_id, label = self.samples[idx]
        for ext in (".flac", ".wav"):
            candidate = self.audio_dir / f"{file_id}{ext}"
            if candidate.exists():
                audio_path = candidate
                break
        else:
            raise FileNotFoundError(f"No audio file found for id {file_id} in {self.audio_dir}")

        waveform = load_audio_any_format(audio_path.read_bytes())
        fixed = fix_length(waveform, settings.AUDIO_SAMPLE_RATE, settings.AUDIO_DURATION_SEC)
        log_mel = waveform_to_log_mel(fixed, settings.AUDIO_SAMPLE_RATE)
        return log_mel[None, :, :], label  # add channel dim -> (1, n_mels, T)


class SimpleAudioFolderDataset(Dataset):
    """label 0 = REAL, label 1 = FAKE, folder layout: root/{real,fake}/*.wav|mp3|flac|m4a"""

    AUDIO_EXTENSIONS = {".wav", ".mp3", ".flac", ".m4a"}

    def __init__(self, dataset_root: str | Path):
        self.root = Path(dataset_root)
        self.samples: list[tuple[Path, int]] = []
        for path in sorted((self.root / "real").rglob("*")):
            if path.suffix.lower() in self.AUDIO_EXTENSIONS:
                self.samples.append((path, 0))
        for path in sorted((self.root / "fake").rglob("*")):
            if path.suffix.lower() in self.AUDIO_EXTENSIONS:
                self.samples.append((path, 1))
        if not self.samples:
            raise RuntimeError(f"No audio files found under {self.root}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        path, label = self.samples[idx]
        waveform = load_audio_any_format(path.read_bytes())
        fixed = fix_length(waveform, settings.AUDIO_SAMPLE_RATE, settings.AUDIO_DURATION_SEC)
        log_mel = waveform_to_log_mel(fixed, settings.AUDIO_SAMPLE_RATE)
        return log_mel[None, :, :], label
