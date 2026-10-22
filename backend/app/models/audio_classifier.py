"""
CNN spectrogram classifier for audio deepfake (voice cloning / TTS /
voice-conversion) detection. Operates on log-mel spectrograms rather
than raw waveform, which is lightweight, fast on CPU, and a
well-established approach for spoof detection (used in ASVspoof
baseline systems).
"""
from __future__ import annotations

import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class AudioSpectrogramClassifier(nn.Module):
    """Binary (real=0, fake=1) classifier over log-mel spectrograms."""

    def __init__(self, n_mels: int = 128):
        super().__init__()
        self.n_mels = n_mels
        self.conv1 = ConvBlock(1, 32)
        self.conv2 = ConvBlock(32, 64)
        self.conv3 = ConvBlock(64, 128)
        self.conv4 = ConvBlock(128, 256)

        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(128, 2),
        )
        self.target_layer_name = "conv4"

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 1, n_mels, T)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.global_pool(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)

    def get_target_layer(self) -> nn.Module:
        return self.conv4


def load_audio_classifier(weights_path, device: str) -> AudioSpectrogramClassifier:
    model = AudioSpectrogramClassifier()
    if weights_path.exists():
        state_dict = torch.load(weights_path, map_location=device)
        model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model
