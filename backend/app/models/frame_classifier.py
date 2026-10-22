"""
EfficientNetV2-based binary classifier (Real vs Fake) used for both the
image pipeline and, frame-by-frame, the video pipeline.

Architecture: torchvision efficientnet_v2_s backbone + a small custom
classification head. The backbone can be initialized with ImageNet
weights (downloaded by torchvision the first time you run this code
on a machine with internet access) and then fine-tuned on
FaceForensics++ / CelebDF / DFDC via training/train_image.py.
"""
from __future__ import annotations

import torch
import torch.nn as nn
from torchvision.models import efficientnet_v2_s, EfficientNet_V2_S_Weights


class DeepfakeFrameClassifier(nn.Module):
    """Binary (real=0, fake=1) classifier built on EfficientNetV2-S."""

    def __init__(self, pretrained_backbone: bool = True, freeze_backbone: bool = False):
        super().__init__()
        weights = EfficientNet_V2_S_Weights.IMAGENET1K_V1 if pretrained_backbone else None
        backbone = efficientnet_v2_s(weights=weights)

        # Keep the feature extractor, replace the classifier head.
        self.features = backbone.features
        self.avgpool = backbone.avgpool
        in_features = backbone.classifier[1].in_features

        self.classifier = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(in_features, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),
            nn.Linear(256, 2),  # [real_logit, fake_logit]
        )

        if freeze_backbone:
            for param in self.features.parameters():
                param.requires_grad = False

        # Kept so GradCAM can hook the final conv block by name.
        self.target_layer_name = "features.7"

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)

    def get_target_layer(self) -> nn.Module:
        """Last convolutional stage, used as the GradCAM target layer."""
        return self.features[-1]


def load_frame_classifier(weights_path, device: str, pretrained_backbone: bool = True) -> DeepfakeFrameClassifier:
    """
    Instantiate the model and load fine-tuned weights if they exist.
    Falls back to an ImageNet-initialized backbone with a randomly
    initialized head if no fine-tuned checkpoint is present yet, so the
    API stays usable end-to-end before you've trained on deepfake data.
    """
    model = DeepfakeFrameClassifier(pretrained_backbone=pretrained_backbone)
    if weights_path.exists():
        state_dict = torch.load(weights_path, map_location=device)
        model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model
