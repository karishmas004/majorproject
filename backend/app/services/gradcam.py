"""
Grad-CAM (Gradient-weighted Class Activation Mapping) implementation.

Works with any nn.Module that exposes get_target_layer() -> nn.Module,
which both DeepfakeFrameClassifier and AudioSpectrogramClassifier do.
Returns a heatmap the same spatial size as the input, normalized 0-1,
which callers overlay on top of the original image/spectrogram.
"""
from __future__ import annotations

import cv2
import numpy as np
import torch
import torch.nn.functional as F


class GradCAM:
    def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module):
        self.model = model
        self.target_layer = target_layer
        self._activations: torch.Tensor | None = None
        self._gradients: torch.Tensor | None = None

        self._fwd_handle = target_layer.register_forward_hook(self._save_activation)
        self._bwd_handle = target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, inp, out):
        self._activations = out.detach()

    def _save_gradient(self, module, grad_in, grad_out):
        self._gradients = grad_out[0].detach()

    def __call__(self, input_tensor: torch.Tensor, class_idx: int | None = None) -> tuple[np.ndarray, int]:
        """
        input_tensor: (1, C, H, W), already normalized/on the right device.
        Returns (heatmap in [0,1] as HxW float32 array, predicted class idx).
        """
        self.model.zero_grad(set_to_none=True)
        output = self.model(input_tensor)  # (1, num_classes)
        probs = F.softmax(output, dim=1)

        if class_idx is None:
            class_idx = int(torch.argmax(probs, dim=1).item())

        score = output[0, class_idx]
        score.backward(retain_graph=True)

        gradients = self._gradients  # (1, C, h, w)
        activations = self._activations  # (1, C, h, w)
        weights = gradients.mean(dim=(2, 3), keepdim=True)  # global-average-pool gradients

        cam = torch.relu((weights * activations).sum(dim=1)).squeeze(0)  # (h, w)
        cam = cam.cpu().numpy()

        if cam.max() > 0:
            cam = cam / cam.max()

        target_h, target_w = input_tensor.shape[2], input_tensor.shape[3]
        cam_resized = cv2.resize(cam, (target_w, target_h))
        return cam_resized.astype(np.float32), class_idx

    def close(self):
        self._fwd_handle.remove()
        self._bwd_handle.remove()


def overlay_heatmap(bgr_image: np.ndarray, heatmap: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    """Overlay a normalized [0,1] heatmap onto a BGR image, returns BGR uint8."""
    heatmap_uint8 = np.uint8(255 * np.clip(heatmap, 0, 1))
    heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    if heatmap_color.shape[:2] != bgr_image.shape[:2]:
        heatmap_color = cv2.resize(heatmap_color, (bgr_image.shape[1], bgr_image.shape[0]))
    overlaid = cv2.addWeighted(heatmap_color, alpha, bgr_image, 1 - alpha, 0)
    return overlaid
