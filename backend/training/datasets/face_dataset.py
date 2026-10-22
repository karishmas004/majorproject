"""
Generic face-frame dataset loader that works for FaceForensics++, CelebDF,
DFDC (after frame extraction), and any custom dataset, as long as the
data is laid out as:

    dataset_root/
        real/
            img_0001.jpg
            ...
        fake/
            img_0001.jpg
            ...

For FaceForensics++: point `real/` at the `original_sequences` extracted
frames and `fake/` at any/all of `manipulated_sequences/{Deepfakes,
Face2Face,FaceSwap,NeuralTextures}` extracted frames.

For CelebDF: point `real/` at Celeb-real + YouTube-real frames and
`fake/` at Celeb-synthesis frames.

For DFDC: the raw dataset ships as videos + a metadata.json labeling
each video REAL/FAKE. Run `extract_dfdc_frames.py` (below) first to
convert it into the real/fake folder convention, then use this same
loader.

Use utils/extract_faces_from_videos.py (also in this folder) to turn
raw FaceForensics++/CelebDF videos into cropped face frames before
training, since training directly on full video frames wastes capacity
on background pixels.
"""
from __future__ import annotations

from pathlib import Path

import cv2
from torch.utils.data import Dataset

from app.utils.preprocessing import image_transform


class FaceFrameDataset(Dataset):
    """
    label 0 = REAL, label 1 = FAKE — matches DeepfakeFrameClassifier's
    output ordering ([real_logit, fake_logit]).
    """

    IMG_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

    def __init__(self, dataset_root: str | Path, split_transform=None):
        self.root = Path(dataset_root)
        self.transform = split_transform or image_transform
        self.samples: list[tuple[Path, int]] = []

        real_dir = self.root / "real"
        fake_dir = self.root / "fake"
        if not real_dir.exists() or not fake_dir.exists():
            raise FileNotFoundError(
                f"Expected '{real_dir}' and '{fake_dir}' to exist. "
                f"See module docstring for the required layout."
            )

        for path in sorted(real_dir.rglob("*")):
            if path.suffix.lower() in self.IMG_EXTENSIONS:
                self.samples.append((path, 0))
        for path in sorted(fake_dir.rglob("*")):
            if path.suffix.lower() in self.IMG_EXTENSIONS:
                self.samples.append((path, 1))

        if not self.samples:
            raise RuntimeError(f"No images found under {self.root}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        path, label = self.samples[idx]
        bgr = cv2.imread(str(path))
        if bgr is None:
            raise RuntimeError(f"Failed to read image {path}")
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        tensor = self.transform(rgb)
        return tensor, label
