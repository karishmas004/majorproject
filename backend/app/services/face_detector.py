from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass
class FaceBox:
    x: int
    y: int
    w: int
    h: int
    score: float = 0.0

    def as_dict(self):
        return {
            "x": self.x,
            "y": self.y,
            "width": self.w,
            "height": self.h,
        }


class FaceDetector:
    """
    YuNet-based face detector.

    Supports images of different dimensions by detecting at multiple
    resolutions and mapping the detected boxes to original coordinates.
    """

    def __init__(self):
        model_path = (
            Path(__file__).resolve().parents[2]
            / "assets"
            / "face_detection_yunet_2023mar.onnx"
        )

        if not model_path.is_file():
            raise FileNotFoundError(
                f"YuNet model not found: {model_path}"
            )

        self.model_path = str(model_path)
        self.score_threshold = 0.70
        self.nms_threshold = 0.30
        self.top_k = 5000

        # Verify that OpenCV can load the ONNX model.
        try:
            self._detector = cv2.FaceDetectorYN.create(
                self.model_path,
                "",
                (320, 320),
                self.score_threshold,
                self.nms_threshold,
                self.top_k,
            )
        except cv2.error as exc:
            raise RuntimeError(
                f"Failed to initialize YuNet face detector: {exc}"
            ) from exc

    @staticmethod
    def _valid_image(image):
        return (
            isinstance(image, np.ndarray)
            and image.ndim == 3
            and image.shape[2] == 3
            and image.shape[0] > 0
            and image.shape[1] > 0
        )

    @staticmethod
    def _iou(a, b):
        ax1, ay1 = a.x, a.y
        ax2, ay2 = a.x + a.w, a.y + a.h

        bx1, by1 = b.x, b.y
        bx2, by2 = b.x + b.w, b.y + b.h

        ix1 = max(ax1, bx1)
        iy1 = max(ay1, by1)
        ix2 = min(ax2, bx2)
        iy2 = min(ay2, by2)

        intersection = max(0, ix2 - ix1) * max(0, iy2 - iy1)
        area_a = a.w * a.h
        area_b = b.w * b.h
        union = area_a + area_b - intersection

        return intersection / union if union > 0 else 0.0

    def detect(self, bgr_image):
        """
        Detect faces at multiple image scales.

        Returns boxes in the coordinate system of the original image.
        No arbitrary center-crop or Haar fallback is used.
        """
        if not self._valid_image(bgr_image):
            return []

        original_h, original_w = bgr_image.shape[:2]

        # Normalize extremely large images for efficient detection.
        # Multiple scales help when faces occupy a small part of an image.
        max_side = max(original_h, original_w)
        base_scale = min(1.0, 1600.0 / max_side)

        base_w = max(1, int(round(original_w * base_scale)))
        base_h = max(1, int(round(original_h * base_scale)))

        base_image = cv2.resize(
            bgr_image,
            (base_w, base_h),
            interpolation=(
                cv2.INTER_AREA
                if base_scale < 1.0
                else cv2.INTER_LINEAR
            ),
        )

        candidates = []

        # Detect at different scales to improve detection of small faces.
        for extra_scale in (1.0, 0.75, 0.50):
            det_w = max(1, int(round(base_w * extra_scale)))
            det_h = max(1, int(round(base_h * extra_scale)))

            if (det_w, det_h) == (base_w, base_h):
                detection_image = base_image
            else:
                detection_image = cv2.resize(
                    base_image,
                    (det_w, det_h),
                    interpolation=cv2.INTER_AREA,
                )

            try:
                self._detector.setInputSize((det_w, det_h))
                _, results = self._detector.detect(detection_image)
            except cv2.error:
                continue

            if results is None:
                continue

            scale_x = original_w / det_w
            scale_y = original_h / det_h

            for row in results:
                x, y, w, h = map(float, row[:4])
                score = float(row[14])

                if score < self.score_threshold or w <= 0 or h <= 0:
                    continue

                x1 = max(0, int(round(x * scale_x)))
                y1 = max(0, int(round(y * scale_y)))
                x2 = min(
                    original_w,
                    int(round((x + w) * scale_x)),
                )
                y2 = min(
                    original_h,
                    int(round((y + h) * scale_y)),
                )

                if x2 <= x1 or y2 <= y1:
                    continue

                candidates.append(
                    FaceBox(
                        x=x1,
                        y=y1,
                        w=x2 - x1,
                        h=y2 - y1,
                        score=score,
                    )
                )

        # Remove duplicate detections from the multiple scales.
        candidates.sort(
            key=lambda face: (
                face.score,
                face.w * face.h,
            ),
            reverse=True,
        )

        final_faces = []

        for candidate in candidates:
            if all(
                self._iou(candidate, accepted) < 0.45
                for accepted in final_faces
            ):
                final_faces.append(candidate)

        return final_faces

    def largest_face(self, bgr_image):
        """
        Select the largest confidently detected face.

        Returns None when no face is detected.
        """
        faces = self.detect(bgr_image)

        if not faces:
            return None

        return max(
            faces,
            key=lambda face: face.w * face.h,
        )

    @staticmethod
    def crop_with_margin(bgr_image, box, margin=0.65):
        """
        Create a square crop around the detected face.

        The crop includes surrounding context, stays within the image
        where possible, and uses edge padding when necessary.
        """
        if box is None or not FaceDetector._valid_image(bgr_image):
            return np.empty((0, 0, 3), dtype=np.uint8)

        image_h, image_w = bgr_image.shape[:2]

        face_size = max(box.w, box.h)
        if face_size <= 0:
            return np.empty((0, 0, 3), dtype=np.uint8)

        # Expand around the detected face.
        crop_size = int(round(face_size * (1.0 + 2.0 * margin)))
        crop_size = max(crop_size, face_size)

        center_x = box.x + box.w // 2
        center_y = box.y + box.h // 2

        x1 = center_x - crop_size // 2
        y1 = center_y - crop_size // 2
        x2 = x1 + crop_size
        y2 = y1 + crop_size

        # Intersect the crop with the available image.
        src_x1 = max(0, x1)
        src_y1 = max(0, y1)
        src_x2 = min(image_w, x2)
        src_y2 = min(image_h, y2)

        crop = bgr_image[src_y1:src_y2, src_x1:src_x2]

        if crop.size == 0:
            return np.empty((0, 0, 3), dtype=np.uint8)

        # Pad missing regions at image boundaries.
        top = src_y1 - y1
        left = src_x1 - x1
        bottom = y2 - src_y2
        right = x2 - src_x2

        if top or bottom or left or right:
            crop = cv2.copyMakeBorder(
                crop,
                top,
                bottom,
                left,
                right,
                cv2.BORDER_REPLICATE,
            )

        # Defensive correction to guarantee a square crop.
        if crop.shape[0] != crop.shape[1]:
            crop = cv2.resize(
                crop,
                (crop_size, crop_size),
                interpolation=cv2.INTER_AREA,
            )

        return crop


face_detector = FaceDetector()