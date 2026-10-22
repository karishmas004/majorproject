# Architecture Notes

See the main [README.md](../README.md) for the full setup guide. This file
covers a few implementation details worth knowing before you extend the
system.

## Why Haar cascades instead of a deep face detector?

`cv2.CascadeClassifier` ships inside the `opencv-python` wheel itself, so
face detection works with zero extra downloads and zero external network
calls. It's less accurate than something like RetinaFace/MTCNN on
hard poses, but for a face-forward image/video dataset (which is what
FaceForensics++/CelebDF/DFDC are) it's a reliable, fast, dependency-free
choice. If you want a stronger detector later, swap the implementation in
`app/services/face_detector.py` — every caller only depends on
`FaceDetector.largest_face()` / `.detect()` / `.crop_with_margin()`, so
nothing else needs to change.

## Why the same architecture for image and video?

Both the image and video pipelines classify individual face crops — video
just does it many times per clip and aggregates. Sharing
`DeepfakeFrameClassifier` means:
- One codebase to maintain and improve
- You can optionally pretrain on a large image-based dataset, then
  fine-tune the video checkpoint separately (see `train_video.py`) so it
  specializes in video-compression/temporal artifacts without starting
  from scratch.

## Why the webcam pipeline is browser-driven

A backend process cannot access a user's local camera hardware — only the
browser can, via `getUserMedia`. So "live webcam detection" here means:
the browser captures and displays the feed, streams JPEG frames to the
backend over a WebSocket (`/api/webcam/stream`), and the backend runs the
same classifier used for images on each frame, streaming back
label/confidence/FPS for the frontend to overlay.

## Grad-CAM target layers

Both `DeepfakeFrameClassifier` and `AudioSpectrogramClassifier` expose
`get_target_layer()`, pointing at their final convolutional block. This
keeps `app/services/gradcam.py` architecture-agnostic — it registers
forward/backward hooks on whatever module is returned, so swapping either
backbone later doesn't require touching the Grad-CAM code.
