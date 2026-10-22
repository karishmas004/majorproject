# Multimodal Deepfake Detection System

A full-stack deepfake detection platform covering **images, video, audio,
and live webcam feeds**, built with FastAPI + PyTorch on the backend and
React + TypeScript + Tailwind on the frontend. Includes Grad-CAM
explainability, a per-frame video timeline, waveform/spectrogram
visualization for audio, and complete training/evaluation pipelines for
FaceForensics++, CelebDF, DFDC, and ASVspoof.

---

## 1. Project Overview

| Modality | Model | What it does |
|---|---|---|
| Image | EfficientNetV2-S + custom head | Face detect → classify → Grad-CAM heatmap |
| Video | Same EfficientNetV2-S architecture, frame-level | Sample frames → per-frame classify → temporal-vote aggregate → timeline |
| Audio | CNN over log-mel spectrograms | Decode → spectrogram → classify (spoof/voice-clone detection) |
| Live Webcam | Same image model, streamed | Browser captures camera → streams frames over WebSocket → real-time overlay |

**Important — read before you train:** this repository ships the model
*architectures* and *training pipelines*, initialized with ImageNet
weights via `torchvision`. It does **not** ship pretrained
deepfake-specific weights, because:

1. FaceForensics++/CelebDF/DFDC/ASVspoof are license-gated datasets — their
   weights can't be legally redistributed in a public repo.
2. There is no legitimate pip-installable "pretrained deepfake detector."

Out of the box, the API runs end-to-end and returns predictions immediately
— but those predictions are **not meaningful** until you fine-tune each
model on real data using the scripts in `backend/training/` (see §8).
The `/api/health` endpoint tells you which models are still un-fine-tuned.

---

## 2. Architecture

```
                     ┌─────────────────────┐
                     │   React Frontend     │
                     │  (Vite + TS + Tailwind)│
                     └──────────┬───────────┘
                                │ REST + WebSocket
                     ┌──────────▼───────────┐
                     │     FastAPI Backend   │
                     │  routers/ (image,     │
                     │  video, audio, webcam)│
                     └──────────┬───────────┘
                                │
          ┌─────────────────────┼─────────────────────┐
          ▼                     ▼                     ▼
  ┌───────────────┐   ┌────────────────┐   ┌──────────────────┐
  │ Image Service  │   │  Video Service  │   │  Audio Service    │
  │ face detect →  │   │ frame sample →  │   │ decode → mel-spec  │
  │ EffNetV2 →     │   │ face detect →   │   │ → CNN classifier   │
  │ Grad-CAM       │   │ EffNetV2 →      │   │                    │
  │                │   │ temporal vote   │   │                    │
  └───────────────┘   └────────────────┘   └──────────────────┘
```

---

## 3. Folder Structure

```
deepfake-detection/
├── backend/
│   ├── app/
│   │   ├── core/config.py          # paths, device, thresholds
│   │   ├── models/                 # EfficientNetV2 + audio CNN architectures
│   │   ├── services/                # image/video/audio/webcam inference logic, GradCAM, face detection
│   │   ├── routers/                 # REST + WebSocket endpoints
│   │   ├── utils/                   # preprocessing, audio features, validation, exceptions
│   │   ├── schemas.py                # Pydantic response models
│   │   └── main.py                   # FastAPI app entrypoint
│   ├── training/
│   │   ├── datasets/                 # FaceForensics++/CelebDF/DFDC/ASVspoof loaders
│   │   ├── train_image.py
│   │   ├── train_video.py
│   │   ├── train_audio.py
│   │   ├── evaluate.py
│   │   └── infer.py                   # CLI inference without running the server
│   ├── weights/                       # fine-tuned .pt files go here (gitignored)
│   ├── assets/                        # sample test files
│   ├── tests/                         # pytest API smoke tests
│   ├── requirements.txt
│   └── Dockerfile
├── frontend/
│   ├── src/
│   │   ├── components/                # ImageDetection, VideoDetection, AudioDetection, WebcamDetection, shared UI
│   │   ├── api/client.ts               # Axios + WebSocket client
│   │   ├── types/api.ts                # TypeScript types mirroring backend schemas
│   │   └── App.tsx
│   ├── package.json
│   └── Dockerfile
├── docker-compose.yml
├── .gitignore
└── README.md
```

---

## 4. Installation

### 4.1 Prerequisites

- **Python 3.12** — [python.org/downloads](https://www.python.org/downloads/)
- **Node.js 20+** and npm — [nodejs.org](https://nodejs.org/)
- **FFmpeg** (needed for video decoding — OpenCV and MoviePy both use it)
  - Windows: `winget install ffmpeg` or download from ffmpeg.org and add to PATH
  - macOS: `brew install ffmpeg`
  - Linux: `sudo apt-get install ffmpeg`

Verify installs:
```bash
python --version     # should print 3.12.x
node --version        # should print v20.x or higher
ffmpeg -version
```

### 4.2 Clone / unzip the project

```bash
unzip multimodal-deepfake-detection.zip
cd deepfake-detection
```

---

## 5. Backend Setup

```bash
cd backend

# 1. Create a virtual environment
python -m venv venv

# 2. Activate it
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the API
uvicorn app.main:app --reload
```

The first run downloads ImageNet weights for EfficientNetV2 automatically
via `torchvision` (requires internet access, one-time, ~80MB).

Backend will be live at **http://localhost:8000**
Interactive Swagger docs: **http://localhost:8000/docs**
Health check: **http://localhost:8000/api/health**

---

## 6. Frontend Setup

Open a **new terminal** (keep the backend running):

```bash
cd frontend
npm install
npm run dev
```

Frontend will be live at **http://localhost:5173** and will proxy
`/api/*` requests (including the webcam WebSocket) to the backend on
port 8000 automatically (see `vite.config.ts`).

---

## 7. Using the App

1. **Image tab** — drag & drop or click to upload a jpg/jpeg/png/bmp/webp.
   Get REAL/FAKE, confidence, probabilities, processing time, and a
   toggleable Grad-CAM heatmap.
2. **Video tab** — upload mp4/avi/mov/mkv. Frames are sampled and
   face-classified; you get an aggregated verdict, a per-frame timeline
   bar chart, and a heatmap of the most suspicious frame.
3. **Audio tab** — upload wav/mp3/flac/m4a. Get REAL/FAKE, waveform plot,
   and log-mel spectrogram plot.
4. **Live Webcam tab** — click "Start Camera" to grant camera access;
   frames stream to the backend over a WebSocket roughly 3x/second, and
   you'll see a green border (real) or red border (fake) live, plus FPS
   and per-frame processing time. Click "Stop Camera" to end the session.

---

## 8. Training on Real Datasets

Out of the box the models use untrained (randomly initialized) heads on
top of ImageNet backbones — fine for testing the pipeline, not for real
detection accuracy. To get real detection performance:

### 8.1 Image / Video model (FaceForensics++, CelebDF, or DFDC)

```bash
cd backend

# Step 1: extract face crops from raw videos into a real/ + fake/ folder layout
python training/datasets/extract_faces_from_videos.py \
    --real-dir /path/to/ffpp/original_sequences \
    --fake-dir /path/to/ffpp/manipulated_sequences/Deepfakes \
    --output-root /data/ffpp_frames \
    --frames-per-video 20

# (For DFDC, which ships as videos + metadata.json:)
python training/datasets/extract_faces_from_videos.py \
    --videos-dir /path/to/dfdc/train \
    --dfdc-metadata /path/to/dfdc/train/metadata.json \
    --output-root /data/dfdc_frames \
    --frames-per-video 15

# Step 2: train the image model
python training/train_image.py --data-root /data/ffpp_frames --epochs 10 --batch-size 32

# Step 3 (optional): train a separate video-specific checkpoint
python training/train_video.py --data-root /data/ffpp_frames --epochs 10
```

Checkpoints are saved to `backend/weights/image_efficientnetv2.pt` and
`backend/weights/video_efficientnetv2.pt` automatically whenever
validation accuracy improves.

### 8.2 Audio model (ASVspoof)

```bash
python training/train_audio.py \
    --asvspoof-protocol /path/to/ASVspoof2019.LA.cm.train.trn.txt \
    --asvspoof-audio-dir /path/to/ASVspoof2019_LA_train/flac \
    --epochs 15
```

Or with a simple custom real/fake folder of audio clips:
```bash
python training/train_audio.py --data-root /data/custom_audio --epochs 15
```

### 8.3 Evaluation

```bash
python training/evaluate.py --task image --data-root /data/ffpp_test --weights weights/image_efficientnetv2.pt
python training/evaluate.py --task audio --data-root /data/asvspoof_test --weights weights/audio_cnn_spectrogram.pt
```

Prints precision/recall/F1/confusion matrix via scikit-learn.

### 8.4 CLI inference (no server needed)

```bash
python training/infer.py --task image --input assets/sample_face.jpg
python training/infer.py --task audio --input assets/sample_audio.wav
```

---

## 9. Running with Docker

```bash
docker compose up --build
```

- Backend: http://localhost:8000
- Frontend: http://localhost:80

Fine-tuned weights persist across container rebuilds via the
`backend/weights` volume mount in `docker-compose.yml` — drop your
trained `.pt` files there and restart the backend container to pick
them up.

---

## 10. Troubleshooting

| Problem | Fix |
|---|---|
| `ModuleNotFoundError: torch` | Make sure the virtual environment is activated before `pip install -r requirements.txt` |
| Video upload fails with "container/codec could not be opened" | Install FFmpeg and ensure it's on your PATH; restart the terminal |
| `cv2.error` on face detection | Reinstall opencv-python: `pip install --force-reinstall opencv-python` |
| Webcam won't start in the browser | Browsers require HTTPS or `localhost` for camera access — use `http://localhost:5173`, not an IP address |
| WebSocket for live webcam doesn't connect | Confirm the backend is running on port 8000 and `vite.config.ts`'s proxy target matches |
| Predictions seem random / always the same class | You're running on an un-fine-tuned model — see §8 to train on real data |
| `413 file_too_large` | Limits: images 15MB, video 200MB, audio 30MB — configurable in `backend/app/core/config.py` |
| Slow inference on CPU | Expected — EfficientNetV2 on CPU is much slower than GPU. Install a CUDA-enabled `torch` build if you have an NVIDIA GPU |
| `pip install torch` fails / wrong CUDA version | Use the correct index URL from [pytorch.org/get-started/locally](https://pytorch.org/get-started/locally/) for your CUDA version |

---

## 11. Code Quality Notes

- Full type hints throughout the Python codebase (`from __future__ import annotations`)
- Modular architecture: routers only handle HTTP/WebSocket I/O and validation;
  all inference logic lives in `services/`; models are pure `nn.Module`s
- Every upload path validates extension, size, and decodability before
  hitting the model, with typed exceptions mapped to proper HTTP status codes
- Single-responsibility services: face detection, Grad-CAM, and each
  modality's pipeline are independently testable modules

---

## 12. API Reference (Swagger)

Full interactive documentation, generated automatically by FastAPI, is
available at **http://localhost:8000/docs** once the backend is running.
Endpoints:

- `GET  /api/health`
- `POST /api/image/detect`
- `POST /api/video/detect`
- `POST /api/audio/detect`
- `WS   /api/webcam/stream`
