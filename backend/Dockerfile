FROM python:3.12-slim

# ffmpeg is required by moviepy / librosa's audioread fallback and by
# OpenCV's video decoding backend for mp4/avi/mov/mkv.
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Weights are volume-mounted in docker-compose so fine-tuned checkpoints
# persist and can be swapped without rebuilding the image.
RUN mkdir -p weights tmp assets

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
