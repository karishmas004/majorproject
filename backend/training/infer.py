"""
Standalone CLI inference (no server needed) — useful for quick checks
or batch-scoring a folder of files.

Usage:
    python infer.py --task image --input /path/to/photo.jpg
    python infer.py --task video --input /path/to/clip.mp4
    python infer.py --task audio --input /path/to/clip.wav
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))
from app.services import model_registry  # noqa: E402
from app.services.audio_service import analyze_audio  # noqa: E402
from app.services.image_service import analyze_image  # noqa: E402
from app.services.video_service import analyze_video  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=["image", "video", "audio"], required=True)
    parser.add_argument("--input", required=True)
    args = parser.parse_args()

    model_registry.load_all_models()

    file_bytes = Path(args.input).read_bytes()
    filename = Path(args.input).name

    if args.task == "image":
        result = analyze_image(file_bytes, filename)
    elif args.task == "video":
        result = analyze_video(file_bytes, filename)
    else:
        result = analyze_audio(file_bytes, filename)

    # Strip large base64 payloads for terminal-friendly output.
    printable = {k: v for k, v in result.items() if "base64" not in k and k != "timeline"}
    print(json.dumps(printable, indent=2))


if __name__ == "__main__":
    main()
