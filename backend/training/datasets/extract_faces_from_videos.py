"""
Convert a folder of REAL videos and a folder of FAKE videos into
face-cropped JPEG frames laid out for FaceFrameDataset:

    output_root/real/<video_stem>_<frame_idx>.jpg
    output_root/fake/<video_stem>_<frame_idx>.jpg

Also handles DFDC's metadata.json (video filename -> "REAL"/"FAKE")
convention via --dfdc-metadata.

Usage:
    python extract_faces_from_videos.py \
        --real-dir /data/ffpp/original_sequences \
        --fake-dir /data/ffpp/manipulated_sequences/Deepfakes \
        --output-root /data/ffpp_frames \
        --frames-per-video 20

    python extract_faces_from_videos.py \
        --videos-dir /data/dfdc/train \
        --dfdc-metadata /data/dfdc/train/metadata.json \
        --output-root /data/dfdc_frames \
        --frames-per-video 15
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2

sys.path.append(str(Path(__file__).resolve().parents[2]))  # allow `app.*` imports
from app.services.face_detector import face_detector  # noqa: E402

VIDEO_EXT = {".mp4", ".avi", ".mov", ".mkv"}


def extract_from_video(video_path: Path, out_dir: Path, frames_per_video: int) -> int:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"  [skip] could not open {video_path}")
        return 0

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
    step = max(1, total // frames_per_video)
    out_dir.mkdir(parents=True, exist_ok=True)

    saved = 0
    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if idx % step == 0 and saved < frames_per_video:
            box = face_detector.largest_face(frame)
            if box is not None:
                crop = face_detector.crop_with_margin(frame, box)
                if crop.size > 0:
                    out_path = out_dir / f"{video_path.stem}_{idx:05d}.jpg"
                    cv2.imwrite(str(out_path), crop)
                    saved += 1
        idx += 1
    cap.release()
    return saved


def process_folder(video_dir: Path, out_dir: Path, frames_per_video: int) -> None:
    videos = [p for p in sorted(video_dir.rglob("*")) if p.suffix.lower() in VIDEO_EXT]
    print(f"Found {len(videos)} videos in {video_dir}")
    for i, video_path in enumerate(videos, 1):
        n = extract_from_video(video_path, out_dir, frames_per_video)
        print(f"  [{i}/{len(videos)}] {video_path.name}: saved {n} face crops")


def process_dfdc(videos_dir: Path, metadata_path: Path, output_root: Path, frames_per_video: int) -> None:
    metadata = json.loads(metadata_path.read_text())
    for filename, info in metadata.items():
        video_path = videos_dir / filename
        if not video_path.exists():
            continue
        label = info.get("label", "").upper()
        subdir = "fake" if label == "FAKE" else "real"
        out_dir = output_root / subdir
        n = extract_from_video(video_path, out_dir, frames_per_video)
        print(f"{filename} [{label}]: saved {n} face crops")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--real-dir", type=Path, help="Folder of real videos (FaceForensics++/CelebDF style)")
    parser.add_argument("--fake-dir", type=Path, help="Folder of fake videos")
    parser.add_argument("--videos-dir", type=Path, help="Folder of DFDC videos (used with --dfdc-metadata)")
    parser.add_argument("--dfdc-metadata", type=Path, help="Path to DFDC metadata.json")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--frames-per-video", type=int, default=20)
    args = parser.parse_args()

    if args.dfdc_metadata:
        if not args.videos_dir:
            parser.error("--videos-dir is required with --dfdc-metadata")
        process_dfdc(args.videos_dir, args.dfdc_metadata, args.output_root, args.frames_per_video)
    else:
        if not args.real_dir or not args.fake_dir:
            parser.error("--real-dir and --fake-dir are required unless using --dfdc-metadata")
        process_folder(args.real_dir, args.output_root / "real", args.frames_per_video)
        process_folder(args.fake_dir, args.output_root / "fake", args.frames_per_video)

    print("Done.")


if __name__ == "__main__":
    main()
