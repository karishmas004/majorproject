import argparse
import random
import sys
from pathlib import Path
from collections import Counter

import cv2
from PIL import Image, ImageOps, UnidentifiedImageError

ROOT = Path(__file__).resolve().parents[2]
BACKEND = ROOT / "backend"

# Make the existing application modules importable.
sys.path.insert(0, str(BACKEND))
from app.services.face_detector import face_detector

REAL_DIRS = [
    BACKEND / "raw" / "real_openimages_kids",
    BACKEND / "raw" / "real_openimages_adults",
    BACKEND / "raw" / "real_coco",
]

FAKE_DIR = ROOT / "ai_generated"

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
SIZE = (512, 512)


def find_images(directory):
    if not directory.exists():
        print(f"WARNING: Missing directory: {directory}")
        return []

    return sorted(
        p for p in directory.rglob("*")
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )


def prepare_face(source, destination):
    """Detect the largest face, crop it, and save a 512x512 RGB image."""
    try:
        with Image.open(source) as image:
            image = ImageOps.exif_transpose(image).convert("RGB")
            rgb = cv2.cvtColor(
                __import__("numpy").array(image),
                cv2.COLOR_RGB2BGR,
            )

        box = face_detector.largest_face(rgb)
        if box is None:
            return "no_face"

        crop = face_detector.crop_with_margin(rgb, box, margin=0.65)
        if crop.size == 0:
            return "invalid_crop"

        crop = cv2.resize(
            crop,
            SIZE,
            interpolation=cv2.INTER_AREA
            if crop.shape[0] > SIZE[1]
            else cv2.INTER_CUBIC,
        )

        if crop.shape[:2] != (512, 512):
            return "invalid_crop"

        # Save as JPEG; convert BGR to RGB first.
        crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        Image.fromarray(crop).save(
            destination,
            format="JPEG",
            quality=95,
            optimize=True,
        )
        return "saved"

    except (
        OSError,
        ValueError,
        UnidentifiedImageError,
        cv2.error,
    ) as error:
        print(f"SKIPPED: {source} ({error})")
        return "error"


def process_class(label, sources, destination, limit, rng):
    rng.shuffle(sources)
    counts = Counter()
    failures = []
    saved = 0
    checked = 0

    print(f"\nProcessing {label}: target {limit} face crops")
    for source in sources:
        if saved >= limit:
            break

        checked += 1
        target = destination / f"{label}_{saved + 1:06d}.jpg"
        result = prepare_face(source, target)
        counts[result] += 1

        if result == "saved":
            saved += 1
        else:
            failures.append(str(source))

        if checked % 100 == 0:
            print(
                f"  Checked {checked}/{len(sources)}; "
                f"saved {saved}/{limit}"
            )

    print(
        f"{label}: saved={saved}, checked={checked}, "
        f"no_face={counts['no_face']}, "
        f"invalid_crop={counts['invalid_crop']}, "
        f"errors={counts['error']}"
    )
    return saved, failures, counts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=str(BACKEND / "dataset_512"))
    parser.add_argument("--max-real", type=int, default=6975)
    parser.add_argument("--max-fake", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.max_real < 1 or args.max_fake < 1:
        raise SystemExit("Both class limits must be positive.")

    output = Path(args.out).expanduser().resolve()
    real_output = output / "real"
    fake_output = output / "fake"

    for folder in (real_output, fake_output):
        if folder.exists() and any(folder.iterdir()):
            raise SystemExit(
                f"ERROR: {folder} is not empty. "
                "Remove the existing dataset before rebuilding it."
            )

    real_images = []
    for directory in REAL_DIRS:
        found = find_images(directory)
        print(f"Found {len(found)} images in {directory}")
        real_images.extend(found)

    fake_images = find_images(FAKE_DIR)
    print(f"Found {len(fake_images)} AI-generated images in {FAKE_DIR}")

    real_output.mkdir(parents=True, exist_ok=True)
    fake_output.mkdir(parents=True, exist_ok=True)

    rng = random.Random(args.seed)

    real_saved, real_failures, real_counts = process_class(
        "real", real_images, real_output, args.max_real, rng
    )
    fake_saved, fake_failures, fake_counts = process_class(
        "fake", fake_images, fake_output, args.max_fake, rng
    )

    report = output / "preparation_report.txt"
    report.write_text(
        "\n".join([
            "Face-crop dataset preparation report",
            f"Output: {output}",
            "Saved image dimensions: 512 x 512",
            "Format: JPEG",
            "Face detector: YuNet",
            "Crop selection: largest detected face with 0.65 margin",
            f"Real face crops saved: {real_saved}",
            f"Fake face crops saved: {fake_saved}",
            f"Real source images checked: {real_counts.total()}",
            f"Fake source images checked: {fake_counts.total()}",
            f"Real images without a face: {real_counts['no_face']}",
            f"Fake images without a face: {fake_counts['no_face']}",
            f"Real processing errors: {real_counts['error']}",
            f"Fake processing errors: {fake_counts['error']}",
            "",
            "Note: Image counts do not guarantee unique people or unique source images.",
        ]),
        encoding="utf-8",
    )

    print("\nPreparation complete.")
    print(f"Real face crops: {real_saved}")
    print(f"Fake face crops: {fake_saved}")
    print(f"Report: {report}")


if __name__ == "__main__":
    main()
