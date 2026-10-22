from pathlib import Path
from PIL import Image, ImageOps
import random
import hashlib
import shutil

ROOT = Path(__file__).resolve().parents[2]
BACKEND = ROOT / "backend"
OUTPUT = BACKEND / "test_dataset"

REAL_SOURCES = [
    BACKEND / "raw" / "real_openimages_kids",
    BACKEND / "raw" / "real_openimages_adults",
    BACKEND / "raw" / "real_coco",
]
FAKE_SOURCE = ROOT / "ai_generated"

EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
SEED = 7319
TEST_PER_CLASS = 300
SIZE = (224, 224)


def files_under(folder):
    if not folder.exists():
        raise FileNotFoundError(f"Missing source directory: {folder}")

    return sorted(
        p for p in folder.rglob("*")
        if p.is_file() and p.suffix.lower() in EXTENSIONS
    )


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def make_jpeg(source, destination):
    with Image.open(source) as im:
        im = ImageOps.exif_transpose(im).convert("RGB")
        im = ImageOps.pad(
            im, SIZE,
            method=Image.Resampling.LANCZOS,
            color=(0, 0, 0),
        )
        im.save(destination, "JPEG", quality=95)


def main():
    if OUTPUT.exists() and any(OUTPUT.rglob("*")):
        raise SystemExit(
            f"{OUTPUT} already contains files. "
            "Inspect or rename it before rerunning."
        )

    real_candidates = []
    for folder in REAL_SOURCES:
        real_candidates.extend(files_under(folder))

    fake_candidates = files_under(FAKE_SOURCE)

    # Exclude exact duplicate file contents within each class.
    def unique_files(paths):
        seen = set()
        result = []
        for p in paths:
            digest = sha256(p)
            if digest not in seen:
                seen.add(digest)
                result.append(p)
        return result

    real_candidates = unique_files(real_candidates)
    fake_candidates = unique_files(fake_candidates)

    # Exclude exact cross-class duplicates.
    real_hashes = {sha256(p) for p in real_candidates}
    fake_candidates = [
        p for p in fake_candidates if sha256(p) not in real_hashes
    ]

    rng = random.Random(SEED)
    rng.shuffle(real_candidates)
    rng.shuffle(fake_candidates)

    # Only images beyond the first 2,000 in each source list are
    # eligible for this test set. This prevents overlap with the
    # current dataset_512 sampling pool.
    real_candidates = real_candidates[2000:]
    fake_candidates = fake_candidates[2000:]

    if len(real_candidates) < TEST_PER_CLASS:
        raise SystemExit(
            f"Not enough eligible real images: {len(real_candidates)}. "
            "A separate source or a redesigned split is needed."
        )

    if len(fake_candidates) < TEST_PER_CLASS:
        raise SystemExit(
            f"Not enough eligible fake images: {len(fake_candidates)}."
        )

    selected = {
        "real": real_candidates[:TEST_PER_CLASS],
        "fake": fake_candidates[:TEST_PER_CLASS],
    }

    for label, sources in selected.items():
        destination = OUTPUT / label
        destination.mkdir(parents=True, exist_ok=True)

        for i, source in enumerate(sources, start=1):
            target = destination / f"{label}_{i:04d}.jpg"
            try:
                make_jpeg(source, target)
            except (OSError, ValueError) as exc:
                print(f"Skipped {source}: {exc}")

        print(
            f"{label}: "
            f"{len(list(destination.glob('*.jpg')))} test images"
        )

    print(f"\nTest dataset created at: {OUTPUT}")
    print("Verify source overlap before treating results as independent.")


if __name__ == "__main__":
    main()
