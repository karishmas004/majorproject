import argparse
import shutil
from pathlib import Path

import fiftyone.zoo as foz

ap = argparse.ArgumentParser()
ap.add_argument("--source", choices=["coco", "openimages"], required=True)
ap.add_argument("--classes", nargs="+", required=True)
ap.add_argument("--n", type=int, default=500)
ap.add_argument("--min-area", type=float, default=0.12)  # person must fill 12%+ of the photo
ap.add_argument("--out", required=True)
ap.add_argument("--seed", type=int, default=51)
a = ap.parse_args()

name = "coco-2017" if a.source == "coco" else "open-images-v7"
ds = foz.load_zoo_dataset(
    name, split="validation", label_types=["detections"], classes=a.classes,
    shuffle=True, seed=a.seed, max_samples=a.n * 4,
)
wanted = {c.lower() for c in a.classes}
out = Path(a.out)
out.mkdir(parents=True, exist_ok=True)
kept = 0
for s in ds:
    dets = s.ground_truth.detections if s.ground_truth else []
    areas = [d.bounding_box[2] * d.bounding_box[3] for d in dets if d.label.lower() in wanted]
    if areas and max(areas) >= a.min_area:
        destination = out / Path(s.filepath).name

        if destination.exists():
            continue

        shutil.copy2(s.filepath, destination)
        kept += 1
        if kept >= a.n:
            break
print(f"kept {kept} images in {out}")
