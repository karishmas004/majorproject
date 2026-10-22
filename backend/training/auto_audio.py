"""Run from backend: python training/auto_audio.py"""

from __future__ import annotations

import argparse
import random
import sys
from collections import defaultdict
from pathlib import Path

import joblib
import librosa
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.append(str(Path(__file__).resolve().parents[1]))
from app.services import audio_probe as probe  # noqa: E402


EXT = {".wav", ".mp3", ".flac", ".m4a", ".ogg"}
SR = probe.SR


def frac(key: str) -> float:
    """Deterministic split value."""
    return random.Random(key).random()


# ------------------------------------------------------------
# ASVspoof: use exactly the IDs in the verified manifest
# ------------------------------------------------------------

def manifest_items(la: Path, manifest: Path):
    if not manifest.is_file():
        raise SystemExit(f"Manifest not found: {manifest}")

    audio_dir = la / "ASVspoof2019_LA_train" / "flac"

    ids = [
        line.strip()
        for line in manifest.read_text().splitlines()
        if line.strip()
    ]

    if len(ids) != 2000:
        raise SystemExit(
            f"Expected exactly 2000 manifest entries, found {len(ids)}"
        )

    if len(set(ids)) != len(ids):
        raise SystemExit("Manifest contains duplicate recording IDs.")

    missing = [
        file_id for file_id in ids
        if not (audio_dir / f"{file_id}.flac").is_file()
    ]

    if missing:
        raise SystemExit(
            f"{len(missing)} manifest audio files are missing. "
            f"First missing IDs: {missing[:5]}"
        )

    # Reproducible split: 80% training, 20% held-out testing.
    ids = sorted(ids)
    random.Random(42).shuffle(ids)

    train_ids = ids[:1600]
    test_ids = ids[1600:]

    items = []

    for file_id in train_ids:
        items.append({
            "path": str(audio_dir / f"{file_id}.flac"),
            "label": 1,  # spoof = FAKE
            "group": "ASVspoof spoof (manifest)",
            "split": "train",
            "trusted": False,
        })

    for file_id in test_ids:
        items.append({
            "path": str(audio_dir / f"{file_id}.flac"),
            "label": 1,
            "group": "ASVspoof spoof (manifest holdout)",
            "split": "test",
            "trusted": False,
        })

    print(f"  Manifest recordings: {len(ids)}")
    print(f"  ASVspoof training: {len(train_ids)}")
    print(f"  ASVspoof held-out test: {len(test_ids)}")

    return items


# ------------------------------------------------------------
# Genuine recordings and TTS recordings
# ------------------------------------------------------------

def extra_items(audio: Path):
    items = []

    if not audio.exists():
        return items

    folders = sorted(audio.glob("real_*")) + sorted(audio.glob("fake_*"))

    for folder in folders:
        if not folder.is_dir():
            continue

        label = 0 if folder.name.startswith("real_") else 1
        trusted = folder.name.startswith("real_own")

        files = sorted(
            p for p in folder.rglob("*")
            if p.is_file() and p.suffix.lower() in EXT
        )

        for p in files:
            if label == 0:
                # Hold out approximately 20% of genuine clips.
                split = "test" if frac(str(p)) < 0.20 else "train"

            else:
                # TTS: keep each voice entirely in one split.
                voice = p.name.split("__")[0]
                split = "test" if frac("tts:" + voice) < 0.25 else "train"

            items.append({
                "path": str(p),
                "label": label,
                "group": folder.name,
                "split": split,
                "trusted": trusted,
            })

    return items


# ------------------------------------------------------------
# Audio augmentation
# ------------------------------------------------------------

def degrade(w, rnd):
    """Apply mild noise and resampling to some clips."""
    if len(w) == 0:
        return w.astype(np.float32)

    w = w * rnd.uniform(0.3, 1.0)

    if rnd.random() < 0.5:
        w = librosa.resample(
            librosa.resample(w, orig_sr=SR, target_sr=8000),
            orig_sr=8000,
            target_sr=SR,
        )

    snr = rnd.uniform(15, 35)
    noise = np.random.RandomState(
        rnd.randint(0, 2**31 - 1)
    ).randn(len(w))

    noise *= np.sqrt(
        np.mean(w ** 2) / (10 ** (snr / 10)) + 1e-12
    )

    return (w + noise).astype(np.float32)


def load_clip(path: str):
    w, _ = librosa.load(
        path,
        sr=SR,
        mono=True,
        duration=20,
    )

    if len(w) == 0:
        raise ValueError("Audio file contains no readable samples.")

    rnd = random.Random(path)

    if rnd.random() < 0.5:
        w = degrade(w, rnd)

    return w


# ------------------------------------------------------------
# Feature extraction and persistent cache
# ------------------------------------------------------------

def embed_items(items, cache_path: Path):
    cache = {}

    if cache_path.exists():
        with np.load(cache_path, allow_pickle=False) as z:
            cached_paths = z["paths"].tolist()
            cached_features = z["X"]

            cache = {
                path: feature
                for path, feature in zip(
                    cached_paths,
                    cached_features,
                )
            }

        print(f"  Existing cached embeddings: {len(cache)}")

    def save():
        if not cache:
            return

        np.savez(
            cache_path,
            paths=np.array(list(cache.keys())),
            X=np.stack(list(cache.values())),
        )

    todo = [it for it in items if it["path"] not in cache]

    print(f"  Already cached: {len(items) - len(todo)}")
    print(f"  New embeddings required: {len(todo)}")

    for n, it in enumerate(todo, start=1):
        try:
            cache[it["path"]] = probe.embed_clip(
                load_clip(it["path"])
            )
        except Exception as e:
            print(f"  SKIP {it['path']}: {e}")

        if n % 100 == 0 or n == len(todo):
            print(f"  Embedded {n}/{len(todo)} new clips")

        if n % 500 == 0:
            save()

    if todo:
        save()

    usable = [
        it for it in items
        if it["path"] in cache
    ]

    if not usable:
        raise SystemExit("No audio embeddings were successfully created.")

    X = np.stack([
        cache[it["path"]]
        for it in usable
    ])

    return usable, X


# ------------------------------------------------------------
# Classifier and held-out evaluation
# ------------------------------------------------------------

def make_clf():
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(
            C=0.05,
            max_iter=3000,
            class_weight="balanced",
            random_state=42,
        ),
    )


def evaluate(items, X, y):
    train_idx = [
        i for i, it in enumerate(items)
        if it["split"] == "train"
    ]

    test_idx = [
        i for i, it in enumerate(items)
        if it["split"] == "test"
    ]

    if not train_idx or not test_idx:
        raise SystemExit("Training or testing set is empty.")

    if len(set(y[train_idx])) != 2:
        raise SystemExit(
            "Training data must contain both REAL and FAKE examples."
        )

    if len(set(y[test_idx])) != 2:
        raise SystemExit(
            "Testing data must contain both REAL and FAKE examples."
        )

    clf = make_clf()

    weights = np.array([
        5.0 if items[i]["trusted"] else 1.0
        for i in train_idx
    ])

    clf.fit(
        X[train_idx],
        y[train_idx],
        logisticregression__sample_weight=weights,
    )

    probabilities = clf.predict_proba(X[test_idx])[:, 1]
    predictions = (probabilities >= 0.5).astype(int)
    actual = y[test_idx]

    print("\n" + "=" * 65)
    print("HELD-OUT EVALUATION")
    print("=" * 65)

    print(f"Training clips: {len(train_idx)}")
    print(f"Testing clips:  {len(test_idx)}")
    print(f"Accuracy:       {accuracy_score(actual, predictions):.4f}")
    print(f"ROC-AUC:        {roc_auc_score(actual, probabilities):.4f}")

    print("\nClassification report:")
    print(classification_report(
        actual,
        predictions,
        labels=[0, 1],
        target_names=["REAL", "FAKE"],
        zero_division=0,
    ))

    print("Confusion matrix (rows=actual, columns=predicted):")
    print(confusion_matrix(
        actual,
        predictions,
        labels=[0, 1],
    ))

    groups = defaultdict(list)

    for position, item_idx in enumerate(test_idx):
        groups[items[item_idx]["group"]].append(position)

    print("\nPer-group test results:")

    for group, positions in sorted(groups.items()):
        positions = np.asarray(positions)
        group_accuracy = accuracy_score(
            actual[positions],
            predictions[positions],
        )
        fake_probability = probabilities[positions].mean()

        print(
            f"{group:45} "
            f"N={len(positions):5d} "
            f"accuracy={group_accuracy:.1%} "
            f"mean_P(fake)={fake_probability:.3f}"
        )

    return clf


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--la", default="../LA")
    parser.add_argument("--audio", default="audio")
    parser.add_argument("--data", default="data")
    parser.add_argument(
        "--manifest",
        default="audio/la_spoof_2000_manifest.txt",
    )

    args = parser.parse_args()

    la = Path(args.la)
    audio = Path(args.audio)
    manifest = Path(args.manifest)
    data = Path(args.data)

    data.mkdir(parents=True, exist_ok=True)

    print("== 1/3: Collecting audio")

    items = manifest_items(la, manifest)
    items += extra_items(audio)

    print(f"\nTotal clips discovered: {len(items)}")
    print(f"REAL clips: {sum(it['label'] == 0 for it in items)}")
    print(f"FAKE clips: {sum(it['label'] == 1 for it in items)}")

    print("\nDataset split before feature extraction:")

    for split in ("train", "test"):
        subset = [it for it in items if it["split"] == split]
        real_count = sum(it["label"] == 0 for it in subset)
        fake_count = sum(it["label"] == 1 for it in subset)

        print(
            f"  {split}: {len(subset)} total | "
            f"{real_count} REAL | {fake_count} FAKE"
        )

    print("\n== 2/3: Extracting Wav2Vec2 embeddings")
    print("Existing embeddings will be reused where available.")

    items, X = embed_items(
        items,
        data / "audio_feats_cache.npz",
    )

    y = np.array([
        it["label"]
        for it in items
    ])

    print("\n== 3/3: Evaluating and training final classifier")

    evaluate(items, X, y)

    # Fit the final deployable classifier on all successfully
    # embedded clips after completing the held-out evaluation.
    weights = np.array([
        5.0 if it["trusted"] else 1.0
        for it in items
    ])

    final_clf = make_clf()

    final_clf.fit(
        X,
        y,
        logisticregression__sample_weight=weights,
    )

    probe.PROBE_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    joblib.dump(final_clf, probe.PROBE_PATH)

    print(f"\nFinal classifier saved to: {probe.PROBE_PATH}")
    print("Training pipeline completed.")


if __name__ == "__main__":
    main()