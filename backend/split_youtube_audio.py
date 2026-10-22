from pathlib import Path
import subprocess
import json

SOURCE_DIR = Path("youtube_audio")
OUTPUT_DIR = Path("audio/real_own")
STATE_FILE = OUTPUT_DIR / "processed_sources.json"

TARGET_COUNT = 6000
SKIP_SECONDS = 300
CLIP_SECONDS = 6

SUPPORTED_FORMATS = {
    ".mp3", ".wav", ".m4a", ".aac",
    ".flac", ".ogg", ".opus", ".wma"
}

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def count_wav_files():
    return sum(
        1 for p in OUTPUT_DIR.iterdir()
        if p.is_file() and p.suffix.lower() == ".wav"
    )

if not SOURCE_DIR.is_dir():
    raise SystemExit(f"Source folder not found: {SOURCE_DIR.resolve()}")

if STATE_FILE.exists():
    processed = set(json.loads(STATE_FILE.read_text()))
else:
    processed = set()

audio_files = sorted(
    p for p in SOURCE_DIR.iterdir()
    if p.is_file() and p.suffix.lower() in SUPPORTED_FORMATS
)

new_files = [p for p in audio_files if p.name not in processed]

print(f"Existing WAV clips: {count_wav_files()}")
print(f"Previously processed sources: {len(processed)}")
print(f"New source files to process: {len(new_files)}")

if count_wav_files() >= TARGET_COUNT:
    print("Target of 6000 WAV files already reached.")
    raise SystemExit(0)

if not new_files:
    print("No new audio files found. Nothing to process.")
    raise SystemExit(0)

for source in new_files:
    if count_wav_files() >= TARGET_COUNT:
        print("Reached 6000 WAV files. Stopping.")
        break

    print(f"\nProcessing new audio: {source.name}")

    try:
        result = subprocess.run(
            [
                "ffprobe", "-v", "error",
                "-show_entries", "format=duration",
                "-of", "default=noprint_wrappers=1:nokey=1",
                str(source)
            ],
            capture_output=True,
            text=True,
            check=True
        )

        duration = float(result.stdout.strip())

        if duration <= SKIP_SECONDS:
            print("Audio is 5 minutes or shorter; skipping.")
            processed.add(source.name)
            STATE_FILE.write_text(json.dumps(sorted(processed), indent=2))
            continue

        start = SKIP_SECONDS
        clip_number = 0

        while start + CLIP_SECONDS <= duration:
            if count_wav_files() >= TARGET_COUNT:
                print("Reached 6000 WAV files. Stopping.")
                break

            output = OUTPUT_DIR / (
                f"{source.stem}_clip_{clip_number:04d}.wav"
            )

            if not output.exists():
                subprocess.run(
                    [
                        "ffmpeg", "-nostdin", "-hide_banner",
                        "-loglevel", "error",
                        "-ss", str(start),
                        "-i", str(source),
                        "-t", str(CLIP_SECONDS),
                        "-vn", "-ac", "1", "-ar", "16000",
                        "-c:a", "pcm_s16le",
                        str(output)
                    ],
                    capture_output=True,
                    text=True,
                    check=True
                )

            start += CLIP_SECONDS
            clip_number += 1

        # Mark the source processed only after its loop finishes.
        if count_wav_files() < TARGET_COUNT:
            processed.add(source.name)
            STATE_FILE.write_text(json.dumps(sorted(processed), indent=2))

        print(f"Current WAV count: {count_wav_files()}")

    except Exception as error:
        print(f"ERROR processing {source.name}: {error}")

print("\nProcessing finished.")
print(f"Total WAV files: {count_wav_files()}")
print(f"Remaining: {max(0, TARGET_COUNT - count_wav_files())}")
