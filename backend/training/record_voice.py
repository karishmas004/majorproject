"""Continuously record genuine voice clips into audio/real_own/."""

import signal
import time
from pathlib import Path

import sounddevice as sd
import soundfile as sf

SR = 16000
CLIP_SECONDS = 6
OUTPUT_DIR = Path("audio/real_own")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

recorded = 0
running = True


def stop_recording(signum, frame):
    global running
    running = False
    print("\nStopping recording after the current clip...")


signal.signal(signal.SIGINT, stop_recording)


def main():
    global recorded, running

    existing = len(list(OUTPUT_DIR.glob("*.wav")))

    print("=" * 55)
    print("CONTINUOUS VOICE RECORDING")
    print("=" * 55)
    print(f"Output directory: {OUTPUT_DIR.resolve()}")
    print(f"Existing WAV recordings: {existing}")
    print(f"Sample rate: {SR} Hz")
    print(f"Duration per clip: {CLIP_SECONDS} seconds")
    print("\nSpeak naturally. Recording starts in 3 seconds.")
    print("Press Ctrl+C to stop.")
    print("=" * 55)

    try:
        time.sleep(3)

        while running:
            print(
                f"\nRecording clip {recorded + 1} "
                f"({CLIP_SECONDS} seconds)...",
                flush=True,
            )

            audio = sd.rec(
                int(CLIP_SECONDS * SR),
                samplerate=SR,
                channels=1,
                dtype="float32",
            )
            sd.wait()

            if not running:
                break

            timestamp = time.time_ns()
            output_file = OUTPUT_DIR / f"own_{timestamp}_{recorded:05d}.wav"

            sf.write(output_file, audio, SR)

            recorded += 1
            print(f"Saved: {output_file.name}", flush=True)

    except KeyboardInterrupt:
        pass
    except Exception as error:
        print(f"\nRecording error: {error}")
    finally:
        print("\n" + "=" * 55)
        print("RECORDING SESSION FINISHED")
        print(f"New recordings saved: {recorded}")
        print(f"Output folder: {OUTPUT_DIR.resolve()}")
        print("=" * 55)


if __name__ == "__main__":
    main()