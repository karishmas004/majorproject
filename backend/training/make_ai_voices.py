"""Generate 2000 AI voice clips concurrently, preserving valid existing files."""

import asyncio
import random
from pathlib import Path

import edge_tts

TARGET_COUNT = 2000
CONCURRENCY = 5
OUTPUT_DIR = Path("audio/fake_tts")
MAX_RETRIES = 3
REQUEST_TIMEOUT = 30

SENTENCES = [
    "The weather today is quite pleasant, so we decided to walk to the market.",
    "Please remember to submit your assignment before the deadline on Friday.",
    "I never expected the meeting to run this long, but we finally reached an agreement.",
    "Can you tell me how to get to the nearest railway station from here?",
    "My brother bought a new laptop last week and it works really well.",
    "The company announced record profits during the third quarter of the year.",
    "She opened the window and listened to the rain falling on the old roof.",
    "We will start the project next Monday and finish it within two months.",
    "Could you please send me the report by tomorrow morning?",
    "The train was delayed by almost an hour because of heavy fog.",
    "He has been learning to play the guitar for about three years now.",
    "There are twenty four students in the class and most of them live nearby.",
    "I would like to order two coffees and a slice of chocolate cake.",
    "The doctor advised him to drink more water and sleep early every night.",
    "Our flight lands at half past six, so please wait for us at the gate.",
    "It was a long journey, but the view from the top of the hill was worth it.",
    "The new library opens next month and will have more than ten thousand books.",
    "She explained the problem slowly so that everyone could understand it.",
    "They celebrated their anniversary with a quiet dinner at a small restaurant.",
    "Nobody knew that the old house at the end of the street had a secret garden.",
    "Turn left at the second traffic light and the pharmacy will be on your right.",
    "I forgot my umbrella at home, and now it is raining heavily outside.",
    "The students gathered in the hall to listen to the principal's speech.",
    "He spent the whole weekend repairing the bicycle that his father gave him.",
    "Please keep your phone on silent while the presentation is going on.",
    "The market was crowded with people buying fruits, vegetables and flowers.",
    "After years of hard work, she finally opened her own bakery in the city.",
    "We should leave early tomorrow to avoid the traffic near the bridge.",
    "The film was longer than expected, but the ending made it memorable.",
    "Thank you for calling, and have a wonderful rest of your day.",
]


def valid_mp3(path):
    # Reject empty or suspiciously small files.
    return path.is_file() and path.stat().st_size > 1024


async def generate_one(semaphore, voice, index, sentence):
    output = OUTPUT_DIR / f"{voice}__{index:02d}.mp3"

    async with semaphore:
        # Preserve existing valid recordings.
        if valid_mp3(output):
            return False

        for attempt in range(1, MAX_RETRIES + 1):
            temp = output.with_suffix(".tmp.mp3")

            try:
                await asyncio.wait_for(
                    edge_tts.Communicate(sentence, voice).save(str(temp)),
                    timeout=REQUEST_TIMEOUT,
                )

                if not valid_mp3(temp):
                    raise RuntimeError("Generated audio is empty or too small")

                # Replace an invalid old file only after successful generation.
                temp.replace(output)
                return True

            except Exception as error:
                temp.unlink(missing_ok=True)

                if attempt == MAX_RETRIES:
                    print(
                        f"\nFailed: {voice} sentence {index}: {error}",
                        flush=True,
                    )
                else:
                    await asyncio.sleep(attempt * 2)

        return False


async def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Remove only empty MP3 files; keep all valid recordings.
    for path in OUTPUT_DIR.glob("*.mp3"):
        if path.exists() and path.stat().st_size == 0:
            path.unlink()
            print(f"Removed empty file: {path.name}")

    existing = sum(1 for p in OUTPUT_DIR.glob("*.mp3") if valid_mp3(p))

    print(f"Valid existing recordings: {existing}", flush=True)
    print(f"Target: {TARGET_COUNT}", flush=True)

    if existing >= TARGET_COUNT:
        print("Target already reached.")
        return

    voices = sorted(
        v["ShortName"]
        for v in await asyncio.wait_for(edge_tts.list_voices(), timeout=30)
        if v["Locale"].startswith("en-")
    )

    random.Random(0).shuffle(voices)

    semaphore = asyncio.Semaphore(CONCURRENCY)
    jobs = []

    for voice in voices:
        # Preserve the original filename convention.
        for index, sentence in enumerate(SENTENCES):
            output = OUTPUT_DIR / f"{voice}__{index:02d}.mp3"

            if valid_mp3(output):
                continue

            jobs.append((voice, index, sentence))

    jobs = jobs[: TARGET_COUNT - existing]

    print(f"Pending clips: {len(jobs)}", flush=True)

    completed = 0
    failed = 0

    tasks = [
        asyncio.create_task(generate_one(semaphore, voice, index, sentence))
        for voice, index, sentence in jobs
    ]

    try:
        for task in asyncio.as_completed(tasks):
            try:
                generated = await task
                if generated:
                    completed += 1
                else:
                    failed += 1

                if (completed + failed) % 10 == 0:
                    current = sum(
                        1 for p in OUTPUT_DIR.glob("*.mp3") if valid_mp3(p)
                    )
                    print(
                        f"Progress: {current}/{TARGET_COUNT} valid clips "
                        f"| New: {completed} | Failed/skipped: {failed}",
                        flush=True,
                    )
            except Exception as error:
                failed += 1
                print(f"Task error: {error}", flush=True)

    except KeyboardInterrupt:
        print("\nStopping generation...", flush=True)
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        raise

    final_count = sum(
        1 for p in OUTPUT_DIR.glob("*.mp3") if valid_mp3(p)
    )

    print("\nGeneration finished.", flush=True)
    print(f"Valid MP3 recordings: {final_count}/{TARGET_COUNT}")
    print(f"Generated during this run: {completed}")
    print(f"Failed/skipped: {failed}")

    if final_count < TARGET_COUNT:
        print("Some clips failed. Run this script again to retry them.")


if __name__ == "__main__":
    asyncio.run(main())