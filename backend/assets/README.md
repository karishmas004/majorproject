# Sample Assets

These are **synthetically generated placeholders**, not real photos/audio
(this project was built in an offline sandbox with no access to image/audio
hosting sites). They're only good for a quick smoke test that the API,
face detector, and audio pipeline run end-to-end without crashing:

- `sample_face.jpg` — a simple drawn face shape. OpenCV's Haar cascade may
  or may not detect it as a face (it's not photorealistic); if it doesn't,
  swap in a real face photo to test the `/api/image/detect` endpoint properly.
- `sample_audio.wav` — a 2-second pure sine tone, not real speech. Good for
  testing that `/api/audio/detect` decodes audio and returns a response;
  meaningless for judging real/fake accuracy.

For real testing and, more importantly, for training meaningful models,
download proper datasets: FaceForensics++, CelebDF, DFDC, or ASVspoof
(see the main README's "Training" section).
