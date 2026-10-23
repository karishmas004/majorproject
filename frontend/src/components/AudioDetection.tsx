import { useState } from "react";
import { AudioLines, Clock, Timer } from "lucide-react";
import UploadDropzone from "./UploadDropzone";
import ConfidenceBar from "./ConfidenceBar";
import ErrorBanner from "./ErrorBanner";
import { Spinner } from "./Loader";
import { detectAudio, ApiError } from "../api/client";
import type { AudioDetectionResponse } from "../types/api";

export default function AudioDetection() {
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<AudioDetectionResponse | null>(null);
  const [error, setError] = useState<string | null>(null);

  async function handleFile(file: File) {
    setLoading(true);
    setError(null);
    setResult(null);
    try {
      const data = await detectAudio(file);
      setResult(data);
    } catch (err) {
      setError(err instanceof ApiError ? err.message : "Something went wrong analyzing this audio clip.");
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="grid gap-6 lg:grid-cols-2">
      <div className="space-y-4">
        <UploadDropzone
          accept=".wav,.mp3,.flac,.m4a,.opus,.aac,.ogg,.wma,.aiff,.aif,.amr,.caf,.webm,.mp4,.mp2,.au"
          label="Drop an audio clip, or click to browse"
          hint="WAV, MP3, FLAC, M4A, OPUS, AAC, OGG, WMA, AIFF, AMR, CAF, WebM, MP4, MP2, AU · up to 30MB"
          onFileSelected={handleFile}
          disabled={loading}
        />
        {error && <ErrorBanner message={error} />}
      </div>

      <div className="rounded-2xl border border-[#DCE3DA] bg-bg-card p-5 min-h-[320px]">
        {loading && <Spinner message="Extracting spectrogram and running inference..." />}

        {!loading && !result && !error && (
          <div className="flex h-full flex-col items-center justify-center gap-2 py-14 text-[#707A74]">
            <AudioLines className="h-8 w-8" />
            <p className="text-sm">Results will appear here</p>
          </div>
        )}

        {!loading && result && (
          <div className="space-y-5">
            <div className="flex items-center justify-between">
              <h3 className="font-semibold text-[#24352E]">Detection Result</h3>
              <div className="flex items-center gap-3 text-xs text-[#707A74]">
                <span className="flex items-center gap-1">
                  <Timer className="h-3.5 w-3.5" /> {result.duration_sec}s
                </span>
                <span className="flex items-center gap-1">
                  <Clock className="h-3.5 w-3.5" /> {result.processing_time_ms.toFixed(0)}ms
                </span>
              </div>
            </div>

            <ConfidenceBar
              label={result.label}
              confidence={result.confidence}
              fakeProbability={result.fake_probability}
              realProbability={result.real_probability}
            />

            
            <div>
              <p className="text-xs uppercase tracking-wide text-[#707A74] mb-2">Waveform</p>
              <img
                src={`data:image/png;base64,${result.waveform_base64}`}
                alt="Waveform"
                className="w-full rounded-lg border border-[#DCE3DA]"
              />
            </div>

            <div>
              <p className="text-xs uppercase tracking-wide text-[#707A74] mb-2">Log-Mel Spectrogram</p>
              <img
                src={`data:image/png;base64,${result.spectrogram_base64}`}
                alt="Spectrogram"
                className="w-full rounded-lg border border-[#DCE3DA]"
              />
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
