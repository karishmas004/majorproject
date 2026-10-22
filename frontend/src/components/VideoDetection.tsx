import { useState } from "react";
import { Video, Clock, Layers, ScanFace } from "lucide-react";
import UploadDropzone from "./UploadDropzone";
import ConfidenceBar from "./ConfidenceBar";
import ErrorBanner from "./ErrorBanner";
import { Spinner } from "./Loader";
import { detectVideo, ApiError } from "../api/client";
import type { VideoDetectionResponse } from "../types/api";

export default function VideoDetection() {
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<VideoDetectionResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [showHeatmap, setShowHeatmap] = useState(true);

  async function handleFile(file: File) {
    setLoading(true);
    setError(null);
    setResult(null);
    try {
      const data = await detectVideo(file);
      setResult(data);
    } catch (err) {
      setError(err instanceof ApiError ? err.message : "Something went wrong analyzing this video.");
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="grid gap-6 lg:grid-cols-2">
      <div className="space-y-4">
        <UploadDropzone
          accept=".mp4,.avi,.mov,.mkv"
          label="Drop a video, or click to browse"
          hint="MP4, AVI, MOV, MKV · up to 200MB"
          onFileSelected={handleFile}
          disabled={loading}
        />
        {error && <ErrorBanner message={error} />}
      </div>

      <div className="rounded-2xl border border-[#DCE3DA] bg-bg-card p-5 min-h-[320px]">
        {loading && <Spinner message="Sampling frames, detecting faces, and aggregating predictions..." />}

        {!loading && !result && !error && (
          <div className="flex h-full flex-col items-center justify-center gap-2 py-14 text-[#707A74]">
            <Video className="h-8 w-8" />
            <p className="text-sm">Results will appear here</p>
          </div>
        )}

        {!loading && result && (
          <div className="space-y-5">
            <div className="flex items-center justify-between">
              <h3 className="font-semibold text-[#24352E]">Detection Result</h3>
              <div className="flex items-center gap-1 text-xs text-[#707A74]">
                <Clock className="h-3.5 w-3.5" /> {(result.processing_time_ms / 1000).toFixed(1)}s
              </div>
            </div>

            {result.most_suspicious_frame_base64 && (
              <div className="relative rounded-xl overflow-hidden border border-[#DCE3DA]">
                <img
                  src={`data:image/jpeg;base64,${
                    showHeatmap && result.most_suspicious_heatmap_base64
                      ? result.most_suspicious_heatmap_base64
                      : result.most_suspicious_frame_base64
                  }`}
                  alt="Most suspicious frame"
                  className="w-full object-contain max-h-64"
                />
                <span className="absolute top-2 left-2 rounded-lg bg-black/60 px-2 py-1 text-xs text-white backdrop-blur">
                  Most suspicious frame
                </span>
                {result.most_suspicious_heatmap_base64 && (
                  <button
                    onClick={() => setShowHeatmap((s) => !s)}
                    className="absolute bottom-2 right-2 rounded-lg bg-black/60 px-3 py-1.5 text-xs text-white backdrop-blur hover:bg-black/80"
                  >
                    {showHeatmap ? "Show Original" : "Show Heatmap"}
                  </button>
                )}
              </div>
            )}

            <ConfidenceBar
              label={result.label}
              confidence={result.confidence}
              fakeProbability={result.fake_probability}
              realProbability={result.real_probability}
            />

            <div className="flex items-center gap-4 text-sm text-[#5F6868]">
              <span className="flex items-center gap-1">
                <Layers className="h-4 w-4" /> {result.frames_analyzed} frames analyzed
              </span>
              <span className="flex items-center gap-1">
                <ScanFace className="h-4 w-4" /> {result.frames_with_face} with a face
              </span>
            </div>

            <FrameTimeline timeline={result.timeline} />
          </div>
        )}
      </div>
    </div>
  );
}

function FrameTimeline({ timeline }: { timeline: VideoDetectionResponse["timeline"] }) {
  if (timeline.length === 0) return null;
  return (
    <div>
      <p className="text-xs uppercase tracking-wide text-[#707A74] mb-2">Fake-probability timeline</p>
      <div className="flex items-end gap-0.5 h-20 rounded-lg bg-[#F2F5EF] p-2">
        {timeline.map((frame) => (
          <div
            key={frame.frame_index}
            title={`t=${frame.timestamp_sec}s · ${(frame.fake_probability * 100).toFixed(0)}% fake`}
            className={`flex-1 rounded-sm ${frame.fake_probability >= 0.5 ? "bg-[#B94B45]" : "bg-[#28734D]"}`}
            style={{ height: `${Math.max(6, frame.fake_probability * 100)}%`, opacity: frame.face_detected ? 1 : 0.3 }}
          />
        ))}
      </div>
    </div>
  );
}
