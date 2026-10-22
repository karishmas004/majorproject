import { useState } from "react";
import { ImageIcon, Clock, ScanFace } from "lucide-react";
import UploadDropzone from "./UploadDropzone";
import ConfidenceBar from "./ConfidenceBar";
import ErrorBanner from "./ErrorBanner";
import { Spinner } from "./Loader";
import { detectImage, ApiError } from "../api/client";
import type { ImageDetectionResponse } from "../types/api";

export default function ImageDetection() {
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<ImageDetectionResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [showHeatmap, setShowHeatmap] = useState(true);

  async function handleFile(file: File) {
    setLoading(true);
    setError(null);
    setResult(null);
    try {
      const data = await detectImage(file);
      setResult(data);
    } catch (err) {
      setError(err instanceof ApiError ? err.message : "Something went wrong analyzing this image.");
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="grid gap-6 lg:grid-cols-2">
      <div className="space-y-4">
        <UploadDropzone
          accept=".jpg,.jpeg,.png,.bmp,.webp"
          label="Drop an image, or click to browse"
          hint="JPG, JPEG, PNG, BMP, WEBP · up to 15MB"
          onFileSelected={handleFile}
          disabled={loading}
        />
        {error && <ErrorBanner message={error} />}
      </div>

      <div className="rounded-2xl border border-[#DCE3DA] bg-bg-card p-5 min-h-[320px]">
        {loading && <Spinner message="Detecting faces and running EfficientNetV2 inference..." />}

        {!loading && !result && !error && (
          <div className="flex h-full flex-col items-center justify-center gap-2 py-14 text-[#707A74]">
            <ImageIcon className="h-8 w-8" />
            <p className="text-sm">Results will appear here</p>
          </div>
        )}

        {!loading && result && (
          <div className="space-y-5">
            <div className="flex items-center justify-between">
              <h3 className="font-semibold text-[#24352E]">Detection Result</h3>
              <div className="flex items-center gap-1 text-xs text-[#707A74]">
                <Clock className="h-3.5 w-3.5" /> {result.processing_time_ms.toFixed(0)}ms
              </div>
            </div>

            <div className="relative rounded-xl overflow-hidden border border-[#DCE3DA]">
              <img
                src={`data:image/jpeg;base64,${showHeatmap && result.heatmap_base64 ? result.heatmap_base64 : result.original_base64}`}
                alt="Analyzed"
                className="w-full object-contain max-h-72"
              />
              {result.heatmap_base64 && (
                <button
                  onClick={() => setShowHeatmap((s) => !s)}
                  className="absolute bottom-2 right-2 rounded-lg bg-black/60 px-3 py-1.5 text-xs text-white backdrop-blur hover:bg-black/80"
                >
                  {showHeatmap ? "Show Original" : "Show Grad-CAM Heatmap"}
                </button>
              )}
            </div>

            <ConfidenceBar
              label={result.label}
              confidence={result.confidence}
              fakeProbability={result.fake_probability}
              realProbability={result.real_probability}
            />

            <div className="flex items-center gap-2 text-sm text-[#5F6868]">
              <ScanFace className="h-4 w-4" />
              {result.face_detected
                ? `Face region: ${result.face_region?.width}×${result.face_region?.height}px`
                : "No face detected"}
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
