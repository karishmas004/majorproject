import { useEffect, useRef, useState } from "react";
import { Camera, CameraOff, Gauge, Clock } from "lucide-react";
import ErrorBanner from "./ErrorBanner";
import { webcamSocketUrl } from "../api/client";
import type { WebcamFrameResult } from "../types/api";

const CAPTURE_INTERVAL_MS = 300; // ~3 inferences/sec — keeps the UI responsive without flooding the socket

export default function WebcamDetection() {
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const wsRef = useRef<WebSocket | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const intervalRef = useRef<number | null>(null);

  const [running, setRunning] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<WebcamFrameResult | null>(null);

  async function start() {
    setError(null);
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ video: { width: 640, height: 480 } });
      streamRef.current = stream;
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        await videoRef.current.play();
      }

      const ws = new WebSocket(webcamSocketUrl());
      wsRef.current = ws;

      ws.onmessage = (event) => {
        const data: WebcamFrameResult = JSON.parse(event.data);
        setResult(data);
      };
      ws.onerror = () => setError("Lost connection to the detection server. Is the backend running?");
      ws.onclose = () => setError((prev) => prev ?? "Detection stream closed.");

      ws.onopen = () => {
        setRunning(true);
        intervalRef.current = window.setInterval(captureAndSend, CAPTURE_INTERVAL_MS);
      };
    } catch (err) {
      setError(
        err instanceof DOMException
          ? "Camera unavailable: permission denied or no camera found."
          : "Could not access the webcam."
      );
    }
  }

  function captureAndSend() {
    const video = videoRef.current;
    const canvas = canvasRef.current;
    const ws = wsRef.current;
    if (!video || !canvas || !ws || ws.readyState !== WebSocket.OPEN) return;

    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
    const frame = canvas.toDataURL("image/jpeg", 0.7);
    ws.send(JSON.stringify({ frame }));
  }

  function stop() {
    if (intervalRef.current) window.clearInterval(intervalRef.current);
    wsRef.current?.close();
    streamRef.current?.getTracks().forEach((t) => t.stop());
    wsRef.current = null;
    streamRef.current = null;
    setRunning(false);
    setResult(null);
  }

  useEffect(() => () => stop(), []);

  const borderColor = !result?.face_detected ? "border-[#DCE3DA]" : result.label === "FAKE" ? "border-[#B94B45]" : "border-[#28734D]";

  return (
    <div className="grid gap-6 lg:grid-cols-2">
      <div className="space-y-4">
        <div className={`relative rounded-2xl overflow-hidden border-4 transition-colors ${borderColor} bg-black aspect-video`}>
          <video ref={videoRef} className="w-full h-full object-cover" muted playsInline />
          <canvas ref={canvasRef} className="hidden" />

          {running && result?.face_detected && result.label && (
            <div
              className={`absolute top-3 left-3 rounded-lg px-3 py-1.5 text-sm font-bold backdrop-blur ${
                result.label === "FAKE" ? "bg-[#B94B45]/80 text-white" : "bg-[#28734D]/80 text-white"
              }`}
            >
              {result.label} · {((result.confidence ?? 0) * 100).toFixed(0)}%
            </div>
          )}

          {!running && (
            <div className="absolute inset-0 flex items-center justify-center text-[#707A74]">
              <Camera className="h-10 w-10" />
            </div>
          )}
        </div>

        <div className="flex gap-3">
          {!running ? (
            <button
              onClick={start}
              className="flex items-center gap-2 rounded-xl bg-[#164D3B]/90 hover:bg-[#164D3B] px-5 py-2.5 font-semibold text-white transition-colors"
            >
              <Camera className="h-4 w-4" /> Start Camera
            </button>
          ) : (
            <button
              onClick={stop}
              className="flex items-center gap-2 rounded-xl bg-[#B94B45]/90 hover:bg-[#B94B45] px-5 py-2.5 font-semibold text-white transition-colors"
            >
              <CameraOff className="h-4 w-4" /> Stop Camera
            </button>
          )}
        </div>

        {error && <ErrorBanner message={error} />}
      </div>

      <div className="rounded-2xl border border-[#DCE3DA] bg-bg-card p-5 min-h-[320px]">
        <h3 className="font-semibold text-[#24352E] mb-4">Live Stats</h3>
        {!running && <p className="text-sm text-[#707A74]">Start the camera to see live detection stats.</p>}
        {running && (
          <div className="space-y-4">
            <div className="flex items-center gap-2 text-sm text-[#5F6868]">
              <Gauge className="h-4 w-4" /> {result?.fps ?? 0} FPS
            </div>
            <div className="flex items-center gap-2 text-sm text-[#5F6868]">
              <Clock className="h-4 w-4" /> {result?.processing_time_ms?.toFixed(0) ?? 0}ms per frame
            </div>
            <div className="text-sm text-[#5F6868]">
              Face detected: <span className="text-[#24352E]">{result?.face_detected ? "Yes" : "No"}</span>
            </div>
            {result?.face_detected && result.label && (
              <div className="text-sm text-[#5F6868]">
                Fake probability:{" "}
                <span className="text-[#24352E]">{((result.fake_probability ?? 0) * 100).toFixed(1)}%</span>
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
}
