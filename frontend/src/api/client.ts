import axios, { AxiosError } from "axios";
import type { ImageDetectionResponse, VideoDetectionResponse, AudioDetectionResponse, ApiErrorResponse } from "../types/api";

const api = axios.create({
  baseURL: "/api",
  timeout: 120_000, // video analysis can legitimately take a while
});

export class ApiError extends Error {
  code: string;
  constructor(code: string, detail: string) {
    super(detail);
    this.code = code;
  }
}

function unwrapError(err: unknown): never {
  const axiosErr = err as AxiosError<ApiErrorResponse>;
  if (axiosErr.response?.data) {
    const { error, detail } = axiosErr.response.data;
    throw new ApiError(error ?? "unknown_error", detail ?? axiosErr.message);
  }
  throw new ApiError("network_error", axiosErr.message ?? "Could not reach the API. Is the backend running?");
}

export async function detectImage(file: File): Promise<ImageDetectionResponse> {
  const form = new FormData();
  form.append("file", file);
  try {
    const { data } = await api.post<ImageDetectionResponse>("/image/detect", form);
    return data;
  } catch (err) {
    unwrapError(err);
  }
}

export async function detectVideo(file: File): Promise<VideoDetectionResponse> {
  const form = new FormData();
  form.append("file", file);
  try {
    const { data } = await api.post<VideoDetectionResponse>("/video/detect", form);
    return data;
  } catch (err) {
    unwrapError(err);
  }
}

export async function detectAudio(file: File): Promise<AudioDetectionResponse> {
  const form = new FormData();
  form.append("file", file);
  try {
    const { data } = await api.post<AudioDetectionResponse>("/audio/detect", form);
    return data;
  } catch (err) {
    unwrapError(err);
  }
}

export function webcamSocketUrl(): string {
  const protocol = window.location.protocol === "https:" ? "wss" : "ws";
  // Vite dev proxy forwards /api websocket traffic to the FastAPI backend.
  return `${protocol}://${window.location.host}/api/webcam/stream`;
}
