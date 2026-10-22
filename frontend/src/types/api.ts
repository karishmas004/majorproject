export interface FaceRegion {
  x: number;
  y: number;
  width: number;
  height: number;
}

export interface ImageDetectionResponse {
  label: "REAL" | "FAKE";
  confidence: number;
  fake_probability: number;
  real_probability: number;
  processing_time_ms: number;
  face_detected: boolean;
  face_region: FaceRegion | null;
  heatmap_base64: string | null;
  original_base64: string;
}

export interface FrameScore {
  frame_index: number;
  timestamp_sec: number;
  fake_probability: number;
  face_detected: boolean;
}

export interface VideoDetectionResponse {
  label: "REAL" | "FAKE";
  confidence: number;
  fake_probability: number;
  real_probability: number;
  frames_analyzed: number;
  frames_with_face: number;
  processing_time_ms: number;
  timeline: FrameScore[];
  most_suspicious_frame_base64: string | null;
  most_suspicious_heatmap_base64: string | null;
}

export interface AudioDetectionResponse {
  label: "REAL" | "FAKE";
  confidence: number;
  fake_probability: number;
  real_probability: number;
  processing_time_ms: number;
  duration_sec: number;
  waveform_base64: string;
  spectrogram_base64: string;
  cnn_label: "REAL" | "FAKE";
  cnn_fake_probability: number;
  acoustic_label: "REAL" | "FAKE";
  acoustic_fake_probability: number;
  hybrid_prediction: boolean;
}

export interface ApiErrorResponse {
  error: string;
  detail: string;
}

export interface WebcamFrameResult {
  face_detected: boolean;
  label: "REAL" | "FAKE" | null;
  confidence: number | null;
  fake_probability?: number;
  face_region?: FaceRegion;
  fps: number;
  processing_time_ms: number;
  error?: string;
  detail?: string;
}
